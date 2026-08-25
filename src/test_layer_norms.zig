const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const quant = @import("quant.zig");
const types_dispatch = @import("gpu/types_dispatch.zig");

fn runLayerTrace(allocator: std.mem.Allocator, mode: quant.QuantMode, mode_name: []const u8) !void {
    const model_dir = "../gemma-4-12B-it";
    var config_path_buf: [512]u8 = undefined;
    const config = try model.ModelConfig.loadFromJson(allocator, try std.fmt.bufPrint(&config_path_buf, "{s}/config.json", .{model_dir}));
    var tok_path_buf: [512]u8 = undefined;
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{model_dir}));
    defer tok.deinit();

    var st = try safetensors.SafeTensors.openDir(allocator, model_dir);
    defer st.deinit();
    var m = try model.Model.loadFromSafeTensors(allocator, &st, config);
    defer m.deinit();

    var gpu_ctx = try context.GpuContext.init(allocator);
    defer gpu_ctx.deinit();
    var gpu_model = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, mode, 0.0);
    defer gpu_model.deinit();

    const H = config.hidden_size;
    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));
    var x: [3840]f32 = undefined;
    // Embed token 100
    const emb_off = 100 * H;
    for (&x, m.embed_tokens[emb_off .. emb_off + H]) |*dst, e| dst.* = e.toF32() * embed_scale;

    @memcpy(gpu_model.buf_x.asSlice(f32)[0..H], &x);

    std.debug.print("\n=== {s} Activation Norms ===\n", .{mode_name});
    var initial_norm: f32 = 0.0;
    for (x) |v| initial_norm += v * v;
    std.debug.print("Initial X norm: {d:.3}\n", .{@sqrt(initial_norm)});

    // Execute all 48 layers forward and inspect final norms
    const cmd = gpu_model.engine.cmd_buf;
    const begin_info = types_dispatch.VkCommandBufferBeginInfo{};
    _ = gpu_ctx.api.vkBeginCommandBuffer(cmd, &begin_info);

    for (gpu_model.layers, 0..) |l_gpu, i| {
        const l_cpu = m.layers[i];
        const eps = config.rms_norm_eps;
        const q_dim = l_cpu.q_dim;
        const kv_dim = l_cpu.kv_dim;
        const inter = config.intermediate_size;
        const head_dim: u32 = @intCast(l_cpu.head_dim);
        const rot_dim: u32 = @intCast(l_cpu.rotary_dim);
        const theta: f32 = if (l_cpu.layer_type == .full_attention) config.rope_theta_full else config.rope_theta;

        if (i == 0) {
            gpu_model.engine.recordRmsNorm(cmd, l_gpu.desc.input_norm, H, eps);
            gpu_model.engine.recordBarrier(cmd);
        }
        gpu_model.engine.recordGemv(cmd, l_gpu.desc.q_proj, q_dim, H);
        gpu_model.engine.recordGemv(cmd, l_gpu.desc.k_proj, kv_dim, H);
        gpu_model.engine.recordGemv(cmd, l_gpu.desc.v_proj, kv_dim, H);
        gpu_model.engine.recordBarrier(cmd);
        gpu_model.engine.recordQkvRope(cmd, l_gpu.desc.qkv_rope, config.num_attention_heads, l_cpu.num_kv_heads, head_dim, rot_dim, l_cpu.k_eq_v, theta, eps);
        gpu_model.engine.recordBarrier(cmd);
        gpu_model.engine.recordDecodeAttn(cmd, l_gpu.desc.attn, head_dim, kv_dim, 2, 1.0, config.num_attention_heads);
        gpu_model.engine.recordBarrier(cmd);
        gpu_model.engine.recordGemv(cmd, l_gpu.desc.o_proj, H, q_dim);
        gpu_model.engine.recordBarrier(cmd);
        gpu_model.engine.recordRmsNorm(cmd, l_gpu.desc.post_attn_norm, H, eps);
        gpu_model.engine.recordBarrier(cmd);
        gpu_model.engine.recordAddRmsNorm(cmd, l_gpu.desc.pre_ffn_norm, H, eps, 1.0);
        gpu_model.engine.recordBarrier(cmd);
        gpu_model.engine.recordGateUpSwiGlu(cmd, l_gpu.desc.gate_up_swiglu, inter, H);
        gpu_model.engine.recordBarrier(cmd);
        gpu_model.engine.recordGemvMlp(cmd, l_gpu.desc.down_proj, H, inter);
        gpu_model.engine.recordBarrier(cmd);
        gpu_model.engine.recordRmsNorm(cmd, l_gpu.desc.post_ffn_norm, H, eps);
        gpu_model.engine.recordBarrier(cmd);
        gpu_model.engine.recordAddRmsNorm(cmd, l_gpu.desc.post_ffn_add, H, eps, l_gpu.layer_scalar);
        gpu_model.engine.recordBarrier(cmd);
    }
    gpu_model.engine.recordGemvLogits(cmd, gpu_model.desc_logits, config.vocab_size, H, 0);
    gpu_model.engine.recordBarrier(cmd);

    _ = gpu_ctx.api.vkEndCommandBuffer(cmd);
    try gpu_model.engine.submitPreRecorded(cmd);

    var final_x_norm: f32 = 0.0;
    for (gpu_model.buf_x.asSlice(f32)[0..H]) |v| final_x_norm += v * v;
    var final_normed_norm: f32 = 0.0;
    for (gpu_model.buf_normed_x.asSlice(f32)[0..H]) |v| final_normed_norm += v * v;

    std.debug.print("Final X norm after 48 layers: {d:.3}\n", .{@sqrt(final_x_norm)});
    std.debug.print("Final normed_X norm: {d:.3}\n", .{@sqrt(final_normed_norm)});

    // Top 5 logits
    var top5: [5]struct { id: u32, val: f32 } = undefined;
    for (&top5) |*t| t.* = .{ .id = 0, .val = -1e9 };
    for (gpu_model.buf_logits.asSlice(f32), 0..) |val, id| {
        for (0..5) |j| {
            if (val > top5[j].val) {
                var k: usize = 4;
                while (k > j) : (k -= 1) top5[k] = top5[k - 1];
                top5[j] = .{ .id = @intCast(id), .val = val };
                break;
            }
        }
    }
    std.debug.print("Top 5 logits:\n", .{});
    for (top5) |t| std.debug.print("  token {} ('{s}'): {d:.3}\n", .{ t.id, tok.decode(t.id), t.val });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    try runLayerTrace(allocator, .none, "BF16");
    try runLayerTrace(allocator, .q8, "Q8_0");
    try runLayerTrace(allocator, .q4, "Q4_0");
}
