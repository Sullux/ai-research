const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const quant = @import("quant.zig");
const gpu_dispatch = @import("gpu/model_dispatch.zig");

pub fn runSingleTokenTest(allocator: std.mem.Allocator, mode: quant.QuantMode, mode_name: []const u8) !void {
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
    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring.deinit();

    const prompt = "<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n<|channel>thought\n";
    const tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(tokens);

    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));
    for (tokens, 0..) |t, i| {
        const emb_offset = @as(usize, t) * H;
        for (scratch.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
        const slot_idx = ring.activateSlot(0, i);
        const active_count = ring.getActiveSlots(0, i, scratch.active_slots);
        const is_last = (i == tokens.len - 1);
        _ = gpu_dispatch.gpuDispatchForwardToken(&gpu_model, &config, m.layers, scratch.x, if (is_last) scratch.logits else scratch.logits[0..0], i, slot_idx, scratch.active_slots[0..active_count]);
    }

    std.debug.print("\n=== Single-token forward ({s}) ===\n", .{mode_name});
    var top10: [10]struct { id: u32, val: f32 } = undefined;
    for (&top10) |*t| t.* = .{ .id = 0, .val = -1e9 };
    for (gpu_model.buf_logits.asSlice(f32), 0..) |val, id| {
        for (0..10) |j| {
            if (val > top10[j].val) {
                var k: usize = 9;
                while (k > j) : (k -= 1) top10[k] = top10[k - 1];
                top10[j] = .{ .id = @intCast(id), .val = val };
                break;
            }
        }
    }
    for (top10) |t| std.debug.print("  token {} ('{s}'): {d:.3}\n", .{ t.id, tok.decode(t.id), t.val });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    try runSingleTokenTest(allocator, .none, "BF16");
    try runSingleTokenTest(allocator, .q8, "Q8_0");
    try runSingleTokenTest(allocator, .q4, "Q4_0");
}
