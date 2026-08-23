const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const batch_dispatch = @import("gpu/batch_dispatch.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

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

    var gpu_model = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .mixed, 0.001);
    defer gpu_model.deinit();

    var kernel_buf: [4096]u8 = undefined;
    const kernel_bytes = try std.fs.cwd().readFile("tui/PROMPT_KERNEL.md", &kernel_buf);
    var full_prompt_buf: [8192]u8 = undefined;
    const prompt = try std.fmt.bufPrint(&full_prompt_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHello, how are you today?<turn|>\n<|turn>model\n<|channel>thought\n", .{kernel_bytes});
    const tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(tokens);

    const H = config.hidden_size;
    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring1 = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring1.deinit();

    var slots = try allocator.alloc(u32, tokens.len);
    defer allocator.free(slots);
    for (tokens, 0..) |_, i| slots[i] = @intCast(ring1.activateSlot(0, i));

    const bp = gpu_model.batch_prefill_ctx.?;
    const batch_logits = try allocator.alloc(f32, config.vocab_size);
    defer allocator.free(batch_logits);
    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, &config, m.layers, tokens, m.embed_tokens, slots, batch_logits);

    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));
    for (0..config.num_hidden_layers) |l_idx| {
        const kv_d = m.layers[l_idx].kv_dim;
        const b_k = gpu_model.layers[l_idx].buf_k_cache.asSlice(f32)[0 .. tokens.len * kv_d];
        const s_k = try allocator.alloc(f32, tokens.len * kv_d);
        defer allocator.free(s_k);

        // Run serial for layer l_idx
        var ring_l = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
        defer ring_l.deinit();
        var sc_l = try model.ForwardScratch.init(allocator, config);
        defer sc_l.deinit(allocator);

        for (tokens, 0..) |t, i| {
            const emb_offset = @as(usize, t) * H;
            for (sc_l.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
            const slot_idx = ring_l.activateSlot(0, i);
            const active_count = ring_l.getActiveSlots(0, i, sc_l.active_slots);
            _ = @import("gpu/model_dispatch.zig").gpuDispatchForwardToken(&gpu_model, &config, m.layers, sc_l.x, sc_l.logits[0..0], i, slot_idx, sc_l.active_slots[0..active_count]);
        }
        @memcpy(s_k, gpu_model.layers[l_idx].buf_k_cache.asSlice(f32)[0 .. tokens.len * kv_d]);

        var max_diff: f32 = 0.0;
        for (b_k, s_k) |bk, sk| {
            const diff = @abs(bk - sk);
            if (diff > max_diff) max_diff = diff;
        }
        std.debug.print("Layer {d:2} K_cache max diff: {d:.6}\n", .{ l_idx, max_diff });
        if (l_idx >= 5) break;
    }
}
