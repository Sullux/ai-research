const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const model_dispatch = @import("gpu/model_dispatch.zig");
const batch_dispatch = @import("gpu/batch_dispatch.zig");
const sampler = @import("sampler.zig");

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

    const prompt = "<|turn>user\nHello<turn|>\n<|turn>model\n";
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
    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, &config, m.layers, tokens, m.embed_tokens, slots, 0, 0, batch_logits);

    // Now run serial single-token forward for the same tokens
    var ring2 = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring2.deinit();
    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    const serial_logits = try allocator.alloc(f32, config.vocab_size);
    defer allocator.free(serial_logits);

    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));
    for (tokens, 0..) |t, i| {
        const emb_offset = @as(usize, t) * H;
        for (scratch.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
        const slot_idx = ring2.activateSlot(0, i);
        const active_count = ring2.getActiveSlots(0, i, scratch.active_slots);
        _ = model_dispatch.gpuDispatchForwardToken(&gpu_model, &config, m.layers, scratch.x, if (i == tokens.len - 1) serial_logits else serial_logits[0..0], i, slot_idx, scratch.active_slots[0..active_count]);
    }
    @memcpy(serial_logits, gpu_model.buf_logits.asSlice(f32)[0..config.vocab_size]);

    var max_diff: f32 = 0.0;
    for (batch_logits, serial_logits) |b, s| {
        const diff = @abs(b - s);
        if (diff > max_diff) max_diff = diff;
    }
    std.debug.print("Max logit diff (batch vs serial on 10 tokens): {d:.6}\n", .{max_diff});

    // Check top 5 tokens for both
    var s_sampler = sampler.Sampler.init(1337, 0.0, 0.95);
    const top_batch = s_sampler.sample(batch_logits);
    const top_serial = s_sampler.sample(serial_logits);
    std.debug.print("Top token batch:  {} ('{s}')\n", .{ top_batch, tok.decode(top_batch) });
    std.debug.print("Top token serial: {} ('{s}')\n", .{ top_serial, tok.decode(top_serial) });
}
