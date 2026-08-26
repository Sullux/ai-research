const std = @import("std");
const model = @import("model/types.zig");
const loader = @import("model/loader.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const quiescence = @import("quiescence.zig");
const tokenizer = @import("tokenizer.zig");
const gpu = @import("gpu.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const model_dir = "../gemma-4-12B-it-qat-q4_0-unquantized";
    const config = try model.ModelConfig.loadFromJson(alloc, model_dir ++ "/config.json");
    var tok = try tokenizer.Tokenizer.loadFromJson(alloc, model_dir ++ "/tokenizer.json");
    defer tok.deinit();

    var st = try safetensors.SafeTensors.openDir(alloc, model_dir);
    defer st.deinit();

    var m = try loader.Model.loadFromSafeTensors(alloc, &st, config);
    defer m.deinit();

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring1 = try ring_buffer.DynamicRingBuffer.init(alloc, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring1.deinit();
    var ring2 = try ring_buffer.DynamicRingBuffer.init(alloc, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring2.deinit();

    var q_tracker = quiescence.QuiescenceTracker.init(.{ .threshold = 0.0 }, config.num_hidden_layers);
    var scratch1 = try model.ForwardScratch.init(alloc, config);
    defer scratch1.deinit(alloc);
    var scratch2 = try model.ForwardScratch.init(alloc, config);
    defer scratch2.deinit(alloc);

    var gctx = try gpu.context.GpuContext.init(alloc);
    defer gctx.deinit();

    var gpu_ctx1 = try gpu.model_gpu.GpuModelContext.init(alloc, &gctx, &m, config, .q4, 0.0);
    defer gpu_ctx1.deinit();
    var gpu_ctx2 = try gpu.model_gpu.GpuModelContext.init(alloc, &gctx, &m, config, .q4, 0.0);
    defer gpu_ctx2.deinit();

    var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = alloc });
    defer pool.deinit();

    const prompt = "<|turn>system\n<|think|>\nYou are a helpful AI assistant.<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n";
    const tokens = try tok.encode(alloc, prompt, true);
    defer alloc.free(tokens);

    // Run 1: Sequential forwardToken
    for (tokens, 0..) |t, i| {
        _ = m.forwardToken(&ring1, &scratch1, t, i, &pool, null, &q_tracker, &gpu_ctx1, i + 1 == tokens.len);
    }

    // Run 2: Batch prefill
    const slots = try alloc.alloc(u32, tokens.len);
    defer alloc.free(slots);
    for (slots, 0..) |*s, i| {
        s.* = @intCast(ring2.getSlotIndex(i));
        for (0..config.num_hidden_layers) |l| _ = ring2.activateSlot(l, i);
    }
    const logits_batch = try alloc.alloc(f32, config.vocab_size);
    defer alloc.free(logits_batch);

    try gpu.batch_dispatch.gpuDispatchPrefillBatch(
        gpu_ctx2.batch_prefill_ctx.?,
        &gpu_ctx2,
        &config,
        m.layers,
        tokens,
        m.embed_tokens,
        slots,
        0,
        0,
        logits_batch,
        null,
        null,
    );

    // Compare logits
    var max_diff: f32 = 0.0;
    var sum_diff: f64 = 0.0;
    for (scratch1.logits, logits_batch) |v1, v2| {
        const diff = @abs(v1 - v2);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
    }
    std.debug.print("Logits comparison over {d} tokens:\n", .{tokens.len});
    std.debug.print("  Max logit difference: {d:.6}\n", .{max_diff});
    std.debug.print("  Avg logit difference: {d:.6}\n", .{sum_diff / @as(f64, @floatFromInt(config.vocab_size))});

    // Top 5 from sequential:
    std.debug.print("\nTop 5 from sequential:\n", .{});
    for (0..5) |_| {}
}
