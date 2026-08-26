const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const sampler = @import("sampler.zig");
const batch_dispatch = @import("gpu/batch_dispatch.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const model_dir = "../gemma-4-12B-it-qat-q4_0-unquantized";
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

    var gpu_model = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .q4, 0.001);
    defer gpu_model.deinit();

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 384, 512, 96);
    defer ring.deinit();

    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    const p1 = "<|turn>user\nHow are you today?<turn|>\n<|turn>model\n<|channel>thought\n";
    const tok1 = try tok.encode(allocator, p1, true);
    defer allocator.free(tok1);

    var clock: usize = 0;
    const bp = gpu_model.batch_prefill_ctx.?;
    var slots1 = try allocator.alloc(u32, tok1.len);
    defer allocator.free(slots1);
    for (tok1, 0..) |_, i| {
        const c = clock + i;
        slots1[i] = @intCast(ring.getSlotIndex(c));
        for (0..config.num_hidden_layers) |l| _ = ring.activateSlot(l, c);
    }
    const logits1 = try allocator.alloc(f32, config.vocab_size);
    defer allocator.free(logits1);

    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, &config, m.layers, tok1, m.embed_tokens, slots1, clock, 0, logits1, null, null);
    clock += tok1.len;

    var samp = sampler.Sampler.init(42, 0.0, 1.0);
    var cur = samp.sample(logits1);

    // Warmup 5 tokens
    for (0..5) |_| {
        cur = m.forwardToken(&ring, &scratch, cur, clock, null, null, null, &gpu_model, false);
        clock += 1;
    }

    const NUM_TOKENS: usize = 100;
    const start_ns = std.time.nanoTimestamp();
    for (0..NUM_TOKENS) |_| {
        cur = m.forwardToken(&ring, &scratch, cur, clock, null, null, null, &gpu_model, false);
        clock += 1;
    }
    const end_ns = std.time.nanoTimestamp();

    const elapsed_ms = @as(f64, @floatFromInt(end_ns - start_ns)) / 1e6;
    const tok_per_sec = (@as(f64, @floatFromInt(NUM_TOKENS)) / elapsed_ms) * 1000.0;
    const ms_per_tok = elapsed_ms / @as(f64, @floatFromInt(NUM_TOKENS));

    std.debug.print("\n=== Pure Decode Benchmark (100 tokens) ===\n", .{});
    std.debug.print("Total Time: {d:.2} ms\n", .{elapsed_ms});
    std.debug.print("Per-Token Latency: {d:.2} ms\n", .{ms_per_tok});
    std.debug.print("Decode Speed: {d:.2} tok/s\n", .{tok_per_sec});
}
