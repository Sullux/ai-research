const std = @import("std");
const model = @import("model/types.zig");
const loader = @import("model/loader.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const gpu = @import("gpu.zig");
const sampler_mod = @import("sampler.zig");

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

    var gctx = try gpu.context.GpuContext.init(alloc);
    defer gctx.deinit();

    var gpu_ctx = try gpu.model_gpu.GpuModelContext.init(alloc, &gctx, &m, config, .q4, 0.0);
    defer gpu_ctx.deinit();

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(alloc, config.num_hidden_layers, max_kv_dim, 32, 2048, 96);
    defer ring.deinit();

    var scratch = try model.ForwardScratch.init(alloc, config);
    defer scratch.deinit(alloc);

    const prompt_kernel = try std.fs.cwd().readFileAlloc(alloc, "tui/PROMPT_KERNEL.md", 1024 * 1024);
    defer alloc.free(prompt_kernel);

    var pb = std.ArrayList(u8).init(alloc);
    defer pb.deinit();
    try pb.appendSlice("<|turn>system\n<|think|>\n");
    try pb.appendSlice(std.mem.trim(u8, prompt_kernel, " \t\r\n"));
    try pb.appendSlice("\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n");

    const tokens = try tok.encode(alloc, pb.items, true);
    defer alloc.free(tokens);

    const slots = try alloc.alloc(u32, tokens.len);
    defer alloc.free(slots);
    for (slots, 0..) |*s_val, i| {
        s_val.* = @intCast(ring.getSlotIndex(i));
        for (0..config.num_hidden_layers) |l| _ = ring.activateSlot(l, i);
    }

    const logits_batch = try alloc.alloc(f32, config.vocab_size);
    defer alloc.free(logits_batch);

    try gpu.batch_dispatch.gpuDispatchPrefillBatch(
        gpu_ctx.batch_prefill_ctx.?,
        &gpu_ctx,
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

    var s = sampler_mod.Sampler.init(42, 0.7, 0.95);
    @memcpy(scratch.logits, logits_batch);
    var cur = s.sample(scratch.logits, null);
    var clock: usize = tokens.len;

    std.debug.print("Starting 50 decode steps...\n", .{});
    var step_times: [50]i64 = undefined;

    const decode_start = std.time.milliTimestamp();

    for (0..50) |step| {
        const t0 = std.time.microTimestamp();

        const H = config.hidden_size;
        const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));
        const emb_offset = @as(usize, cur) * H;
        for (scratch.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;

        for (0..config.num_hidden_layers) |l| _ = ring.activateSlot(l, clock);
        const slot_idx = ring.getSlotIndex(clock);
        const active_count = ring.getActiveSlots(0, clock, scratch.active_slots);

        _ = gpu.model_dispatch.gpuDispatchForwardToken(&gpu_ctx, &config, m.layers, scratch.x, scratch.logits, clock, slot_idx, scratch.active_slots[0..active_count]);
        @memcpy(scratch.logits, gpu_ctx.buf_logits.asSlice(f32)[0..config.vocab_size]);
        clock += 1;
        cur = s.sample(scratch.logits, null);

        const t1 = std.time.microTimestamp();
        step_times[step] = t1 - t0;
    }

    const decode_elapsed = std.time.milliTimestamp() - decode_start;
    var total_us: i64 = 0;
    for (step_times) |us| total_us += us;
    const avg_ms = @as(f64, @floatFromInt(total_us)) / 50.0 / 1000.0;
    const tps = 50000.0 / @as(f64, @floatFromInt(decode_elapsed));

    std.debug.print("50 tokens completed in {d} ms (avg {d:.2} ms/token = {d:.2} tok/s)\n", .{ decode_elapsed, avg_ms, tps });
    std.debug.print("First 5 steps: {d:.2}ms, {d:.2}ms, {d:.2}ms, {d:.2}ms, {d:.2}ms\n", .{
        @as(f64, @floatFromInt(step_times[0])) / 1000.0,
        @as(f64, @floatFromInt(step_times[1])) / 1000.0,
        @as(f64, @floatFromInt(step_times[2])) / 1000.0,
        @as(f64, @floatFromInt(step_times[3])) / 1000.0,
        @as(f64, @floatFromInt(step_times[4])) / 1000.0,
    });
}
