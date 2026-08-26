const std = @import("std");
const model = @import("model/types.zig");
const loader = @import("model/loader.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const gpu = @import("gpu.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const model_dir = "../gemma-4-12B-it-qat-q4_0-unquantized";
    const config = try model.ModelConfig.loadFromJson(alloc, model_dir ++ "/config.json");
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

    for (0..config.num_hidden_layers) |l| _ = ring.activateSlot(l, 0);
    const active_count = ring.getActiveSlots(0, 0, scratch.active_slots);

    // Warmup
    for (0..5) |_| {
        _ = gpu.model_dispatch.gpuDispatchForwardToken(&gpu_ctx, &config, m.layers, scratch.x, scratch.logits, 0, 0, scratch.active_slots[0..active_count]);
    }

    // Measure decode command buffer (with logits)
    const t0 = std.time.microTimestamp();
    const iters = 50;
    for (0..iters) |_| {
        _ = gpu.model_dispatch.gpuDispatchForwardToken(&gpu_ctx, &config, m.layers, scratch.x, scratch.logits, 0, 0, scratch.active_slots[0..active_count]);
    }
    const elapsed_with_logits = std.time.microTimestamp() - t0;
    const avg_with_logits = @as(f64, @floatFromInt(elapsed_with_logits)) / @as(f64, @floatFromInt(iters)) / 1000.0;

    // Measure prefill command buffer (without logits)
    const t1 = std.time.microTimestamp();
    for (0..iters) |_| {
        _ = gpu.model_dispatch.gpuDispatchForwardToken(&gpu_ctx, &config, m.layers, scratch.x, scratch.logits[0..0], 0, 0, scratch.active_slots[0..active_count]);
    }
    const elapsed_without_logits = std.time.microTimestamp() - t1;
    const avg_without_logits = @as(f64, @floatFromInt(elapsed_without_logits)) / @as(f64, @floatFromInt(iters)) / 1000.0;

    std.debug.print("48 Layers (No Logits): {d:.2} ms ({d:.2} tok/s)\n", .{ avg_without_logits, 1000.0 / avg_without_logits });
    std.debug.print("48 Layers + Vocab Head (With Logits): {d:.2} ms ({d:.2} tok/s)\n", .{ avg_with_logits, 1000.0 / avg_with_logits });
    std.debug.print("Vocab Head Projection (256k x 3840): {d:.2} ms\n", .{ avg_with_logits - avg_without_logits });
}
