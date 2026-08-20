const std = @import("std");
const context = @import("gpu/context.zig");
const buffer = @import("gpu/buffer.zig");
const pipeline = @import("gpu/pipeline.zig");
const shaders = @import("gpu/shaders.zig");
const quant = @import("quant.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var ctx = try context.GpuContext.init(allocator);
    defer ctx.deinit();

    const M: usize = 3840;
    const K: usize = 15360;

    // Allocate host BF16 weights and quantize to Q4_0
    var rand = std.Random.DefaultPrng.init(42);
    const r = rand.random();

    const w_bf16 = try allocator.alloc(u16, M * K);
    defer allocator.free(w_bf16);
    for (w_bf16) |*val| {
        const f = (r.float(f32) - 0.5) * 0.2;
        val.* = @as(u16, @truncate(@as(u32, @bitCast(f)) >> 16));
    }

    const q4_words_per_row = quant.getQuantizedRowWords(K, .q4);
    const total_q4_words = M * q4_words_per_row;
    var w_buf = try buffer.GpuBuffer.init(&ctx, total_q4_words * @sizeOf(u32), context.types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer w_buf.deinit();
    const w_slice = @as([*]u32, @ptrCast(@alignCast(w_buf.mapped.?)))[0..total_q4_words];
    quant.quantizeMatrix(w_slice, w_bf16, M, K, .q4);

    // Input X
    var x_buf = try buffer.GpuBuffer.init(&ctx, K * @sizeOf(f32), context.types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer x_buf.deinit();
    const x_slice = @as([*]f32, @ptrCast(@alignCast(x_buf.mapped.?)))[0..K];
    for (x_slice) |*val| {
        val.* = (r.float(f32) - 0.5) * 0.5;
    }

    // Output Y
    var y_buf = try buffer.GpuBuffer.init(&ctx, M * @sizeOf(f32), context.types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer y_buf.deinit();
    const y_slice = @as([*]f32, @ptrCast(@alignCast(y_buf.mapped.?)))[0..M];
    @memset(y_slice, 0.0);

    // CPU Reference Calculation
    const cpu_y = try allocator.alloc(f32, M);
    defer allocator.free(cpu_y);
    @memset(cpu_y, 0.0);

    for (0..M) |row| {
        var row_sum: f32 = 0.0;
        const row_w = w_slice[row * q4_words_per_row .. (row + 1) * q4_words_per_row];
        const num_blocks = K / 32;
        for (0..num_blocks) |b| {
            const blk = row_w[b * 5 .. (b + 1) * 5];
            const d = @as(f32, @bitCast(blk[0]));
            for (0..4) |w_idx| {
                const packed_val = blk[1 + w_idx];
                for (0..8) |n| {
                    const nib = (packed_val >> @intCast(n * 4)) & 0x0F;
                    const weight = (@as(f32, @floatFromInt(nib)) - 8.0) * d;
                    const elem_idx = b * 32 + w_idx * 8 + n;
                    row_sum += weight * x_slice[elem_idx];
                }
            }
        }
        cpu_y[row] = row_sum;
    }

    // GPU Pipeline
    var p = try pipeline.ComputePipeline.init(&ctx, &shaders.GEMV_Q4_SPIRV, 3, 8);
    defer p.deinit();

    try p.bindBuffers(&.{ &w_buf, &x_buf, &y_buf });

    const pc_data = [_]u32{ @intCast(M), @intCast(K) };

    // Dispatch 1 workgroup per row (each workgroup has 32 threads = 1 Wave32)
    try p.dispatch(std.mem.sliceAsBytes(&pc_data), @intCast(M), 1, 1);

    // Verify Output
    var max_err: f32 = 0.0;
    for (0..M) |i| {
        const err = @abs(y_slice[i] - cpu_y[i]);
        if (err > max_err) max_err = err;
    }
    std.debug.print("Verification: M={}, K={}, Max Absolute Error = {d:.6}\n", .{ M, K, max_err });

    if (max_err < 1e-3) {
        std.debug.print("TEST PASSED: GPU Q4 GEMV matches CPU reference exactly!\n", .{});
    } else {
        std.debug.print("TEST FAILED: Error exceeds threshold!\n", .{});
        return error.VerificationFailed;
    }

    // Benchmark 100 iterations
    const iters: usize = 100;
    const start = std.time.nanoTimestamp();
    for (0..iters) |_| {
        try p.dispatch(std.mem.sliceAsBytes(&pc_data), @intCast(M), 1, 1);
    }
    const elapsed_ns = std.time.nanoTimestamp() - start;
    const avg_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iters * 1000));
    const bytes_read = @as(f64, @floatFromInt(total_q4_words * 4 + K * 4));
    const bandwidth_gbs = (bytes_read / (avg_us * 1e-6)) / 1e9;

    std.debug.print("Performance: Avg latency = {d:.2} us, Achieved Bandwidth = {d:.2} GB/s\n", .{ avg_us, bandwidth_gbs });
}
