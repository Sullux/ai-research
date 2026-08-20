const std = @import("std");
const context = @import("gpu/context.zig");
const buffer = @import("gpu/buffer.zig");
const pipeline = @import("gpu/pipeline.zig");
const shaders = @import("gpu/shaders.zig");
const quant = @import("quant.zig");
const kernels = @import("kernels.zig");

fn testGemvBf16(ctx: *const context.GpuContext, allocator: std.mem.Allocator) !void {
    const M: usize = 512;
    const K: usize = 1024;
    var rand = std.Random.DefaultPrng.init(101);
    const r = rand.random();

    const w_bf16 = try allocator.alloc(u16, M * K);
    defer allocator.free(w_bf16);
    for (w_bf16) |*v| {
        const f = (r.float(f32) - 0.5) * 0.1;
        v.* = @as(u16, @truncate(@as(u32, @bitCast(f)) >> 16));
    }

    var w_buf = try buffer.GpuBuffer.init(ctx, (M * K / 2) * @sizeOf(u32), context.types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer w_buf.deinit();
    @memcpy(@as([*]u16, @ptrCast(@alignCast(w_buf.mapped.?)))[0 .. M * K], w_bf16);

    var x_buf = try buffer.GpuBuffer.init(ctx, K * @sizeOf(f32), context.types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer x_buf.deinit();
    const x_slice = @as([*]f32, @ptrCast(@alignCast(x_buf.mapped.?)))[0..K];
    for (x_slice) |*v| v.* = (r.float(f32) - 0.5) * 0.5;

    var y_buf = try buffer.GpuBuffer.init(ctx, M * @sizeOf(f32), context.types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer y_buf.deinit();
    const y_slice = @as([*]f32, @ptrCast(@alignCast(y_buf.mapped.?)))[0..M];

    const cpu_y = try allocator.alloc(f32, M);
    defer allocator.free(cpu_y);
    const bf16_slice = @as([*]const kernels.tensor.bf16, @ptrCast(w_bf16.ptr))[0 .. M * K];
    kernels.gemv(cpu_y, x_slice, bf16_slice, M, K);

    var p = try pipeline.ComputePipeline.init(ctx, &shaders.GEMV_BF16_SPIRV, 3, 8);
    defer p.deinit();
    try p.bindBuffers(&.{ &w_buf, &x_buf, &y_buf });

    const pc_data = [_]u32{ @intCast(M), @intCast(K) };
    try p.dispatch(std.mem.sliceAsBytes(&pc_data), @intCast(M), 1, 1);

    var max_err: f32 = 0.0;
    for (0..M) |i| {
        const err = @abs(y_slice[i] - cpu_y[i]);
        if (err > max_err) max_err = err;
    }
    std.debug.print("GEMV BF16 Test: Max Error = {d:.6} -> {s}\n", .{ max_err, if (max_err < 1e-3) "PASS" else "FAIL" });
    if (max_err >= 1e-3) return error.Bf16Mismatch;
}

fn testRmsNorm(ctx: *const context.GpuContext, allocator: std.mem.Allocator) !void {
    const D: usize = 2048;
    const eps: f32 = 1e-6;
    var rand = std.Random.DefaultPrng.init(202);
    const r = rand.random();

    var x_buf = try buffer.GpuBuffer.init(ctx, D * @sizeOf(f32), context.types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer x_buf.deinit();
    const x_slice = @as([*]f32, @ptrCast(@alignCast(x_buf.mapped.?)))[0..D];
    for (x_slice) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    var w_buf = try buffer.GpuBuffer.init(ctx, D * @sizeOf(f32), context.types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer w_buf.deinit();
    const w_f32_slice = @as([*]f32, @ptrCast(@alignCast(w_buf.mapped.?)))[0..D];
    const w_bf16 = try allocator.alloc(u16, D);
    defer allocator.free(w_bf16);
    for (w_bf16) |*v| {
        const f = 1.0 + (r.float(f32) - 0.5) * 0.1;
        v.* = @as(u16, @truncate(@as(u32, @bitCast(f)) >> 16));
    }
    for (w_f32_slice, w_bf16) |*dst, src| dst.* = (kernels.tensor.bf16{ .bits = src }).toF32();

    var y_buf = try buffer.GpuBuffer.init(ctx, D * @sizeOf(f32), context.types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer y_buf.deinit();
    const y_slice = @as([*]f32, @ptrCast(@alignCast(y_buf.mapped.?)))[0..D];

    const cpu_y = try allocator.alloc(f32, D);
    defer allocator.free(cpu_y);
    const bf16_weights = @as([*]const kernels.tensor.bf16, @ptrCast(w_bf16.ptr))[0..D];
    kernels.rmsNorm(cpu_y, x_slice, bf16_weights, eps);

    var p = try pipeline.ComputePipeline.init(ctx, &shaders.RMSNORM_SPIRV, 3, 8);
    defer p.deinit();
    try p.bindBuffers(&.{ &x_buf, &w_buf, &y_buf });

    const Pc = extern struct { dim: u32, eps: f32 };
    const pc = Pc{ .dim = @intCast(D), .eps = eps };
    try p.dispatch(std.mem.asBytes(&pc), 1, 1, 1);

    var max_err: f32 = 0.0;
    for (0..D) |i| {
        const err = @abs(y_slice[i] - cpu_y[i]);
        if (err > max_err) max_err = err;
    }
    std.debug.print("RMSNorm Test: Max Error = {d:.6} -> {s}\n", .{ max_err, if (max_err < 1e-4) "PASS" else "FAIL" });
    if (max_err >= 1e-4) return error.RmsNormMismatch;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var ctx = try context.GpuContext.init(allocator);
    defer ctx.deinit();

    try testGemvBf16(&ctx, allocator);
    try testRmsNorm(&ctx, allocator);
    std.debug.print("\n=== ALL GPU COMPUTE KERNEL TESTS PASSED! ===\n", .{});
}
