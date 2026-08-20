const std = @import("std");
pub const types = @import("gpu/types.zig");
pub const types_dispatch = @import("gpu/types_dispatch.zig");
pub const vk_api = @import("gpu/vk_api.zig");
pub const context = @import("gpu/context.zig");
pub const buffer = @import("gpu/buffer.zig");
pub const pipeline = @import("gpu/pipeline.zig");
pub const shaders = @import("gpu/shaders.zig");
pub const gpu_kernels = @import("gpu/kernels.zig");
pub const descriptors = @import("gpu/descriptors.zig");
pub const model_dispatch = @import("gpu/model_dispatch.zig");
pub const model_gpu = @import("gpu/model_gpu.zig");
pub const quant = @import("quant.zig");

fn testCtx() !?context.GpuContext {
    return context.GpuContext.init(std.testing.allocator) catch |err| {
        if (err == error.VulkanLibraryNotFound or err == error.NoVulkanDevices) return null;
        return err;
    };
}

test "vulkan context initialization and AMD device selection" {
    const maybe_ctx = try testCtx();
    if (maybe_ctx == null) return;
    var ctx = maybe_ctx.?;
    defer ctx.deinit();
    try std.testing.expect(ctx.device_name.len > 0);
}

test "vulkan compute pipeline execution on AMD GPU" {
    const maybe_ctx = try testCtx();
    if (maybe_ctx == null) return;
    var ctx = maybe_ctx.?;
    defer ctx.deinit();

    const n: usize = 256;
    var buf_a = try buffer.GpuBuffer.init(&ctx, n * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_a.deinit();
    var buf_b = try buffer.GpuBuffer.init(&ctx, n * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_b.deinit();
    var buf_c = try buffer.GpuBuffer.init(&ctx, n * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_c.deinit();

    for (buf_a.asSlice(f32), 0..) |*a, i| a.* = @as(f32, @floatFromInt(i)) * 1.5;
    @memset(buf_b.asSlice(f32), 10.0);

    var pipe = try pipeline.ComputePipeline.init(&ctx, &shaders.VEC_ADD_SPIRV, 3, 0);
    defer pipe.deinit();
    try pipe.bindBuffers(&[_]*const buffer.GpuBuffer{ &buf_a, &buf_b, &buf_c });
    try pipe.dispatch(null, @intCast(n / 64), 1, 1);
    for (buf_c.asSlice(f32), 0..) |c, i| try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i)) * 1.5 + 10.0, c, 1e-4);
}

test "vulkan gemv compute execution (bf16, q4) on AMD GPU" {
    const maybe_ctx = try testCtx();
    if (maybe_ctx == null) return;
    var ctx = maybe_ctx.?;
    defer ctx.deinit();

    const m: usize = 64;
    const k: usize = 128;
    var buf_x = try buffer.GpuBuffer.init(&ctx, k * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_x.deinit();
    var buf_y = try buffer.GpuBuffer.init(&ctx, m * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_y.deinit();
    for (buf_x.asSlice(f32), 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 5)) * 0.25;

    const raw = try std.testing.allocator.alloc(u16, m * k);
    defer std.testing.allocator.free(raw);
    @memset(raw, 0x3F80);

    // BF16
    var buf_bf16 = try buffer.GpuBuffer.init(&ctx, m * k * 2, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_bf16.deinit();
    @memset(buf_bf16.asSlice(u16), 0x3F80);
    var p_bf16 = try pipeline.ComputePipeline.init(&ctx, &shaders.GEMV_BF16_SPIRV, 3, 8);
    defer p_bf16.deinit();
    try p_bf16.bindBuffers(&[_]*const buffer.GpuBuffer{ &buf_bf16, &buf_x, &buf_y });
    const pc = [_]u32{ @intCast(m), @intCast(k) };
    try p_bf16.dispatch(std.mem.sliceAsBytes(&pc), @intCast((m + 63) / 64), 1, 1);

    // Q4
    const q4_size = quant.getQuantizedSizeBytes(m, k, .q4);
    var buf_q4 = try buffer.GpuBuffer.init(&ctx, q4_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_q4.deinit();
    quant.quantizeMatrix(buf_q4.asSlice(u32), raw, m, k, .q4);
    var p_q4 = try pipeline.ComputePipeline.init(&ctx, &shaders.GEMV_Q4_SPIRV, 3, 8);
    defer p_q4.deinit();
    try p_q4.bindBuffers(&[_]*const buffer.GpuBuffer{ &buf_q4, &buf_x, &buf_y });
    try p_q4.dispatch(std.mem.sliceAsBytes(&pc), @intCast(m), 1, 1);
    for (buf_y.asSlice(f32)) |y| try std.testing.expect(y > 0);
}

test "vulkan fused swiglu and add_rmsnorm on AMD GPU" {
    const maybe_ctx = try testCtx();
    if (maybe_ctx == null) return;
    var ctx = maybe_ctx.?;
    defer ctx.deinit();

    const dim: usize = 128;
    var buf_g = try buffer.GpuBuffer.init(&ctx, dim * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_g.deinit();
    var buf_u = try buffer.GpuBuffer.init(&ctx, dim * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_u.deinit();
    var buf_o = try buffer.GpuBuffer.init(&ctx, dim * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_o.deinit();

    for (buf_g.asSlice(f32), 0..) |*g, i| g.* = @as(f32, @floatFromInt(i)) * 0.1 - 2.0;
    @memset(buf_u.asSlice(f32), 1.5);

    var p_swi = try pipeline.ComputePipeline.init(&ctx, &shaders.FUSED_SWIGLU_SPIRV, 3, 4);
    defer p_swi.deinit();
    try p_swi.bindBuffers(&[_]*const buffer.GpuBuffer{ &buf_g, &buf_u, &buf_o });
    const pc = [_]u32{@intCast(dim)};
    try p_swi.dispatch(std.mem.sliceAsBytes(&pc), 2, 1, 1);

    var buf_w = try buffer.GpuBuffer.init(&ctx, dim * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_w.deinit();
    @memset(buf_w.asSlice(f32), 2.0);

    var p_norm = try pipeline.ComputePipeline.init(&ctx, &shaders.FUSED_ADD_RMSNORM_SPIRV, 4, 8);
    defer p_norm.deinit();
    try p_norm.bindBuffers(&[_]*const buffer.GpuBuffer{ &buf_g, &buf_u, &buf_w, &buf_o });
    const pc_norm = extern struct { h: u32 = @intCast(dim), eps: f32 = 1e-6 }{};
    try p_norm.dispatch(std.mem.asBytes(&pc_norm), 1, 1, 1);
    for (buf_o.asSlice(f32)) |o| try std.testing.expect(o > 0);
}
