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

test "vulkan context initialization and AMD device selection" {
    var ctx = context.GpuContext.init(std.testing.allocator) catch |err| {
        if (err == error.VulkanLibraryNotFound or err == error.NoVulkanDevices) return;
        return err;
    };
    defer ctx.deinit();

    const name_len = std.mem.indexOfScalar(u8, &ctx.device_name, 0) orelse ctx.device_name.len;
    try std.testing.expect(name_len > 0);

    var buf = try buffer.GpuBuffer.init(&ctx, 1024 * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf.deinit();
    const slice = buf.asSlice(f32);
    slice[0] = 42.0;
    try std.testing.expectEqual(@as(f32, 42.0), slice[0]);
}

test "vulkan compute pipeline execution on AMD GPU" {
    var ctx = context.GpuContext.init(std.testing.allocator) catch |err| {
        if (err == error.VulkanLibraryNotFound or err == error.NoVulkanDevices) return;
        return err;
    };
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

    const bufs = [_]*const buffer.GpuBuffer{ &buf_a, &buf_b, &buf_c };
    try pipe.bindBuffers(&bufs);
    try pipe.dispatch(null, @intCast(n / 64), 1, 1);

    for (buf_c.asSlice(f32), 0..) |c, i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i)) * 1.5 + 10.0, c, 1e-4);
    }
}

test "vulkan gemv_bf16 compute execution on AMD GPU" {
    var ctx = context.GpuContext.init(std.testing.allocator) catch |err| {
        if (err == error.VulkanLibraryNotFound or err == error.NoVulkanDevices) return;
        return err;
    };
    defer ctx.deinit();

    const m_rows: usize = 64;
    const k_cols: usize = 128;
    var buf_w = try buffer.GpuBuffer.init(&ctx, m_rows * (k_cols / 2) * @sizeOf(u32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_w.deinit();
    var buf_x = try buffer.GpuBuffer.init(&ctx, k_cols * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_x.deinit();
    var buf_y = try buffer.GpuBuffer.init(&ctx, m_rows * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_y.deinit();

    for (buf_x.asSlice(f32), 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 5)) * 0.25;
    @memset(buf_w.asSlice(u32), 0x3F803F80); // 1.0 in bf16

    var pipe = try pipeline.ComputePipeline.init(&ctx, &shaders.GEMV_BF16_SPIRV, 3, 8);
    defer pipe.deinit();
    const bufs = [_]*const buffer.GpuBuffer{ &buf_w, &buf_x, &buf_y };
    try pipe.bindBuffers(&bufs);
    const pc = [_]u32{ @intCast(m_rows), @intCast(k_cols) };
    try pipe.dispatch(std.mem.sliceAsBytes(&pc), 1, 1, 1);

    var expected: f32 = 0.0;
    for (buf_x.asSlice(f32)) |x| expected += x;
    for (buf_y.asSlice(f32)) |y| try std.testing.expectApproxEqAbs(expected, y, 1e-3);
}

test "vulkan gemv_q8 compute execution on AMD GPU" {
    var ctx = context.GpuContext.init(std.testing.allocator) catch |err| {
        if (err == error.VulkanLibraryNotFound or err == error.NoVulkanDevices) return;
        return err;
    };
    defer ctx.deinit();

    const m_rows: usize = 64;
    const k_cols: usize = 128;
    const q8_size = quant.getQuantizedSizeBytes(m_rows, k_cols, .q8);
    var buf_w = try buffer.GpuBuffer.init(&ctx, q8_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_w.deinit();
    var buf_x = try buffer.GpuBuffer.init(&ctx, k_cols * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_x.deinit();
    var buf_y = try buffer.GpuBuffer.init(&ctx, m_rows * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_y.deinit();

    for (buf_x.asSlice(f32), 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 5)) * 0.25;

    // Fill BF16 source with 1.0 and quantize to Q8
    const bf16_src = try std.testing.allocator.alloc(u16, m_rows * k_cols);
    defer std.testing.allocator.free(bf16_src);
    @memset(bf16_src, 0x3F80); // 1.0 in bf16
    quant.quantizeMatrix(buf_w.asSlice(u32), bf16_src, m_rows, k_cols, .q8);

    var pipe = try pipeline.ComputePipeline.init(&ctx, &shaders.GEMV_Q8_SPIRV, 3, 8);
    defer pipe.deinit();
    const bufs = [_]*const buffer.GpuBuffer{ &buf_w, &buf_x, &buf_y };
    try pipe.bindBuffers(&bufs);
    const pc = [_]u32{ @intCast(m_rows), @intCast(k_cols) };
    try pipe.dispatch(std.mem.sliceAsBytes(&pc), 1, 1, 1);

    var expected: f32 = 0.0;
    for (buf_x.asSlice(f32)) |x| expected += x;
    for (buf_y.asSlice(f32)) |y| try std.testing.expectApproxEqAbs(expected, y, 0.05);
}

test "vulkan gemv_q4 compute execution on AMD GPU" {
    var ctx = context.GpuContext.init(std.testing.allocator) catch |err| {
        if (err == error.VulkanLibraryNotFound or err == error.NoVulkanDevices) return;
        return err;
    };
    defer ctx.deinit();

    const m_rows: usize = 64;
    const k_cols: usize = 128;
    const q4_size = quant.getQuantizedSizeBytes(m_rows, k_cols, .q4);
    var buf_w = try buffer.GpuBuffer.init(&ctx, q4_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_w.deinit();
    var buf_x = try buffer.GpuBuffer.init(&ctx, k_cols * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_x.deinit();
    var buf_y = try buffer.GpuBuffer.init(&ctx, m_rows * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_y.deinit();

    for (buf_x.asSlice(f32), 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 5)) * 0.25;

    const bf16_src = try std.testing.allocator.alloc(u16, m_rows * k_cols);
    defer std.testing.allocator.free(bf16_src);
    @memset(bf16_src, 0x3F80); // 1.0 in bf16
    quant.quantizeMatrix(buf_w.asSlice(u32), bf16_src, m_rows, k_cols, .q4);

    var pipe = try pipeline.ComputePipeline.init(&ctx, &shaders.GEMV_Q4_SPIRV, 3, 8);
    defer pipe.deinit();
    const bufs = [_]*const buffer.GpuBuffer{ &buf_w, &buf_x, &buf_y };
    try pipe.bindBuffers(&bufs);
    const pc = [_]u32{ @intCast(m_rows), @intCast(k_cols) };
    try pipe.dispatch(std.mem.sliceAsBytes(&pc), @intCast(m_rows), 1, 1);

    var expected: f32 = 0.0;
    for (buf_x.asSlice(f32)) |x| expected += x;
    for (buf_y.asSlice(f32)) |y| try std.testing.expectApproxEqAbs(expected, y, 0.1);
}

test "vulkan fused swiglu compute execution on AMD GPU" {
    var ctx = context.GpuContext.init(std.testing.allocator) catch |err| {
        if (err == error.VulkanLibraryNotFound or err == error.NoVulkanDevices) return;
        return err;
    };
    defer ctx.deinit();

    const dim: usize = 128;
    var buf_gate = try buffer.GpuBuffer.init(&ctx, dim * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_gate.deinit();
    var buf_up = try buffer.GpuBuffer.init(&ctx, dim * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_up.deinit();
    var buf_out = try buffer.GpuBuffer.init(&ctx, dim * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_out.deinit();

    for (buf_gate.asSlice(f32), 0..) |*g, i| g.* = @as(f32, @floatFromInt(i)) * 0.1 - 2.0;
    @memset(buf_up.asSlice(f32), 1.5);

    var pipe = try pipeline.ComputePipeline.init(&ctx, &shaders.FUSED_SWIGLU_SPIRV, 3, 4);
    defer pipe.deinit();
    const bufs = [_]*const buffer.GpuBuffer{ &buf_gate, &buf_up, &buf_out };
    try pipe.bindBuffers(&bufs);
    const pc = [_]u32{@intCast(dim)};
    try pipe.dispatch(std.mem.sliceAsBytes(&pc), 2, 1, 1);

    for (0..dim) |i| {
        const g = buf_gate.asSlice(f32)[i];
        const silu = g * (1.0 / (1.0 + @exp(-g)));
        try std.testing.expectApproxEqAbs(silu * 1.5, buf_out.asSlice(f32)[i], 1e-4);
    }
}
