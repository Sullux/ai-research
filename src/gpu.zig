const std = @import("std");
pub const types = @import("gpu/types.zig");
pub const types_dispatch = @import("gpu/types_dispatch.zig");
pub const vk_api = @import("gpu/vk_api.zig");
pub const context = @import("gpu/context.zig");
pub const buffer = @import("gpu/buffer.zig");
pub const pipeline = @import("gpu/pipeline.zig");
pub const shaders = @import("gpu/shaders.zig");

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
    try std.testing.expectEqual(@as(usize, 1024), slice.len);
    slice[0] = 42.0;
    slice[1023] = 99.0;
    try std.testing.expectEqual(@as(f32, 42.0), slice[0]);
    try std.testing.expectEqual(@as(f32, 99.0), slice[1023]);
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

    const a_slice = buf_a.asSlice(f32);
    const b_slice = buf_b.asSlice(f32);
    const c_slice = buf_c.asSlice(f32);

    for (0..n) |i| {
        a_slice[i] = @as(f32, @floatFromInt(i)) * 1.5;
        b_slice[i] = 10.0;
        c_slice[i] = 0.0;
    }

    var pipe = try pipeline.ComputePipeline.init(&ctx, &shaders.VEC_ADD_SPIRV, 3, 0);
    defer pipe.deinit();

    const bufs = [_]*const buffer.GpuBuffer{ &buf_a, &buf_b, &buf_c };
    try pipe.bindBuffers(&bufs);

    // 256 elements / 64 threads per workgroup = 4 workgroups
    try pipe.dispatch(null, @intCast(n / 64), 1, 1);

    for (0..n) |i| {
        const expected = @as(f32, @floatFromInt(i)) * 1.5 + 10.0;
        try std.testing.expectApproxEqAbs(expected, c_slice[i], 1e-4);
    }
}
