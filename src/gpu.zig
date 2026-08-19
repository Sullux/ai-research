const std = @import("std");
pub const types = @import("gpu/types.zig");
pub const types_dispatch = @import("gpu/types_dispatch.zig");
pub const vk_api = @import("gpu/vk_api.zig");
pub const context = @import("gpu/context.zig");
pub const buffer = @import("gpu/buffer.zig");
pub const pipeline = @import("gpu/pipeline.zig");
pub const shaders = @import("gpu/shaders.zig");
pub const gpu_kernels = @import("gpu/kernels.zig");
pub const model_dispatch = @import("gpu/model_dispatch.zig");
pub const model_gpu = @import("gpu/model_gpu.zig");

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

test "vulkan gemv_bf16 compute execution on AMD GPU" {
    var ctx = context.GpuContext.init(std.testing.allocator) catch |err| {
        if (err == error.VulkanLibraryNotFound or err == error.NoVulkanDevices) return;
        return err;
    };
    defer ctx.deinit();

    const m_rows: usize = 64;
    const k_cols: usize = 128; // even count for packed bf16 pairs

    // W: m_rows * (k_cols / 2) packed u32
    var buf_w = try buffer.GpuBuffer.init(&ctx, m_rows * (k_cols / 2) * @sizeOf(u32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_w.deinit();
    var buf_x = try buffer.GpuBuffer.init(&ctx, k_cols * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_x.deinit();
    var buf_y = try buffer.GpuBuffer.init(&ctx, m_rows * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_y.deinit();

    const w_slice = buf_w.asSlice(u32);
    const x_slice = buf_x.asSlice(f32);
    const y_slice = buf_y.asSlice(f32);

    for (x_slice, 0..) |*x, i| {
        x.* = @as(f32, @floatFromInt(i % 5)) * 0.25;
    }
    @memset(y_slice, 0);

    // Fill W with 1.0 (bf16 representation of 1.0 is 0x3F80)
    // Packed pair of 1.0 is (0x3F80 << 16) | 0x3F80 = 0x3F803F80
    @memset(w_slice, 0x3F803F80);

    var pipe = try pipeline.ComputePipeline.init(&ctx, &shaders.GEMV_BF16_SPIRV, 3, 8);
    defer pipe.deinit();

    const bufs = [_]*const buffer.GpuBuffer{ &buf_w, &buf_x, &buf_y };
    try pipe.bindBuffers(&bufs);

    const pc = [_]u32{ @intCast(m_rows), @intCast(k_cols) };
    try pipe.dispatch(std.mem.sliceAsBytes(&pc), 1, 1, 1);

    var expected_sum: f32 = 0.0;
    for (x_slice) |x| expected_sum += 1.0 * x;

    for (y_slice) |y| {
        try std.testing.expectApproxEqAbs(expected_sum, y, 1e-3);
    }
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

    const gate = buf_gate.asSlice(f32);
    const up = buf_up.asSlice(f32);
    const out = buf_out.asSlice(f32);

    for (0..dim) |i| {
        gate[i] = @as(f32, @floatFromInt(i)) * 0.1 - 2.0;
        up[i] = 1.5;
        out[i] = 0.0;
    }

    var pipe = try pipeline.ComputePipeline.init(&ctx, &shaders.FUSED_SWIGLU_SPIRV, 3, 4);
    defer pipe.deinit();

    const bufs = [_]*const buffer.GpuBuffer{ &buf_gate, &buf_up, &buf_out };
    try pipe.bindBuffers(&bufs);

    const pc = [_]u32{@intCast(dim)};
    try pipe.dispatch(std.mem.sliceAsBytes(&pc), 2, 1, 1);

    for (0..dim) |i| {
        const g = gate[i];
        const sig = 1.0 / (1.0 + @exp(-g));
        const silu = g * sig;
        const expected = silu * up[i];
        try std.testing.expectApproxEqAbs(expected, out[i], 1e-4);
    }
}

test "benchmark 12B layer GEMV on GPU" {
    var ctx = context.GpuContext.init(std.testing.allocator) catch |err| {
        if (err == error.VulkanLibraryNotFound or err == error.NoVulkanDevices) return;
        return err;
    };
    defer ctx.deinit();

    const m: usize = 15360;
    const k: usize = 3840;
    var buf_w = try buffer.GpuBuffer.init(&ctx, m * (k / 2) * @sizeOf(u32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_w.deinit();
    var buf_x = try buffer.GpuBuffer.init(&ctx, k * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_x.deinit();
    var buf_y = try buffer.GpuBuffer.init(&ctx, m * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_y.deinit();

    var pipe = try pipeline.ComputePipeline.init(&ctx, &shaders.GEMV_BF16_SPIRV, 3, 8);
    defer pipe.deinit();

    const bufs = [_]*const buffer.GpuBuffer{ &buf_w, &buf_x, &buf_y };
    try pipe.bindBuffers(&bufs);

    const pc = [_]u32{ @intCast(m), @intCast(k) };
    const workgroups: u32 = @intCast((m + 63) / 64);

    var timer = try std.time.Timer.start();
    try pipe.dispatch(std.mem.sliceAsBytes(&pc), workgroups, 1, 1);
    const elapsed_ns = timer.read();
    std.debug.print("\n>>> 15360x3840 GEMV GPU execution: {d:.2} ms <<<\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0});
}
