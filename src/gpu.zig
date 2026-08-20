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

    const q4_size = quant.getQuantizedSizeBytes(m, k, .q4);
    var buf_q4 = try buffer.GpuBuffer.init(&ctx, q4_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_q4.deinit();
    quant.quantizeMatrix(buf_q4.asSlice(u32), raw, m, k, .q4);
    var p_q4 = try pipeline.ComputePipeline.init(&ctx, &shaders.GEMV_Q4_SPIRV, 3, 8);
    defer p_q4.deinit();
    try p_q4.bindBuffers(&[_]*const buffer.GpuBuffer{ &buf_q4, &buf_x, &buf_y });
    const pc = [_]u32{ @intCast(m), @intCast(k) };
    try p_q4.dispatch(std.mem.sliceAsBytes(&pc), @intCast(m), 1, 1);
    for (buf_y.asSlice(f32)) |y| try std.testing.expect(y > 0);
}

test "vulkan decode attention execution on AMD GPU" {
    const maybe_ctx = try testCtx();
    if (maybe_ctx == null) return;
    var ctx = maybe_ctx.?;
    defer ctx.deinit();

    const num_q_heads: usize = 16;
    const num_kv_heads: usize = 8;
    const head_dim: usize = 256;
    const n_active: usize = 4;
    const max_slots: usize = 8;

    const q_size = num_q_heads * head_dim * @sizeOf(f32);
    const kv_size = max_slots * num_kv_heads * head_dim * @sizeOf(f32);

    var buf_q = try buffer.GpuBuffer.init(&ctx, q_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_q.deinit();
    var buf_k = try buffer.GpuBuffer.init(&ctx, kv_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_k.deinit();
    var buf_v = try buffer.GpuBuffer.init(&ctx, kv_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_v.deinit();
    var buf_slots = try buffer.GpuBuffer.init(&ctx, n_active * @sizeOf(u32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_slots.deinit();
    var buf_out = try buffer.GpuBuffer.init(&ctx, q_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_out.deinit();

    @memset(buf_q.asSlice(f32), 0.1);
    @memset(buf_k.asSlice(f32), 0.1);
    @memset(buf_v.asSlice(f32), 1.0);
    for (buf_slots.asSlice(u32), 0..) |*s, i| s.* = @intCast(i);

    var pipe = try pipeline.ComputePipeline.init(&ctx, &shaders.DECODE_ATTENTION_SPIRV, 5, 8);
    defer pipe.deinit();
    const bufs = [_]*const buffer.GpuBuffer{ &buf_q, &buf_k, &buf_v, &buf_slots, &buf_out };
    try pipe.bindBuffers(&bufs);
    const pc = extern struct { n_active: u32 = @intCast(n_active), kv_stride: u32 = @intCast(num_kv_heads * head_dim) }{};
    try pipe.dispatch(std.mem.asBytes(&pc), @intCast(num_q_heads), 1, 1);

    for (buf_out.asSlice(f32)) |v| try std.testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-4);
}
