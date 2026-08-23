const std = @import("std");
const context = @import("gpu/context.zig");
const buffer = @import("gpu/buffer.zig");
const pipeline = @import("gpu/pipeline.zig");
const descriptors = @import("gpu/descriptors.zig");
const types = @import("gpu/types.zig");
const quant = @import("quant.zig");
const shaders_batch = @import("gpu/shaders_batch_gemm_q4_tiled.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var gpu_ctx = try context.GpuContext.init(allocator);
    defer gpu_ctx.deinit();

    const N: u32 = 392;
    const M: u32 = 3840;
    const K: u32 = 3840;

    // Create pipeline for batch_gemm_q4_tiled
    var pipe = try pipeline.ComputePipeline.init(&gpu_ctx, &shaders_batch.BATCH_GEMM_Q4_TILED_SPIRV, 3, @sizeOf(u32) * 4);
    defer pipe.deinit();

    var desc_mgr = try descriptors.DescriptorManager.init(&gpu_ctx, 16);
    defer desc_mgr.deinit();

    const q4_bytes = quant.getQuantizedSizeBytes(M, K, .q4);
    var buf_w = try buffer.GpuBuffer.init(&gpu_ctx, q4_bytes, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_w.deinit();
    @memset(buf_w.asSlice(u8), 0x88); // Zero weights (8 - 8 = 0)

    var buf_x = try buffer.GpuBuffer.init(&gpu_ctx, N * K * 4, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_x.deinit();
    @memset(buf_x.asSlice(f32), 1.0);

    var buf_y = try buffer.GpuBuffer.init(&gpu_ctx, N * M * 4, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_y.deinit();

    const dset = try desc_mgr.allocateSet(pipe.desc_set_layout);
    desc_mgr.bindBuffers(dset, &.{ &buf_w, &buf_x, &buf_y });

    const types_disp = @import("gpu/types_dispatch.zig");
    const cp_info = types_disp.VkCommandPoolCreateInfo{
        .queueFamilyIndex = gpu_ctx.queue_family_index,
    };
    var cmd_pool: types.VkCommandPool = null;
    _ = gpu_ctx.api.vkCreateCommandPool(gpu_ctx.device, &cp_info, null, &cmd_pool);
    defer gpu_ctx.api.vkDestroyCommandPool(gpu_ctx.device, cmd_pool, null);

    const cb_info = types_disp.VkCommandBufferAllocateInfo{
        .commandPool = cmd_pool,
        .commandBufferCount = 1,
    };
    var cmd_buf: types.VkCommandBuffer = null;
    _ = gpu_ctx.api.vkAllocateCommandBuffers(gpu_ctx.device, &cb_info, @ptrCast(&cmd_buf));

    const begin_info = types_disp.VkCommandBufferBeginInfo{};
    _ = gpu_ctx.api.vkBeginCommandBuffer(cmd_buf, &begin_info);

    const pc = [4]u32{ N, M, K, 0 };
    pipe.record(cmd_buf, dset, std.mem.sliceAsBytes(&pc), M, (N + 3) / 4, 1);

    _ = gpu_ctx.api.vkEndCommandBuffer(cmd_buf);

    const fence_info = types_disp.VkFenceCreateInfo{};
    var fence: types.VkFence = null;
    _ = gpu_ctx.api.vkCreateFence(gpu_ctx.device, &fence_info, null, &fence);
    defer gpu_ctx.api.vkDestroyFence(gpu_ctx.device, fence, null);

    const submit_info = types_disp.VkSubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = @ptrCast(&cmd_buf),
    };

    const submits = [1]types_disp.VkSubmitInfo{submit_info};
    // Benchmark 100 runs
    const start = std.time.nanoTimestamp();
    for (0..100) |_| {
        _ = gpu_ctx.api.vkResetFences(gpu_ctx.device, 1, @ptrCast(&fence));
        _ = gpu_ctx.api.vkQueueSubmit(gpu_ctx.queue, 1, &submits, fence);
        _ = gpu_ctx.api.vkWaitForFences(gpu_ctx.device, 1, @ptrCast(&fence), 1, 1_000_000_000);
    }
    const elapsed_ns = std.time.nanoTimestamp() - start;
    const avg_us = @as(f64, @floatFromInt(elapsed_ns)) / 100.0 / 1000.0;

    std.debug.print("Batch GEMM [392, 3840] x [3840, 3840] Q4_0 avg time: {d:.2} us ({d:.3} ms)\n", .{ avg_us, avg_us / 1000.0 });
    std.debug.print("Total prefill layer time (7 GEMMs x 48 layers): {d:.2} ms\n", .{ (avg_us / 1000.0) * 7.0 * 48.0 });
}
