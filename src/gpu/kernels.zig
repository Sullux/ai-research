const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const pipeline = @import("pipeline.zig");
pub const shaders = @import("shaders.zig");
pub const quant = @import("../quant.zig");

pub const GpuEngine = struct {
    ctx: *const context.GpuContext,
    gemv_pipe: pipeline.ComputePipeline,
    swiglu_pipe: pipeline.ComputePipeline,
    cmd_pool: types.VkCommandPool,
    cmd_buf: types.VkCommandBuffer,
    fence: types.VkFence,

    pub fn init(ctx: *const context.GpuContext, mode: quant.QuantMode) !GpuEngine {
        const gemv_spirv = switch (mode) {
            .none => &shaders.GEMV_BF16_SPIRV,
            .q8 => &shaders.GEMV_Q8_SPIRV,
            .q4 => &shaders.GEMV_Q4_SPIRV,
        };
        var gemv = try pipeline.ComputePipeline.init(ctx, gemv_spirv, 3, 8);
        errdefer gemv.deinit();

        var swiglu = try pipeline.ComputePipeline.init(ctx, &shaders.FUSED_SWIGLU_SPIRV, 3, 4);
        errdefer swiglu.deinit();

        const cp_info = types_dispatch.VkCommandPoolCreateInfo{ .queueFamilyIndex = ctx.queue_family_index };
        var pool: types.VkCommandPool = null;
        if (ctx.api.vkCreateCommandPool(ctx.device, &cp_info, null, &pool) != .SUCCESS) return error.VkCmdPoolCreationFailed;
        errdefer ctx.api.vkDestroyCommandPool(ctx.device, pool, null);

        const cb_info = types_dispatch.VkCommandBufferAllocateInfo{ .commandPool = pool, .commandBufferCount = 1 };
        var cmd: types.VkCommandBuffer = null;
        if (ctx.api.vkAllocateCommandBuffers(ctx.device, &cb_info, (&cmd)[0..1].ptr) != .SUCCESS) return error.VkCmdBufferAllocFailed;

        const fence_info = types_dispatch.VkFenceCreateInfo{};
        var fence: types.VkFence = null;
        if (ctx.api.vkCreateFence(ctx.device, &fence_info, null, &fence) != .SUCCESS) return error.VkFenceCreationFailed;

        return .{
            .ctx = ctx,
            .gemv_pipe = gemv,
            .swiglu_pipe = swiglu,
            .cmd_pool = pool,
            .cmd_buf = cmd,
            .fence = fence,
        };
    }

    pub fn deinit(self: *GpuEngine) void {
        self.ctx.api.vkDestroyFence(self.ctx.device, self.fence, null);
        self.ctx.api.vkDestroyCommandPool(self.ctx.device, self.cmd_pool, null);
        self.swiglu_pipe.deinit();
        self.gemv_pipe.deinit();
    }

    pub fn beginBatch(self: *const GpuEngine) void {
        _ = self.ctx.api.vkResetFences(self.ctx.device, 1, (&self.fence)[0..1].ptr);
        const begin_info = types_dispatch.VkCommandBufferBeginInfo{};
        _ = self.ctx.api.vkBeginCommandBuffer(self.cmd_buf, &begin_info);
    }

    pub fn recordGemv(self: *const GpuEngine, set: types.VkDescriptorSet, m: usize, k: usize) void {
        const pc = [_]u32{ @intCast(m), @intCast(k) };
        const workgroups: u32 = @intCast((m + 63) / 64);
        self.gemv_pipe.record(self.cmd_buf, set, std.mem.sliceAsBytes(&pc), workgroups, 1, 1);
    }

    pub fn recordSwiGlu(self: *const GpuEngine, set: types.VkDescriptorSet, dim: usize) void {
        const pc = [_]u32{@intCast(dim)};
        const workgroups: u32 = @intCast((dim + 63) / 64);
        self.swiglu_pipe.record(self.cmd_buf, set, std.mem.sliceAsBytes(&pc), workgroups, 1, 1);
    }

    pub fn recordBarrier(self: *const GpuEngine, buf: *const buffer.GpuBuffer) void {
        const b = types_dispatch.VkBufferMemoryBarrier{
            .srcAccessMask = types.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = types.VK_ACCESS_SHADER_READ_BIT,
            .buffer = buf.buffer,
            .offset = 0,
            .size = buf.size,
        };
        self.ctx.api.vkCmdPipelineBarrier(self.cmd_buf, types.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, types.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, null, 1, (&b)[0..1].ptr, 0, null);
    }

    pub fn submitBatch(self: *const GpuEngine) !void {
        _ = self.ctx.api.vkEndCommandBuffer(self.cmd_buf);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&self.cmd_buf)[0..1].ptr };
        if (self.ctx.api.vkQueueSubmit(self.ctx.queue, 1, (&submit_info)[0..1].ptr, self.fence) != .SUCCESS) return error.VkQueueSubmitFailed;

        var spin: usize = 0;
        while (spin < 500_000) : (spin += 1) {
            if (self.ctx.api.vkGetFenceStatus(self.ctx.device, self.fence) == .SUCCESS) return;
            std.atomic.spinLoopHint();
        }
        if (self.ctx.api.vkWaitForFences(self.ctx.device, 1, (&self.fence)[0..1].ptr, 1, 5_000_000_000) != .SUCCESS) return error.VkFenceTimeout;
    }

    pub fn dispatchGemv(self: *const GpuEngine, w: *const buffer.GpuBuffer, x: *const buffer.GpuBuffer, y: *const buffer.GpuBuffer, m: usize, k: usize) !void {
        const bufs = [_]*const buffer.GpuBuffer{ w, x, y };
        try self.gemv_pipe.bindBuffers(&bufs);
        const pc = [_]u32{ @intCast(m), @intCast(k) };
        const workgroups: u32 = @intCast((m + 63) / 64);
        try self.gemv_pipe.dispatch(std.mem.sliceAsBytes(&pc), workgroups, 1, 1);
    }

    pub fn dispatchSwiGlu(self: *const GpuEngine, gate: *const buffer.GpuBuffer, up: *const buffer.GpuBuffer, out: *const buffer.GpuBuffer, dim: usize) !void {
        const bufs = [_]*const buffer.GpuBuffer{ gate, up, out };
        try self.swiglu_pipe.bindBuffers(&bufs);
        const pc = [_]u32{@intCast(dim)};
        const workgroups: u32 = @intCast((dim + 63) / 64);
        try self.swiglu_pipe.dispatch(std.mem.sliceAsBytes(&pc), workgroups, 1, 1);
    }
};
