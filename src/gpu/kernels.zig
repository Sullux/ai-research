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
    mode: quant.QuantMode,
    gemv_pipe: pipeline.ComputePipeline,
    gemv_logits_pipe: pipeline.ComputePipeline,
    swiglu_pipe: pipeline.ComputePipeline,
    gate_up_pipe: pipeline.ComputePipeline,
    add_rmsnorm_pipe: pipeline.ComputePipeline,
    rmsnorm_pipe: pipeline.ComputePipeline,
    attn_pipe: pipeline.ComputePipeline,
    qkv_rope_pipe: pipeline.ComputePipeline,
    argmax_pipe: pipeline.ComputePipeline,
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

        const logits_spirv = if (mode == .q4) &shaders.GEMV_Q8_SPIRV else gemv_spirv;
        var gemv_logits = try pipeline.ComputePipeline.init(ctx, logits_spirv, 3, 8);
        errdefer gemv_logits.deinit();

        var swiglu = try pipeline.ComputePipeline.init(ctx, &shaders.FUSED_SWIGLU_SPIRV, 3, 4);
        errdefer swiglu.deinit();

        const gate_up_spirv = switch (mode) {
            .q4 => &shaders.FUSED_GATE_UP_SWIGLU_Q4_SPIRV,
            .q8 => &shaders.FUSED_GATE_UP_SWIGLU_Q8_SPIRV,
            .none => &shaders.FUSED_GATE_UP_SWIGLU_Q4_SPIRV,
        };
        var gate_up = try pipeline.ComputePipeline.init(ctx, gate_up_spirv, 4, 8);
        errdefer gate_up.deinit();

        var add_rms = try pipeline.ComputePipeline.init(ctx, &shaders.FUSED_ADD_RMSNORM_SPIRV, 4, 12);
        errdefer add_rms.deinit();

        var rms = try pipeline.ComputePipeline.init(ctx, &shaders.RMSNORM_SPIRV, 3, 8);
        errdefer rms.deinit();

        var attn = try pipeline.ComputePipeline.init(ctx, &shaders.DECODE_ATTENTION_SPIRV, 5, 20);
        errdefer attn.deinit();

        var qkv_rope = try pipeline.ComputePipeline.init(ctx, &shaders.QKV_ROPE_SPIRV, 8, 32);
        errdefer qkv_rope.deinit();

        var argmax = try pipeline.ComputePipeline.init(ctx, &shaders.ARGMAX_SPIRV, 2, 4);
        errdefer argmax.deinit();

        const cp_info = types_dispatch.VkCommandPoolCreateInfo{
            .flags = types.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = ctx.queue_family_index,
        };
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
            .ctx = ctx, .mode = mode, .gemv_pipe = gemv, .gemv_logits_pipe = gemv_logits,
            .swiglu_pipe = swiglu, .gate_up_pipe = gate_up, .add_rmsnorm_pipe = add_rms,
            .rmsnorm_pipe = rms, .attn_pipe = attn, .qkv_rope_pipe = qkv_rope,
            .argmax_pipe = argmax,
            .cmd_pool = pool, .cmd_buf = cmd, .fence = fence,
        };
    }

    pub fn deinit(self: *GpuEngine) void {
        _ = self.ctx.api.vkQueueWaitIdle(self.ctx.queue);
        self.ctx.api.vkDestroyFence(self.ctx.device, self.fence, null);
        self.ctx.api.vkDestroyCommandPool(self.ctx.device, self.cmd_pool, null);
        self.qkv_rope_pipe.deinit(); self.attn_pipe.deinit(); self.rmsnorm_pipe.deinit();
        self.add_rmsnorm_pipe.deinit(); self.gate_up_pipe.deinit(); self.swiglu_pipe.deinit();
        self.gemv_logits_pipe.deinit(); self.gemv_pipe.deinit();
        self.argmax_pipe.deinit();
    }

    pub fn beginBatch(self: *const GpuEngine) void {
        _ = self.ctx.api.vkResetFences(self.ctx.device, 1, (&self.fence)[0..1].ptr);
        const begin_info = types_dispatch.VkCommandBufferBeginInfo{};
        _ = self.ctx.api.vkBeginCommandBuffer(self.cmd_buf, &begin_info);
    }

    pub fn recordGemv(self: *const GpuEngine, set: types.VkDescriptorSet, m: usize, k: usize) void {
        const pc = [_]u32{ @intCast(m), @intCast(k) };
        const workgroups: u32 = @intCast((m + 3) / 4);
        self.gemv_pipe.record(self.cmd_buf, set, std.mem.sliceAsBytes(&pc), workgroups, 1, 1);
    }

    pub fn recordGemvLogits(self: *const GpuEngine, set: types.VkDescriptorSet, m: usize, k: usize) void {
        const pc = [_]u32{ @intCast(m), @intCast(k) };
        const workgroups: u32 = @intCast((m + 3) / 4);
        self.gemv_logits_pipe.record(self.cmd_buf, set, std.mem.sliceAsBytes(&pc), workgroups, 1, 1);
    }

    pub fn recordGateUpSwiGlu(self: *const GpuEngine, set: types.VkDescriptorSet, m: usize, k: usize) void {
        const pc = [_]u32{ @intCast(m), @intCast(k) };
        const workgroups: u32 = @intCast((m + 3) / 4);
        self.gate_up_pipe.record(self.cmd_buf, set, std.mem.sliceAsBytes(&pc), workgroups, 1, 1);
    }

    pub fn recordSwiGlu(self: *const GpuEngine, set: types.VkDescriptorSet, dim: usize) void {
        const pc = [_]u32{@intCast(dim)};
        const workgroups: u32 = @intCast((dim + 63) / 64);
        self.swiglu_pipe.record(self.cmd_buf, set, std.mem.sliceAsBytes(&pc), workgroups, 1, 1);
    }

    pub fn recordAddRmsNorm(self: *const GpuEngine, set: types.VkDescriptorSet, H: usize, eps: f32, scalar: f32) void {
        const pc = extern struct { h: u32, eps: f32, scalar: f32 }{ .h = @intCast(H), .eps = eps, .scalar = scalar };
        self.add_rmsnorm_pipe.record(self.cmd_buf, set, std.mem.asBytes(&pc), 1, 1, 1);
    }

    pub fn recordRmsNorm(self: *const GpuEngine, set: types.VkDescriptorSet, H: usize, eps: f32) void {
        const pc = extern struct { h: u32, eps: f32 }{ .h = @intCast(H), .eps = eps };
        self.rmsnorm_pipe.record(self.cmd_buf, set, std.mem.asBytes(&pc), 1, 1, 1);
    }

    pub fn recordDecodeAttn(self: *const GpuEngine, set: types.VkDescriptorSet, n_active: usize, head_dim: usize, kv_dim: usize, gqa_ratio: usize, inv_sqrt_dim: f32, num_q_heads: usize) void {
        const pc = extern struct { n_active: u32, head_dim: u32, kv_dim: u32, gqa_ratio: u32, inv_sqrt_dim: f32 }{
            .n_active = @intCast(n_active),
            .head_dim = @intCast(head_dim),
            .kv_dim = @intCast(kv_dim),
            .gqa_ratio = @intCast(gqa_ratio),
            .inv_sqrt_dim = inv_sqrt_dim,
        };
        self.attn_pipe.record(self.cmd_buf, set, std.mem.asBytes(&pc), @intCast(num_q_heads), 1, 1);
    }

    pub fn recordQkvRope(self: *const GpuEngine, set: types.VkDescriptorSet, clock: usize, num_q: usize, num_kv: usize, head_dim: usize, rotary_dim: usize, slot_idx: usize, k_eq_v: bool, theta: f32, eps: f32) void {
        const pc = extern struct {
            clock: u32,
            num_q: u32,
            num_kv: u32,
            head_dim: u32,
            rotary_dim: u32,
            slot_idx: u32,
            k_eq_v: u32,
            theta: f32,
            eps: f32,
        }{
            .clock = @intCast(clock),
            .num_q = @intCast(num_q),
            .num_kv = @intCast(num_kv),
            .head_dim = @intCast(head_dim),
            .rotary_dim = @intCast(rotary_dim),
            .slot_idx = @intCast(slot_idx),
            .k_eq_v = if (k_eq_v) 1 else 0,
            .theta = theta,
            .eps = eps,
        };
        const workgroups: u32 = @intCast(num_q + num_kv);
        self.qkv_rope_pipe.record(self.cmd_buf, set, std.mem.asBytes(&pc), workgroups, 1, 1);
    }

    pub fn recordArgmax(self: *const GpuEngine, set: types.VkDescriptorSet, vocab_size: usize) void {
        const pc = extern struct { v: u32 }{ .v = @intCast(vocab_size) };
        self.argmax_pipe.record(self.cmd_buf, set, std.mem.asBytes(&pc), 1, 1, 1);
    }

    pub fn recordBarrier(self: *const GpuEngine, _: ?*const buffer.GpuBuffer) void {
        const b = types_dispatch.VkMemoryBarrier{};
        self.ctx.api.vkCmdPipelineBarrier(self.cmd_buf, types.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, types.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, (&b)[0..1].ptr, 0, null, 0, null);
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
};
