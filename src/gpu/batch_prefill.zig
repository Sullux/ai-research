const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const pipeline = @import("pipeline.zig");
pub const descriptors = @import("descriptors.zig");
pub const model_types = @import("../model/types.zig");
pub const model_gpu = @import("model_gpu.zig");
pub const tensor = @import("../tensor.zig");
pub const quant = @import("../quant.zig");

pub const shaders_rmsnorm = @import("shaders_batch_rmsnorm.zig");
pub const shaders_gemm_q4 = @import("shaders_batch_gemm_q4.zig");
pub const shaders_gemm_q8 = @import("shaders_batch_gemm_q8.zig");
pub const shaders_add_norm = @import("shaders_batch_add_rmsnorm.zig");
pub const shaders_fused_mlp = @import("shaders_batch_fused_mlp_q4.zig");
pub const shaders_qkv_rope = @import("shaders_batch_qkv_rope.zig");
pub const shaders_causal_attn = @import("shaders_batch_causal_attn.zig");

pub const shaders_fused_mlp_q8 = @import("shaders_batch_fused_mlp_q8.zig");

pub const BatchPrefillContext = struct {
    ctx: *const context.GpuContext,
    pipe_rmsnorm: pipeline.ComputePipeline,
    pipe_gemm_q4: pipeline.ComputePipeline,
    pipe_gemm_q8: pipeline.ComputePipeline,
    pipe_add_norm: pipeline.ComputePipeline,
    pipe_fused_mlp_q4: pipeline.ComputePipeline,
    pipe_fused_mlp_q8: pipeline.ComputePipeline,
    pipe_qkv_rope: pipeline.ComputePipeline,
    pipe_causal_attn: pipeline.ComputePipeline,
    desc_mgr: descriptors.DescriptorManager,

    buf_x: buffer.GpuBuffer,
    buf_normed_x: buffer.GpuBuffer,
    buf_q: buffer.GpuBuffer,
    buf_k: buffer.GpuBuffer,
    buf_v: buffer.GpuBuffer,
    buf_attn_out: buffer.GpuBuffer,
    buf_act: buffer.GpuBuffer,
    buf_mlp_out: buffer.GpuBuffer,
    buf_slots: buffer.GpuBuffer,

    cmd_pool: types.VkCommandPool,
    cmd_buf: types.VkCommandBuffer,
    fence: types.VkFence,
    max_tokens: usize,

    pub fn init(ctx: *const context.GpuContext, config: *const model_types.ModelConfig, max_tokens: usize) !BatchPrefillContext {
        const N = max_tokens;
        const H = config.hidden_size;
        const max_head = @max(config.head_dim, config.global_head_dim);
        const max_q = config.num_attention_heads * max_head;
        const max_kv = @max(config.num_key_value_heads, config.num_global_key_value_heads) * max_head;
        const I = config.intermediate_size;
        const sb = types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

        var p_rms = try pipeline.ComputePipeline.init(ctx, &shaders_rmsnorm.BATCH_RMSNORM_SPIRV, 3, 16); errdefer p_rms.deinit();
        var p_g4 = try pipeline.ComputePipeline.init(ctx, &shaders_gemm_q4.BATCH_GEMM_Q4_SPIRV, 3, 16); errdefer p_g4.deinit();
        var p_g8 = try pipeline.ComputePipeline.init(ctx, &shaders_gemm_q8.BATCH_GEMM_Q8_SPIRV, 3, 16); errdefer p_g8.deinit();
        var p_add = try pipeline.ComputePipeline.init(ctx, &shaders_add_norm.BATCH_ADD_RMSNORM_SPIRV, 4, 16); errdefer p_add.deinit();
        var p_mlp4 = try pipeline.ComputePipeline.init(ctx, &shaders_fused_mlp.BATCH_FUSED_MLP_Q4_SPIRV, 4, 16); errdefer p_mlp4.deinit();
        var p_mlp8 = try pipeline.ComputePipeline.init(ctx, &shaders_fused_mlp_q8.BATCH_FUSED_MLP_Q8_SPIRV, 4, 16); errdefer p_mlp8.deinit();
        var p_rope = try pipeline.ComputePipeline.init(ctx, &shaders_qkv_rope.BATCH_QKV_ROPE_SPIRV, 9, 32); errdefer p_rope.deinit();
        var p_attn = try pipeline.ComputePipeline.init(ctx, &shaders_causal_attn.BATCH_CAUSAL_ATTN_SPIRV, 5, 24); errdefer p_attn.deinit();

        var desc_mgr = try descriptors.DescriptorManager.init(ctx, 4096);
        errdefer desc_mgr.deinit();

        var bx = try buffer.GpuBuffer.init(ctx, N * H * 4, sb); errdefer bx.deinit();
        var bnx = try buffer.GpuBuffer.init(ctx, N * H * 4, sb); errdefer bnx.deinit();
        var bq = try buffer.GpuBuffer.init(ctx, N * max_q * 4, sb); errdefer bq.deinit();
        var bk = try buffer.GpuBuffer.init(ctx, N * max_kv * 4, sb); errdefer bk.deinit();
        var bv = try buffer.GpuBuffer.init(ctx, N * max_kv * 4, sb); errdefer bv.deinit();
        var ba = try buffer.GpuBuffer.init(ctx, N * max_q * 4, sb); errdefer ba.deinit();
        var bact = try buffer.GpuBuffer.init(ctx, N * I * 4, sb); errdefer bact.deinit();
        var bmo = try buffer.GpuBuffer.init(ctx, N * H * 4, sb); errdefer bmo.deinit();
        var bs = try buffer.GpuBuffer.init(ctx, N * 4, sb); errdefer bs.deinit();

        const cp_info = types_dispatch.VkCommandPoolCreateInfo{ .flags = types.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = ctx.queue_family_index };
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
            .ctx = ctx, .pipe_rmsnorm = p_rms, .pipe_gemm_q4 = p_g4, .pipe_gemm_q8 = p_g8,
            .pipe_add_norm = p_add, .pipe_fused_mlp_q4 = p_mlp4, .pipe_fused_mlp_q8 = p_mlp8,
            .pipe_qkv_rope = p_rope, .pipe_causal_attn = p_attn, .desc_mgr = desc_mgr,
            .buf_x = bx, .buf_normed_x = bnx, .buf_q = bq, .buf_k = bk, .buf_v = bv,
            .buf_attn_out = ba, .buf_act = bact, .buf_mlp_out = bmo, .buf_slots = bs,
            .cmd_pool = pool, .cmd_buf = cmd, .fence = fence, .max_tokens = max_tokens,
        };
    }

    pub fn deinit(self: *BatchPrefillContext) void {
        _ = self.ctx.api.vkQueueWaitIdle(self.ctx.queue);
        self.ctx.api.vkDestroyFence(self.ctx.device, self.fence, null);
        self.ctx.api.vkDestroyCommandPool(self.ctx.device, self.cmd_pool, null);
        self.buf_slots.deinit(); self.buf_mlp_out.deinit(); self.buf_act.deinit();
        self.buf_attn_out.deinit(); self.buf_v.deinit(); self.buf_k.deinit();
        self.buf_q.deinit(); self.buf_normed_x.deinit(); self.buf_x.deinit();
        self.desc_mgr.deinit();
        self.pipe_causal_attn.deinit(); self.pipe_qkv_rope.deinit();
        self.pipe_fused_mlp_q8.deinit(); self.pipe_fused_mlp_q4.deinit();
        self.pipe_add_norm.deinit(); self.pipe_gemm_q8.deinit(); self.pipe_gemm_q4.deinit();
        self.pipe_rmsnorm.deinit();
    }
};
