const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const model_types = @import("../model/types.zig");
pub const model_gpu = @import("model_gpu.zig");
pub const model_dispatch = @import("model_dispatch.zig");
pub const batch_prefill = @import("batch_prefill.zig");
pub const tensor = @import("../tensor.zig");

fn recordBarrier(gpu: *model_gpu.GpuModelContext, cmd: types.VkCommandBuffer) void {
    gpu.engine.recordBarrier(cmd);
}

pub fn gpuDispatchPrefillBatch(
    prefill: *batch_prefill.BatchPrefillContext,
    gpu: *model_gpu.GpuModelContext,
    config: *const model_types.ModelConfig,
    layers: []const model_types.LayerWeights,
    tokens: []const u32,
    embed_tokens: []const tensor.bf16,
    slots: []const u32,
    logits_out: []f32,
) !void {
    const N: u32 = @intCast(tokens.len);
    if (N == 0 or N > prefill.max_tokens) return error.BatchTooLarge;
    const H = config.hidden_size;
    const inter = config.intermediate_size;
    const eps = config.rms_norm_eps;
    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));

    prefill.desc_mgr.reset();

    // 1. Fill input buffer with token embeddings
    const x_slice = prefill.buf_x.asSlice(f32)[0 .. N * H];
    for (tokens, 0..) |tok, i| {
        const offset = @as(usize, tok) * H;
        const out_off = i * H;
        for (x_slice[out_off .. out_off + H], embed_tokens[offset .. offset + H]) |*dst, src| {
            dst.* = src.toF32() * embed_scale;
        }
    }
    @memcpy(prefill.buf_slots.asSlice(u32)[0..N], slots[0..N]);

    // 2. Record command buffer
    const begin_info = types_dispatch.VkCommandBufferBeginInfo{};
    _ = prefill.ctx.api.vkBeginCommandBuffer(prefill.cmd_buf, &begin_info);
    recordBarrier(gpu, prefill.cmd_buf);

    for (gpu.layers, 0..) |l_gpu, i| {
        const l_cpu = layers[i];
        const q_dim: u32 = @intCast(l_cpu.q_dim);
        const kv_dim: u32 = @intCast(l_cpu.kv_dim);
        const head_dim: u32 = @intCast(l_cpu.head_dim);
        const rot_dim: u32 = @intCast(l_cpu.rotary_dim);
        const theta: f32 = if (l_cpu.layer_type == .full_attention) config.rope_theta_full else config.rope_theta;
        const gqa_ratio: u32 = @intCast(config.num_attention_heads / l_cpu.num_kv_heads);

        // Input RMSNorm (only layer 0 needs it explicitly; subsequent layers get it from post_ffn_add)
        if (i == 0) {
            const pc_rms = extern struct { H: u32, eps: f32, N: u32, pad: u32 }{ .H = @intCast(H), .eps = eps, .N = N, .pad = 0 };
            const d_rms = try prefill.desc_mgr.allocateSet(prefill.pipe_rmsnorm.desc_set_layout);
            prefill.desc_mgr.bindBuffers(d_rms, &.{ &prefill.buf_x, &l_gpu.input_norm, &prefill.buf_normed_x });
            prefill.pipe_rmsnorm.record(prefill.cmd_buf, d_rms, std.mem.asBytes(&pc_rms), N, 1, 1);
            recordBarrier(gpu, prefill.cmd_buf);
        }

        // Q, K, V Projections
        const pc_q = [4]u32{ N, q_dim, @intCast(H), 0 };
        const pc_kv = [4]u32{ N, kv_dim, @intCast(H), 0 };
        const d_q = try prefill.desc_mgr.allocateSet(prefill.pipe_gemm_q8.desc_set_layout);
        const d_k = try prefill.desc_mgr.allocateSet(prefill.pipe_gemm_q8.desc_set_layout);
        const d_v = try prefill.desc_mgr.allocateSet(prefill.pipe_gemm_q8.desc_set_layout);
        prefill.desc_mgr.bindBuffers(d_q, &.{ &l_gpu.q_proj, &prefill.buf_normed_x, &prefill.buf_q });
        prefill.desc_mgr.bindBuffers(d_k, &.{ &l_gpu.k_proj, &prefill.buf_normed_x, &prefill.buf_k });
        prefill.desc_mgr.bindBuffers(d_v, &.{ &l_gpu.v_proj, &prefill.buf_normed_x, &prefill.buf_v });
        prefill.pipe_gemm_q8.record(prefill.cmd_buf, d_q, std.mem.sliceAsBytes(&pc_q), q_dim, N, 1);
        prefill.pipe_gemm_q8.record(prefill.cmd_buf, d_k, std.mem.sliceAsBytes(&pc_kv), kv_dim, N, 1);
        prefill.pipe_gemm_q8.record(prefill.cmd_buf, d_v, std.mem.sliceAsBytes(&pc_kv), kv_dim, N, 1);
        recordBarrier(gpu, prefill.cmd_buf);

        // QKV RoPE + Write KV Cache
        const pc_rope = extern struct {
            num_q_heads: u32, num_kv_heads: u32, head_dim: u32, rotary_dim: u32,
            k_eq_v: u32, rope_theta: f32, eps: f32, N: u32,
        }{
            .num_q_heads = @intCast(config.num_attention_heads), .num_kv_heads = @intCast(l_cpu.num_kv_heads),
            .head_dim = head_dim, .rotary_dim = rot_dim, .k_eq_v = if (l_cpu.k_eq_v) 1 else 0,
            .rope_theta = theta, .eps = eps, .N = N,
        };
        const d_rope = try prefill.desc_mgr.allocateSet(prefill.pipe_qkv_rope.desc_set_layout);
        prefill.desc_mgr.bindBuffers(d_rope, &.{ &prefill.buf_q, &prefill.buf_k, &prefill.buf_v, &l_gpu.q_norm, &l_gpu.k_norm, &prefill.buf_q, &l_gpu.buf_k_cache, &l_gpu.buf_v_cache, &prefill.buf_slots });
        prefill.pipe_qkv_rope.record(prefill.cmd_buf, d_rope, std.mem.asBytes(&pc_rope), @intCast(config.num_attention_heads + l_cpu.num_kv_heads), N, 1);
        recordBarrier(gpu, prefill.cmd_buf);

        // Causal Attention
        const pc_attn = extern struct {
            head_dim: u32, kv_dim: u32, gqa_ratio: u32, inv_sqrt_dim: f32, num_q_heads: u32, N: u32,
        }{
            .head_dim = head_dim, .kv_dim = kv_dim, .gqa_ratio = gqa_ratio, .inv_sqrt_dim = 1.0,
            .num_q_heads = @intCast(config.num_attention_heads), .N = N,
        };
        const d_attn = try prefill.desc_mgr.allocateSet(prefill.pipe_causal_attn.desc_set_layout);
        prefill.desc_mgr.bindBuffers(d_attn, &.{ &prefill.buf_q, &l_gpu.buf_k_cache, &l_gpu.buf_v_cache, &prefill.buf_slots, &prefill.buf_attn_out });
        prefill.pipe_causal_attn.record(prefill.cmd_buf, d_attn, std.mem.asBytes(&pc_attn), @intCast(config.num_attention_heads), N, 1);
        recordBarrier(gpu, prefill.cmd_buf);

        // Output Projection
        const pc_o = [4]u32{ N, @intCast(H), q_dim, 0 };
        const d_o = try prefill.desc_mgr.allocateSet(prefill.pipe_gemm_q8.desc_set_layout);
        prefill.desc_mgr.bindBuffers(d_o, &.{ &l_gpu.o_proj, &prefill.buf_attn_out, &prefill.buf_mlp_out });
        prefill.pipe_gemm_q8.record(prefill.cmd_buf, d_o, std.mem.sliceAsBytes(&pc_o), @intCast(H), N, 1);
        if (l_gpu.has_post_attn_norm) {
            recordBarrier(gpu, prefill.cmd_buf);
            const pc_pan = extern struct { H: u32, eps: f32, N: u32, pad: u32 }{ .H = @intCast(H), .eps = eps, .N = N, .pad = 0 };
            const d_pan = try prefill.desc_mgr.allocateSet(prefill.pipe_rmsnorm.desc_set_layout);
            prefill.desc_mgr.bindBuffers(d_pan, &.{ &prefill.buf_mlp_out, &l_gpu.post_attn_norm, &prefill.buf_mlp_out });
            prefill.pipe_rmsnorm.record(prefill.cmd_buf, d_pan, std.mem.asBytes(&pc_pan), N, 1, 1);
        }
        recordBarrier(gpu, prefill.cmd_buf);

        // Add RMSNorm (Pre-FFN Norm)
        const pc_pre_ffn = extern struct { H: u32, eps: f32, scalar: f32, N: u32 }{ .H = @intCast(H), .eps = eps, .scalar = 1.0, .N = N };
        const d_pre_ffn = try prefill.desc_mgr.allocateSet(prefill.pipe_add_norm.desc_set_layout);
        prefill.desc_mgr.bindBuffers(d_pre_ffn, &.{ &prefill.buf_x, &prefill.buf_mlp_out, &l_gpu.pre_ffn_norm, &prefill.buf_normed_x });
        prefill.pipe_add_norm.record(prefill.cmd_buf, d_pre_ffn, std.mem.asBytes(&pc_pre_ffn), N, 1, 1);
        recordBarrier(gpu, prefill.cmd_buf);

        // Fused Gate/Up MLP
        const pc_mlp = [4]u32{ N, @intCast(inter), @intCast(H), 0 };
        const pipe_mlp = if (gpu.engine.mode == .q8) prefill.pipe_fused_mlp_q8 else prefill.pipe_fused_mlp_q4;
        const d_mlp = try prefill.desc_mgr.allocateSet(pipe_mlp.desc_set_layout);
        prefill.desc_mgr.bindBuffers(d_mlp, &.{ &l_gpu.gate_proj, &l_gpu.up_proj, &prefill.buf_normed_x, &prefill.buf_act });
        pipe_mlp.record(prefill.cmd_buf, d_mlp, std.mem.sliceAsBytes(&pc_mlp), @intCast(inter), N, 1);
        recordBarrier(gpu, prefill.cmd_buf);

        // Down Projection
        const pc_down = [4]u32{ N, @intCast(H), @intCast(inter), 0 };
        const pipe_down = if (gpu.engine.mode == .q8) prefill.pipe_gemm_q8 else prefill.pipe_gemm_q4;
        const d_down = try prefill.desc_mgr.allocateSet(pipe_down.desc_set_layout);
        prefill.desc_mgr.bindBuffers(d_down, &.{ &l_gpu.down_proj, &prefill.buf_act, &prefill.buf_mlp_out });
        pipe_down.record(prefill.cmd_buf, d_down, std.mem.sliceAsBytes(&pc_down), @intCast(H), N, 1);
        if (l_gpu.has_post_ffn_norm) {
            recordBarrier(gpu, prefill.cmd_buf);
            const pc_pfn = extern struct { H: u32, eps: f32, N: u32, pad: u32 }{ .H = @intCast(H), .eps = eps, .N = N, .pad = 0 };
            const d_pfn = try prefill.desc_mgr.allocateSet(prefill.pipe_rmsnorm.desc_set_layout);
            prefill.desc_mgr.bindBuffers(d_pfn, &.{ &prefill.buf_mlp_out, &l_gpu.post_ffn_norm, &prefill.buf_mlp_out });
            prefill.pipe_rmsnorm.record(prefill.cmd_buf, d_pfn, std.mem.asBytes(&pc_pfn), N, 1, 1);
        }
        recordBarrier(gpu, prefill.cmd_buf);

        // Post-FFN Add + Next Input Norm
        const next_norm = if (i + 1 < gpu.layers.len) &gpu.layers[i + 1].input_norm else &gpu.final_norm;
        const pc_post_ffn = extern struct { H: u32, eps: f32, scalar: f32, N: u32 }{ .H = @intCast(H), .eps = eps, .scalar = l_gpu.layer_scalar, .N = N };
        const d_post_ffn = try prefill.desc_mgr.allocateSet(prefill.pipe_add_norm.desc_set_layout);
        prefill.desc_mgr.bindBuffers(d_post_ffn, &.{ &prefill.buf_x, &prefill.buf_mlp_out, next_norm, &prefill.buf_normed_x });
        prefill.pipe_add_norm.record(prefill.cmd_buf, d_post_ffn, std.mem.asBytes(&pc_post_ffn), N, 1, 1);
        recordBarrier(gpu, prefill.cmd_buf);

        if ((i + 1) % 4 == 0 or i == gpu.layers.len - 1) {
            if (i == gpu.layers.len - 1 and logits_out.len > 0) {
                const copy_region = [1]types_dispatch.VkBufferCopy{.{
                    .srcOffset = @as(u64, N - 1) * H * 4,
                    .dstOffset = 0,
                    .size = H * 4,
                }};
                prefill.ctx.api.vkCmdCopyBuffer(prefill.cmd_buf, prefill.buf_normed_x.buffer, gpu.buf_normed_x.buffer, 1, &copy_region);
                recordBarrier(gpu, prefill.cmd_buf);
                const d_logits = try prefill.desc_mgr.allocateSet(gpu.engine.gemv_logits_pipe.desc_set_layout);
                prefill.desc_mgr.bindBuffers(d_logits, &.{ &gpu.embed_tokens, &gpu.buf_normed_x, &gpu.buf_logits });
                gpu.engine.recordGemvLogits(prefill.cmd_buf, d_logits, config.vocab_size, H, 0);
            }
            _ = prefill.ctx.api.vkEndCommandBuffer(prefill.cmd_buf);
            const sub = [1]types_dispatch.VkSubmitInfo{.{ .commandBufferCount = 1, .pCommandBuffers = @ptrCast(&prefill.cmd_buf) }};
            _ = prefill.ctx.api.vkResetFences(prefill.ctx.device, 1, @ptrCast(&prefill.fence));
            _ = prefill.ctx.api.vkQueueSubmit(prefill.ctx.queue, 1, &sub, prefill.fence);
            _ = prefill.ctx.api.vkWaitForFences(prefill.ctx.device, 1, @ptrCast(&prefill.fence), 1, 10_000_000_000);
            if (i < gpu.layers.len - 1) {
                _ = prefill.ctx.api.vkBeginCommandBuffer(prefill.cmd_buf, &begin_info);
                recordBarrier(gpu, prefill.cmd_buf);
            }
        }
    }

    if (logits_out.len > 0) {
        @memcpy(logits_out, gpu.buf_logits.asSlice(f32)[0..config.vocab_size]);
    }
}
