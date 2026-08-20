const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const kernels = @import("kernels.zig");
pub const model_types = @import("../model/types.zig");
pub const model_gpu = @import("model_gpu.zig");

pub fn recordForwardGraph(
    gpu: *model_gpu.GpuModelContext,
    config: *const model_types.ModelConfig,
    layers: []const model_types.LayerWeights,
    cmd_buf: types.VkCommandBuffer,
    include_logits: bool,
) void {
    const H = config.hidden_size;
    const inter = config.intermediate_size;
    const eps = config.rms_norm_eps;
    const V = config.vocab_size;

    const begin_info = types_dispatch.VkCommandBufferBeginInfo{};
    _ = gpu.ctx.api.vkBeginCommandBuffer(cmd_buf, &begin_info);
    gpu.engine.recordBarrier(cmd_buf);

    for (gpu.layers, 0..) |l_gpu, i| {
        const l_cpu = layers[i];
        const q_dim = l_cpu.q_dim;
        const kv_dim = l_cpu.kv_dim;
        const head_dim: u32 = @intCast(l_cpu.head_dim);
        const rot_dim: u32 = @intCast(l_cpu.rotary_dim);
        const theta: f32 = if (l_cpu.layer_type == .full_attention) config.rope_theta_full else config.rope_theta;
        const gqa_ratio: u32 = @intCast(config.num_attention_heads / l_cpu.num_kv_heads);
        const inv_sqrt_dim: f32 = 1.0;

        // 1. Input RMSNorm: buf_x -> buf_normed_x
        gpu.engine.recordRmsNorm(cmd_buf, l_gpu.desc.input_norm, H, eps);
        gpu.engine.recordBarrier(cmd_buf);

        // 2. Q, K, V Projections
        gpu.engine.recordGemv(cmd_buf, l_gpu.desc.q_proj, q_dim, H);
        gpu.engine.recordGemv(cmd_buf, l_gpu.desc.k_proj, kv_dim, H);
        gpu.engine.recordGemv(cmd_buf, l_gpu.desc.v_proj, kv_dim, H);
        gpu.engine.recordBarrier(cmd_buf);

        // 3. QKV RoPE + Cache write
        gpu.engine.recordQkvRope(cmd_buf, l_gpu.desc.qkv_rope, config.num_attention_heads, l_cpu.num_kv_heads, head_dim, rot_dim, l_cpu.k_eq_v, theta, eps);
        gpu.engine.recordBarrier(cmd_buf);

        // 4. Decode Attention
        gpu.engine.recordDecodeAttn(cmd_buf, l_gpu.desc.attn, head_dim, kv_dim, gqa_ratio, inv_sqrt_dim, config.num_attention_heads);
        gpu.engine.recordBarrier(cmd_buf);

        // 5. O Projection
        gpu.engine.recordGemv(cmd_buf, l_gpu.desc.o_proj, H, q_dim);
        if (l_gpu.has_post_attn_norm) {
            gpu.engine.recordBarrier(cmd_buf);
            gpu.engine.recordRmsNorm(cmd_buf, l_gpu.desc.post_attn_norm, H, eps);
        }
        gpu.engine.recordBarrier(cmd_buf);

        // 6. Attention residual + pre-FFN RMSNorm: buf_x += buf_mlp_out -> buf_normed_x
        gpu.engine.recordAddRmsNorm(cmd_buf, l_gpu.desc.pre_ffn_norm, H, eps, 1.0);
        gpu.engine.recordBarrier(cmd_buf);

        // 7. Fused Gate + Up + GeGLU -> buf_act
        gpu.engine.recordGateUpSwiGlu(cmd_buf, l_gpu.desc.gate_up_swiglu, inter, H);
        gpu.engine.recordBarrier(cmd_buf);

        // 8. Down projection: buf_act -> buf_mlp_out
        gpu.engine.recordGemv(cmd_buf, l_gpu.desc.down_proj, H, inter);
        if (l_gpu.has_post_ffn_norm) {
            gpu.engine.recordBarrier(cmd_buf);
            gpu.engine.recordRmsNorm(cmd_buf, l_gpu.desc.post_ffn_norm, H, eps);
        }
        gpu.engine.recordBarrier(cmd_buf);

        // 9. FFN residual: buf_x = (buf_x + buf_mlp_out) * layer_scalar
        gpu.engine.recordAddRmsNorm(cmd_buf, l_gpu.desc.post_ffn_add, H, eps, l_gpu.layer_scalar);
        gpu.engine.recordBarrier(cmd_buf);
    }

    if (include_logits) {
        // 10. Final RMSNorm: buf_x -> buf_normed_x
        gpu.engine.recordRmsNorm(cmd_buf, gpu.desc_final_norm, H, eps);
        gpu.engine.recordBarrier(cmd_buf);

        // 11. Logits projection
        gpu.engine.recordGemvLogits(cmd_buf, gpu.desc_logits, V, H);
        gpu.engine.recordBarrier(cmd_buf);

        // 12. GPU Argmax
        gpu.engine.recordArgmax(cmd_buf, gpu.desc_argmax, V);
    }

    _ = gpu.ctx.api.vkEndCommandBuffer(cmd_buf);
}

pub fn gpuDispatchForwardToken(
    gpu: *model_gpu.GpuModelContext,
    config: *const model_types.ModelConfig,
    layers: []const model_types.LayerWeights,
    x: []const f32,
    logits: []f32,
    clock: usize,
    slot_idx: usize,
    active_slots: []const usize,
) u32 {
    _ = layers;
    const H = config.hidden_size;
    const V = config.vocab_size;

    @memcpy(gpu.buf_x.asSlice(f32)[0..H], x[0..H]);
    for (gpu.buf_active_slots.asSlice(u32)[0..active_slots.len], active_slots) |*dst, s| {
        dst.* = @intCast(s);
    }
    const params = gpu.buf_step_params.asSlice(u32);
    params[0] = @intCast(clock);
    params[1] = @intCast(slot_idx);
    params[2] = @intCast(active_slots.len);
    params[3] = 0;

    const cmd_buf = if (logits.len >= V) gpu.cmd_buf_decode else gpu.cmd_buf_prefill;
    gpu.engine.submitPreRecorded(cmd_buf) catch return 0;

    if (logits.len >= V) {
        return gpu.buf_sampled_token.asSlice(u32)[0];
    }
    return 0;
}
