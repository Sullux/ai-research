const std = @import("std");
pub const types = @import("types.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const kernels = @import("kernels.zig");
pub const model_types = @import("../model/types.zig");
pub const model_gpu = @import("model_gpu.zig");

pub fn gpuDispatchForwardToken(
    gpu: *model_gpu.GpuModelContext,
    config: *const model_types.ModelConfig,
    layers: []const model_types.LayerWeights,
    x: []const f32,
    logits: []f32,
    clock: usize,
    slot_idx: usize,
    active_slots: []const usize,
) bool {
    const H = config.hidden_size;
    const inter = config.intermediate_size;
    const eps = config.rms_norm_eps;
    const V = config.vocab_size;
    const n_active: u32 = @intCast(active_slots.len);

    @memcpy(gpu.buf_x.asSlice(f32)[0..H], x[0..H]);
    for (gpu.buf_active_slots.asSlice(u32)[0..active_slots.len], active_slots) |*dst, s| {
        dst.* = @intCast(s);
    }

    gpu.engine.beginBatch();

    for (gpu.layers, 0..) |l_gpu, i| {
        const l_cpu = layers[i];
        const q_dim = l_cpu.q_dim;
        const kv_dim = l_cpu.kv_dim;
        const head_dim: u32 = @intCast(l_cpu.head_dim);
        const rot_dim: u32 = @intCast(l_cpu.rotary_dim);
        const theta: f32 = if (l_cpu.layer_type == .full_attention) config.rope_theta_full else config.rope_theta;
        const gqa_ratio: u32 = @intCast(config.num_attention_heads / l_cpu.num_kv_heads);
        const inv_sqrt_dim: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(l_cpu.head_dim)));

        // 1. Input RMSNorm: buf_x -> buf_normed_x
        gpu.engine.recordRmsNorm(l_gpu.desc.input_norm, H, eps);
        gpu.engine.recordBarrier(&gpu.buf_normed_x);

        // 2. Q, K, V Projections
        gpu.engine.recordGemv(l_gpu.desc.q_proj, q_dim, H);
        gpu.engine.recordGemv(l_gpu.desc.k_proj, kv_dim, H);
        gpu.engine.recordGemv(l_gpu.desc.v_proj, kv_dim, H);
        gpu.engine.recordBarrier(&gpu.buf_q);
        gpu.engine.recordBarrier(&gpu.buf_k);
        gpu.engine.recordBarrier(&gpu.buf_v);

        // 3. QKV RoPE + Cache write
        gpu.engine.recordQkvRope(l_gpu.desc.qkv_rope, @intCast(clock), @intCast(config.num_attention_heads), @intCast(l_cpu.num_kv_heads), head_dim, rot_dim, @intCast(slot_idx), l_cpu.k_eq_v, theta, eps);
        gpu.engine.recordBarrier(&gpu.buf_q);
        gpu.engine.recordBarrier(&l_gpu.buf_k_cache);
        gpu.engine.recordBarrier(&l_gpu.buf_v_cache);

        // 4. Decode Attention
        gpu.engine.recordDecodeAttn(l_gpu.desc.attn, n_active, head_dim, kv_dim, gqa_ratio, inv_sqrt_dim, config.num_attention_heads);
        gpu.engine.recordBarrier(&gpu.buf_attn_out);

        // 5. O Projection
        gpu.engine.recordGemv(l_gpu.desc.o_proj, H, q_dim);
        gpu.engine.recordBarrier(&gpu.buf_mlp_out);

        // 5.5 Post-Attention RMSNorm
        if (l_gpu.has_post_attn_norm) {
            gpu.engine.recordRmsNorm(l_gpu.desc.post_attn_norm, H, eps);
            gpu.engine.recordBarrier(&gpu.buf_mlp_out);
        }

        // 6. Attention residual + pre-FFN RMSNorm: buf_x += scalar * buf_mlp_out -> buf_normed_x
        gpu.engine.recordAddRmsNorm(l_gpu.desc.pre_ffn_norm, H, eps, l_gpu.layer_scalar);
        gpu.engine.recordBarrier(&gpu.buf_normed_x);
        gpu.engine.recordBarrier(&gpu.buf_x);

        // 7. FFN (Gate + Up + GeGLU)
        gpu.engine.recordGateUpSwiGlu(l_gpu.desc.gate_up_swiglu, inter, H);
        gpu.engine.recordBarrier(&gpu.buf_act);

        // 8. Down projection
        gpu.engine.recordGemv(l_gpu.desc.down_proj, H, inter);
        gpu.engine.recordBarrier(&gpu.buf_mlp_out);

        // 8.5 Post-FFN RMSNorm
        if (l_gpu.has_post_ffn_norm) {
            gpu.engine.recordRmsNorm(l_gpu.desc.post_ffn_norm, H, eps);
            gpu.engine.recordBarrier(&gpu.buf_mlp_out);
        }

        // 9. FFN residual + next RMSNorm: buf_x += scalar * buf_mlp_out -> buf_normed_x
        gpu.engine.recordAddRmsNorm(l_gpu.desc.post_ffn_add, H, eps, l_gpu.layer_scalar);
        gpu.engine.recordBarrier(&gpu.buf_normed_x);
        gpu.engine.recordBarrier(&gpu.buf_x);
    }

    // 10. Final RMSNorm: buf_x -> buf_normed_x
    gpu.engine.recordRmsNorm(gpu.desc_final_norm, H, eps);
    gpu.engine.recordBarrier(&gpu.buf_normed_x);

    // 11. Logits projection
    gpu.engine.recordGemvLogits(gpu.desc_logits, V, H);
    gpu.engine.submitBatch() catch return false;

    @memcpy(logits[0..V], gpu.buf_logits.asSlice(f32)[0..V]);
    return true;
}

pub fn gpuDispatchQkv(
    gpu: *model_gpu.GpuModelContext,
    layer_idx: usize,
    normed_x: []const f32,
    q: []f32,
    k: []f32,
    v: []f32,
    q_dim: usize,
    kv_dim: usize,
    H: usize,
) bool {
    const l = &gpu.layers[layer_idx];
    @memcpy(gpu.buf_normed_x.asSlice(f32)[0..H], normed_x[0..H]);

    gpu.engine.beginBatch();
    gpu.engine.recordGemv(l.desc.q_proj, q_dim, H);
    gpu.engine.recordGemv(l.desc.k_proj, kv_dim, H);
    gpu.engine.recordGemv(l.desc.v_proj, kv_dim, H);
    gpu.engine.submitBatch() catch return false;

    @memcpy(q[0..q_dim], gpu.buf_q.asSlice(f32)[0..q_dim]);
    @memcpy(k[0..kv_dim], gpu.buf_k.asSlice(f32)[0..kv_dim]);
    @memcpy(v[0..kv_dim], gpu.buf_v.asSlice(f32)[0..kv_dim]);
    return true;
}

pub fn gpuDispatchOProj(gpu: *model_gpu.GpuModelContext, layer_idx: usize, attn_out: []const f32, mlp_out: []f32, H: usize, q_dim: usize) bool {
    const l = &gpu.layers[layer_idx];
    @memcpy(gpu.buf_attn_out.asSlice(f32)[0..q_dim], attn_out[0..q_dim]);
    gpu.engine.beginBatch();
    gpu.engine.recordGemv(l.desc.o_proj, H, q_dim);
    gpu.engine.submitBatch() catch return false;
    @memcpy(mlp_out[0..H], gpu.buf_mlp_out.asSlice(f32)[0..H]);
    return true;
}

pub fn gpuDispatchLogits(gpu: *model_gpu.GpuModelContext, normed_x: []const f32, logits: []f32, V: usize, H: usize) bool {
    @memcpy(gpu.buf_normed_x.asSlice(f32)[0..H], normed_x[0..H]);
    gpu.engine.beginBatch();
    gpu.engine.recordGemvLogits(gpu.desc_logits, V, H);
    gpu.engine.submitBatch() catch return false;
    @memcpy(logits[0..V], gpu.buf_logits.asSlice(f32)[0..V]);
    return true;
}
