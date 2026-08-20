const std = @import("std");
pub const types = @import("types.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const kernels = @import("kernels.zig");
pub const model_types = @import("../model/types.zig");
pub const model_gpu = @import("model_gpu.zig");

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

pub fn gpuDispatchOProj(
    gpu: *model_gpu.GpuModelContext,
    layer_idx: usize,
    attn_out: []const f32,
    mlp_out: []f32,
    H: usize,
    q_dim: usize,
) bool {
    const l = &gpu.layers[layer_idx];
    @memcpy(gpu.buf_attn_out.asSlice(f32)[0..q_dim], attn_out[0..q_dim]);

    gpu.engine.beginBatch();
    gpu.engine.recordGemv(l.desc.o_proj, H, q_dim);
    gpu.engine.submitBatch() catch return false;

    @memcpy(mlp_out[0..H], gpu.buf_mlp_out.asSlice(f32)[0..H]);
    return true;
}

pub fn gpuDispatchLayerFFN(
    gpu: *model_gpu.GpuModelContext,
    layer_idx: usize,
    attn_out: []const f32,
    x: []f32,
    H: usize,
    q_dim: usize,
    inter: usize,
    eps: f32,
) bool {
    const l = &gpu.layers[layer_idx];
    @memcpy(gpu.buf_attn_out.asSlice(f32)[0..q_dim], attn_out[0..q_dim]);
    @memcpy(gpu.buf_x.asSlice(f32)[0..H], x[0..H]);

    gpu.engine.beginBatch();
    gpu.engine.recordGemv(l.desc.o_proj, H, q_dim);
    gpu.engine.recordBarrier(&gpu.buf_mlp_out);

    if (l.has_post_attn_norm) {
        gpu.engine.recordRmsNorm(l.desc.post_attn_norm, H, eps);
        gpu.engine.recordBarrier(&gpu.buf_mlp_out);
    }

    gpu.engine.recordAddRmsNorm(l.desc.pre_ffn_norm, H, eps);
    gpu.engine.recordBarrier(&gpu.buf_normed_x);

    if (gpu.engine.mode == .q4) {
        gpu.engine.recordGateUpSwiGlu(l.desc.gate_up_swiglu, inter, H);
        gpu.engine.recordBarrier(&gpu.buf_act);
        gpu.engine.recordGemv(l.desc.down_proj, H, inter);
    } else {
        gpu.engine.recordGemv(l.desc.gate_proj, inter, H);
        gpu.engine.recordGemv(l.desc.up_proj, inter, H);
        gpu.engine.recordBarrier(&gpu.buf_gate);
        gpu.engine.recordBarrier(&gpu.buf_up);
        gpu.engine.recordSwiGlu(l.desc.swiglu, inter);
        gpu.engine.recordBarrier(&gpu.buf_act);
        gpu.engine.recordGemv(l.desc.down_proj, H, inter);
    }
    if (l.has_post_ffn_norm) {
        gpu.engine.recordBarrier(&gpu.buf_mlp_out);
        gpu.engine.recordRmsNorm(l.desc.post_ffn_norm, H, eps);
    }
    gpu.engine.submitBatch() catch return false;

    const x_post_attn = gpu.buf_x.asSlice(f32)[0..H];
    const mlp_out = gpu.buf_mlp_out.asSlice(f32)[0..H];
    for (x[0..H], x_post_attn, mlp_out) |*dst, a, m| dst.* = a + m;
    return true;
}

pub fn gpuDispatchMlp(
    gpu: *model_gpu.GpuModelContext,
    layer_idx: usize,
    normed_x: []const f32,
    mlp_out: []f32,
    H: usize,
    inter: usize,
) bool {
    const l = &gpu.layers[layer_idx];
    @memcpy(gpu.buf_normed_x.asSlice(f32)[0..H], normed_x[0..H]);

    gpu.engine.beginBatch();
    if (gpu.engine.mode == .q4) {
        gpu.engine.recordGateUpSwiGlu(l.desc.gate_up_swiglu, inter, H);
        gpu.engine.recordBarrier(&gpu.buf_act);
        gpu.engine.recordGemv(l.desc.down_proj, H, inter);
    } else {
        gpu.engine.recordGemv(l.desc.gate_proj, inter, H);
        gpu.engine.recordGemv(l.desc.up_proj, inter, H);
        gpu.engine.recordBarrier(&gpu.buf_gate);
        gpu.engine.recordBarrier(&gpu.buf_up);
        gpu.engine.recordSwiGlu(l.desc.swiglu, inter);
        gpu.engine.recordBarrier(&gpu.buf_act);
        gpu.engine.recordGemv(l.desc.down_proj, H, inter);
    }
    if (l.has_post_ffn_norm) {
        gpu.engine.recordBarrier(&gpu.buf_mlp_out);
        gpu.engine.recordRmsNorm(l.desc.post_ffn_norm, H, 1e-6);
    }
    gpu.engine.submitBatch() catch return false;

    @memcpy(mlp_out[0..H], gpu.buf_mlp_out.asSlice(f32)[0..H]);
    return true;
}

pub fn gpuDispatchLogits(
    gpu: *model_gpu.GpuModelContext,
    normed_x: []const f32,
    logits: []f32,
    V: usize,
    H: usize,
) bool {
    @memcpy(gpu.buf_normed_x.asSlice(f32)[0..H], normed_x[0..H]);

    gpu.engine.beginBatch();
    gpu.engine.recordGemvLogits(gpu.desc_logits, V, H);
    gpu.engine.submitBatch() catch return false;

    @memcpy(logits[0..V], gpu.buf_logits.asSlice(f32)[0..V]);
    return true;
}
