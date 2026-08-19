const std = @import("std");
pub const types = @import("types.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const kernels = @import("kernels.zig");
pub const model_types = @import("../model/types.zig");
pub const model_gpu = @import("model_gpu.zig");

pub fn gpuGemv(gpu: *model_gpu.GpuModelContext, w: *const buffer.GpuBuffer, x: []const f32, y: []f32, m: usize, k: usize) bool {
    @memcpy(gpu.buf_normed_x.asSlice(f32)[0..k], x[0..k]);
    gpu.engine.dispatchGemv(w, &gpu.buf_normed_x, &gpu.buf_act, m, k) catch return false;
    @memcpy(y[0..m], gpu.buf_act.asSlice(f32)[0..m]);
    return true;
}

pub fn gpuGatedMlp(gpu: *model_gpu.GpuModelContext, layer: usize, x: []const f32, out: []f32, h: usize, inter: usize) bool {
    const l = &gpu.layers[layer];
    @memcpy(gpu.buf_normed_x.asSlice(f32)[0..h], x[0..h]);
    gpu.engine.dispatchGemv(&l.gate_proj, &gpu.buf_normed_x, &gpu.buf_gate, inter, h) catch return false;
    gpu.engine.dispatchGemv(&l.up_proj, &gpu.buf_normed_x, &gpu.buf_up, inter, h) catch return false;
    gpu.engine.dispatchSwiGlu(&gpu.buf_gate, &gpu.buf_up, &gpu.buf_act, inter) catch return false;
    gpu.engine.dispatchGemv(&l.down_proj, &gpu.buf_act, &gpu.buf_mlp_out, h, inter) catch return false;
    @memcpy(out[0..h], gpu.buf_mlp_out.asSlice(f32)[0..h]);
    return true;
}

pub fn gpuLogits(gpu: *model_gpu.GpuModelContext, x: []const f32, logits: []f32, v: usize, h: usize) bool {
    @memcpy(gpu.buf_normed_x.asSlice(f32)[0..h], x[0..h]);
    gpu.engine.dispatchGemv(&gpu.embed_tokens, &gpu.buf_normed_x, &gpu.buf_logits, v, h) catch return false;
    @memcpy(logits[0..v], gpu.buf_logits.asSlice(f32)[0..v]);
    return true;
}
