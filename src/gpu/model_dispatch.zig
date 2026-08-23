const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const kernels = @import("kernels.zig");
pub const model_types = @import("../model/types.zig");
pub const model_gpu = @import("model_gpu.zig");

fn recordLayerDirect(gpu: *model_gpu.GpuModelContext, l_gpu: model_gpu.GpuLayerWeights, l_cpu: model_types.LayerWeights, config: *const model_types.ModelConfig, cmd: types.VkCommandBuffer, idx: usize) void {
    const H = config.hidden_size;
    const inter = config.intermediate_size;
    const eps = config.rms_norm_eps;
    const q_dim = l_cpu.q_dim;
    const kv_dim = l_cpu.kv_dim;
    const head_dim: u32 = @intCast(l_cpu.head_dim);
    const rot_dim: u32 = @intCast(l_cpu.rotary_dim);
    const theta: f32 = if (l_cpu.layer_type == .full_attention) config.rope_theta_full else config.rope_theta;
    const gqa_ratio: u32 = @intCast(config.num_attention_heads / l_cpu.num_kv_heads);

    if (idx == 0) {
        gpu.engine.recordRmsNorm(cmd, l_gpu.desc.input_norm, H, eps);
        gpu.engine.recordBarrier(cmd);
    }
    gpu.engine.recordGemv(cmd, l_gpu.desc.q_proj, q_dim, H);
    gpu.engine.recordGemv(cmd, l_gpu.desc.k_proj, kv_dim, H);
    gpu.engine.recordGemv(cmd, l_gpu.desc.v_proj, kv_dim, H);
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordQkvRope(cmd, l_gpu.desc.qkv_rope, config.num_attention_heads, l_cpu.num_kv_heads, head_dim, rot_dim, l_cpu.k_eq_v, theta, eps);
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordDecodeAttn(cmd, l_gpu.desc.attn, head_dim, kv_dim, gqa_ratio, 1.0, config.num_attention_heads);
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordGemv(cmd, l_gpu.desc.o_proj, H, q_dim);
    if (l_gpu.has_post_attn_norm) { gpu.engine.recordBarrier(cmd); gpu.engine.recordRmsNorm(cmd, l_gpu.desc.post_attn_norm, H, eps); }
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordAddRmsNorm(cmd, l_gpu.desc.pre_ffn_norm, H, eps, 1.0);
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordGateUpSwiGlu(cmd, l_gpu.desc.gate_up_swiglu, inter, H);
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordGemv(cmd, l_gpu.desc.down_proj, H, inter);
    if (l_gpu.has_post_ffn_norm) { gpu.engine.recordBarrier(cmd); gpu.engine.recordRmsNorm(cmd, l_gpu.desc.post_ffn_norm, H, eps); }
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordAddRmsNorm(cmd, l_gpu.desc.post_ffn_add, H, eps, l_gpu.layer_scalar);
    gpu.engine.recordBarrier(cmd);
}

fn recordLayerIndirect(gpu: *model_gpu.GpuModelContext, l_gpu: model_gpu.GpuLayerWeights, l_cpu: model_types.LayerWeights, config: *const model_types.ModelConfig, cmd: types.VkCommandBuffer, idx: usize, thresh: f32) void {
    const H = config.hidden_size;
    const inter = config.intermediate_size;
    const eps = config.rms_norm_eps;
    const q_dim = l_cpu.q_dim;
    const kv_dim = l_cpu.kv_dim;
    const head_dim: u32 = @intCast(l_cpu.head_dim);
    const rot_dim: u32 = @intCast(l_cpu.rotary_dim);
    const theta: f32 = if (l_cpu.layer_type == .full_attention) config.rope_theta_full else config.rope_theta;
    const gqa_ratio: u32 = @intCast(config.num_attention_heads / l_cpu.num_kv_heads);

    var pc_gate = kernels.QuiescenceGatePushConstants{
        .hidden_size = @intCast(H), .threshold_sq = thresh, .base_cmd_idx = @intCast(idx * 16), .num_cmds = 13,
        .targets = std.mem.zeroes([16]u32),
    };
    pc_gate.targets[0] = 1;
    pc_gate.targets[1] = @intCast((q_dim + 3) / 4);
    pc_gate.targets[2] = @intCast((kv_dim + 3) / 4);
    pc_gate.targets[3] = @intCast((kv_dim + 3) / 4);
    pc_gate.targets[4] = @intCast(config.num_attention_heads + l_cpu.num_kv_heads);
    pc_gate.targets[5] = @intCast(config.num_attention_heads);
    pc_gate.targets[6] = @intCast((H + 3) / 4);
    pc_gate.targets[7] = 1;
    pc_gate.targets[8] = 1;
    pc_gate.targets[9] = @intCast((inter + 3) / 4);
    pc_gate.targets[10] = @intCast((H + 3) / 4);
    pc_gate.targets[11] = 1;
    pc_gate.targets[12] = 1;

    gpu.engine.recordQuiescenceGate(cmd, l_gpu.desc.quiescence_gate, &pc_gate);
    gpu.engine.recordBarrier(cmd);

    const buf = gpu.buf_indirect_cmds.buffer;
    const base: u64 = @intCast(idx * 16 * 12);
    gpu.engine.recordRmsNormIndirect(cmd, l_gpu.desc.input_norm, H, eps, buf, base + 0 * 12);
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordGemvIndirect(cmd, l_gpu.desc.q_proj, q_dim, H, buf, base + 1 * 12);
    gpu.engine.recordGemvIndirect(cmd, l_gpu.desc.k_proj, kv_dim, H, buf, base + 2 * 12);
    gpu.engine.recordGemvIndirect(cmd, l_gpu.desc.v_proj, kv_dim, H, buf, base + 3 * 12);
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordQkvRopeIndirect(cmd, l_gpu.desc.qkv_rope, config.num_attention_heads, l_cpu.num_kv_heads, head_dim, rot_dim, l_cpu.k_eq_v, theta, eps, buf, base + 4 * 12);
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordDecodeAttnIndirect(cmd, l_gpu.desc.attn, head_dim, kv_dim, gqa_ratio, 1.0, buf, base + 5 * 12);
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordGemvIndirect(cmd, l_gpu.desc.o_proj, H, q_dim, buf, base + 6 * 12);
    if (l_gpu.has_post_attn_norm) { gpu.engine.recordBarrier(cmd); gpu.engine.recordRmsNormIndirect(cmd, l_gpu.desc.post_attn_norm, H, eps, buf, base + 7 * 12); }
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordAddRmsNormIndirect(cmd, l_gpu.desc.pre_ffn_norm, H, eps, 1.0, buf, base + 8 * 12);
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordGateUpSwiGluIndirect(cmd, l_gpu.desc.gate_up_swiglu, inter, H, buf, base + 9 * 12);
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordGemvIndirect(cmd, l_gpu.desc.down_proj, H, inter, buf, base + 10 * 12);
    if (l_gpu.has_post_ffn_norm) { gpu.engine.recordBarrier(cmd); gpu.engine.recordRmsNormIndirect(cmd, l_gpu.desc.post_ffn_norm, H, eps, buf, base + 11 * 12); }
    gpu.engine.recordBarrier(cmd);
    gpu.engine.recordAddRmsNormIndirect(cmd, l_gpu.desc.post_ffn_add, H, eps, l_gpu.layer_scalar, buf, base + 12 * 12);
    gpu.engine.recordBarrier(cmd);
}

pub fn recordForwardGraph(gpu: *model_gpu.GpuModelContext, config: *const model_types.ModelConfig, layers: []const model_types.LayerWeights, cmd_buf: types.VkCommandBuffer, include_logits: bool, quiescence_thresh: f32) void {
    const H = config.hidden_size;
    const V = config.vocab_size;
    const dense_limit = config.num_hidden_layers / 3;

    const begin_info = types_dispatch.VkCommandBufferBeginInfo{};
    _ = gpu.ctx.api.vkBeginCommandBuffer(cmd_buf, &begin_info);
    gpu.engine.recordBarrier(cmd_buf);

    for (gpu.layers, 0..) |l_gpu, i| {
        if (i < dense_limit) {
            recordLayerDirect(gpu, l_gpu, layers[i], config, cmd_buf, i);
        } else {
            recordLayerIndirect(gpu, l_gpu, layers[i], config, cmd_buf, i, quiescence_thresh);
        }
    }

    if (include_logits) {
        gpu.engine.recordGemvLogits(cmd_buf, gpu.desc_logits, V, H);
        gpu.engine.recordBarrier(cmd_buf);
        gpu.engine.recordArgmax(cmd_buf, gpu.desc_argmax, V);
    }
    _ = gpu.ctx.api.vkEndCommandBuffer(cmd_buf);
}

pub fn gpuDispatchForwardToken(gpu: *model_gpu.GpuModelContext, config: *const model_types.ModelConfig, layers: []const model_types.LayerWeights, x: []const f32, logits: []f32, clock: usize, slot_idx: usize, active_slots: []const usize) u32 {
    _ = layers;
    const H = config.hidden_size;
    const V = config.vocab_size;

    @memcpy(gpu.buf_x.asSlice(f32)[0..H], x[0..H]);
    for (gpu.buf_active_slots.asSlice(u32)[0..active_slots.len], active_slots) |*dst, s| dst.* = @intCast(s);
    const params = gpu.buf_step_params.asSlice(u32);
    params[0] = @intCast(clock);
    params[1] = @intCast(slot_idx);
    params[2] = @intCast(active_slots.len);
    params[3] = 0;

    const cmd_buf = if (logits.len >= V) gpu.cmd_buf_decode else gpu.cmd_buf_prefill;
    gpu.engine.submitPreRecorded(cmd_buf) catch return 0;
    return if (logits.len >= V) gpu.buf_sampled_token.asSlice(u32)[0] else 0;
}
