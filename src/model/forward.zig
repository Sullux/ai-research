const std = @import("std");
pub const tensor = @import("../tensor.zig");
pub const kernels = @import("../kernels.zig");
pub const types = @import("types.zig");
pub const loader = @import("loader.zig");

const bf16 = tensor.bf16;
const Model = loader.Model;
const LayerWeights = types.LayerWeights;
const KVCache = types.KVCache;
const ForwardScratch = types.ForwardScratch;

pub fn forwardToken(self: *const Model, cache: *KVCache, scratch: *ForwardScratch, token_id: u32, pos: usize, tp: ?*std.Thread.Pool) void {
    const H = self.config.hidden_size;
    const ple_dim = self.config.hidden_size_per_layer_input;
    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));

    // 1. Embedding lookup & scale
    const emb_offset = @as(usize, token_id) * H;
    for (scratch.x, self.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| {
        out.* = e.toF32() * embed_scale;
    }

    // 2. Precompute Per-Layer Embeddings (PLE) if present
    preparePLE(self, scratch, token_id, H, ple_dim, tp);

    // 3. Decoder layers
    for (self.layers, 0..) |l, layer_idx| {
        forwardAttention(self, l, cache, scratch, layer_idx, pos, H, tp);
        forwardMLP(self, l, scratch, H, tp);
        forwardPLE(self, l, scratch, layer_idx, ple_dim, H);

        if (l.layer_scalar) |s| {
            for (scratch.x) |*x_val| x_val.* *= s;
        }
    }

    // 4. Final RMSNorm & Output Logits Projection
    kernels.rmsNorm(scratch.normed_x, scratch.x, self.final_norm, self.config.rms_norm_eps);
    if (tp) |pool| {
        kernels.gemvParallel(scratch.logits, scratch.normed_x, self.embed_tokens, self.config.vocab_size, H, pool);
    } else {
        kernels.gemv(scratch.logits, scratch.normed_x, self.embed_tokens, self.config.vocab_size, H);
    }

    const cap: f32 = 30.0;
    const inv_cap: f32 = 1.0 / cap;
    for (scratch.logits) |*logit| {
        logit.* = cap * std.math.tanh(logit.* * inv_cap);
    }
}

fn preparePLE(self: *const Model, scratch: *ForwardScratch, token_id: u32, H: usize, ple_dim: usize, tp: ?*std.Thread.Pool) void {
    const ple_table = self.embed_tokens_per_layer orelse return;
    const total_ple_dim = self.layers.len * ple_dim;
    const ple_scale = @sqrt(@as(f32, @floatFromInt(ple_dim)));
    const inv_sqrt_2: f32 = 0.70710678118;

    if (self.per_layer_model_projection) |plmp| {
        if (tp) |pool| {
            kernels.gemvParallel(scratch.ple_context, scratch.x, plmp, total_ple_dim, H, pool);
        } else {
            kernels.gemv(scratch.ple_context, scratch.x, plmp, total_ple_dim, H);
        }

        const proj_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(H)));
        for (scratch.ple_context) |*v| v.* *= proj_scale;

        if (self.per_layer_projection_norm) |plpn| {
            for (0..self.layers.len) |l| {
                const layer_ctx = scratch.ple_context[l * ple_dim .. (l + 1) * ple_dim];
                kernels.rmsNorm(layer_ctx, layer_ctx, plpn, self.config.rms_norm_eps);
            }
        }

        const tok_ple_offset = @as(usize, token_id) * total_ple_dim;
        const tok_ple_row = ple_table[tok_ple_offset .. tok_ple_offset + total_ple_dim];

        for (scratch.ple_context, tok_ple_row) |*ctx, tok_p| {
            ctx.* = (ctx.* + tok_p.toF32() * ple_scale) * inv_sqrt_2;
        }
    }
}

fn forwardAttention(self: *const Model, l: LayerWeights, cache: *KVCache, scratch: *ForwardScratch, layer_idx: usize, pos: usize, H: usize, tp: ?*std.Thread.Pool) void {
    kernels.rmsNorm(scratch.normed_x, scratch.x, l.input_layernorm, self.config.rms_norm_eps);

    if (tp) |pool| {
        kernels.gemvParallel(scratch.q[0..l.q_dim], scratch.normed_x, l.q_proj, l.q_dim, H, pool);
    } else {
        kernels.gemv(scratch.q[0..l.q_dim], scratch.normed_x, l.q_proj, l.q_dim, H);
    }

    for (0..self.config.num_attention_heads) |h| {
        const head_q = scratch.q[h * l.head_dim .. (h + 1) * l.head_dim];
        kernels.rmsNorm(head_q, head_q, l.q_norm, self.config.rms_norm_eps);
    }

    const theta = if (l.layer_type == .full_attention) self.config.rope_theta_full else self.config.rope_theta;
    kernels.applyRopePartial(scratch.q[0..l.q_dim], pos, l.head_dim, l.rotary_dim, theta);

    // KV Cache Source: Layers 0..14 compute their own KV. Layers 15..34 reuse layer 13 (sliding) or layer 14 (full).
    const is_shared = (layer_idx >= 15);
    const kv_layer = if (is_shared) (if (l.layer_type == .full_attention) @as(usize, 14) else @as(usize, 13)) else layer_idx;

    if (!is_shared) {
        if (tp) |pool| {
            kernels.gemvParallel(scratch.k[0..l.kv_dim], scratch.normed_x, l.k_proj, l.kv_dim, H, pool);
            kernels.gemvParallel(scratch.v[0..l.kv_dim], scratch.normed_x, l.v_proj, l.kv_dim, H, pool);
        } else {
            kernels.gemv(scratch.k[0..l.kv_dim], scratch.normed_x, l.k_proj, l.kv_dim, H);
            kernels.gemv(scratch.v[0..l.kv_dim], scratch.normed_x, l.v_proj, l.kv_dim, H);
        }

        kernels.rmsNorm(scratch.k[0..l.kv_dim], scratch.k[0..l.kv_dim], l.k_norm, self.config.rms_norm_eps);
        kernels.unitRmsNorm(scratch.v[0..l.kv_dim], scratch.v[0..l.kv_dim], self.config.rms_norm_eps);
        kernels.applyRopePartial(scratch.k[0..l.kv_dim], pos, l.head_dim, l.rotary_dim, theta);

        const kv = cache.getKV(kv_layer, pos, l.kv_dim);
        @memcpy(kv.k, scratch.k[0..l.kv_dim]);
        @memcpy(kv.v, scratch.v[0..l.kv_dim]);
    }

    const history_len = pos + 1;
    const start_p = if (l.layer_type == .sliding_attention and history_len > self.config.sliding_window) history_len - self.config.sliding_window else 0;
    const active_p_count = history_len - start_p;

    for (0..self.config.num_attention_heads) |h| {
        const q_head = scratch.q[h * l.head_dim .. (h + 1) * l.head_dim];
        const head_scores = scratch.attn_scores[0..active_p_count];

        for (start_p..history_len, 0..) |p, i| {
            const k_hist = cache.getKV(kv_layer, p, l.kv_dim).k;
            var dot: f32 = 0.0;
            for (q_head, k_hist) |q_val, k_val| dot += q_val * k_val;
            head_scores[i] = dot;
        }

        kernels.softmax(head_scores);
        const out_head = scratch.attn_out[h * l.head_dim .. (h + 1) * l.head_dim];
        @memset(out_head, 0);

        for (start_p..history_len, 0..) |p, i| {
            const weight = head_scores[i];
            const v_hist = cache.getKV(kv_layer, p, l.kv_dim).v;
            for (out_head, v_hist) |*o, v_val| o.* += weight * v_val;
        }
    }

    if (tp) |pool| {
        kernels.gemvParallel(scratch.mlp_out, scratch.attn_out[0..l.q_dim], l.o_proj, H, l.q_dim, pool);
    } else {
        kernels.gemv(scratch.mlp_out, scratch.attn_out[0..l.q_dim], l.o_proj, H, l.q_dim);
    }

    if (l.post_attention_layernorm) |pal| kernels.rmsNorm(scratch.mlp_out, scratch.mlp_out, pal, self.config.rms_norm_eps);

    for (scratch.x, scratch.mlp_out) |*x_val, attn_v| x_val.* += attn_v;
}

fn forwardMLP(self: *const Model, l: LayerWeights, scratch: *ForwardScratch, H: usize, tp: ?*std.Thread.Pool) void {
    kernels.rmsNorm(scratch.normed_x, scratch.x, l.pre_feedforward_layernorm, self.config.rms_norm_eps);
    kernels.gatedMlp(scratch.mlp_out, scratch.normed_x, l.gate_proj, l.up_proj, l.down_proj, H, l.intermediate_dim, scratch.mlp_gate_up, tp);
    if (l.post_feedforward_layernorm) |pfl| kernels.rmsNorm(scratch.mlp_out, scratch.mlp_out, pfl, self.config.rms_norm_eps);
    for (scratch.x, scratch.mlp_out) |*x_val, ffn_v| x_val.* += ffn_v;
}

fn forwardPLE(self: *const Model, l: LayerWeights, scratch: *ForwardScratch, layer_idx: usize, ple_dim: usize, H: usize) void {
    if (self.embed_tokens_per_layer == null) return;
    if (l.per_layer_input_gate == null or l.per_layer_projection == null or l.post_per_layer_input_norm == null) return;

    // 1. Gate on un-normed scratch.x with GeLU
    kernels.gemv(scratch.ple_buf_1, scratch.x, l.per_layer_input_gate.?, ple_dim, H);

    const layer_ple_ctx = scratch.ple_context[layer_idx * ple_dim .. (layer_idx + 1) * ple_dim];
    for (scratch.ple_buf_1, layer_ple_ctx) |*g, ctx_val| {
        g.* = kernels.geluTanh(g.*) * ctx_val;
    }

    // 2. Project back to hidden_size & post RMSNorm
    kernels.gemv(scratch.ple_buf_2, scratch.ple_buf_1, l.per_layer_projection.?, H, ple_dim);
    kernels.rmsNorm(scratch.ple_buf_2, scratch.ple_buf_2, l.post_per_layer_input_norm.?, self.config.rms_norm_eps);

    for (scratch.x, scratch.ple_buf_2) |*x_val, p_val| x_val.* += p_val;
}
