const std = @import("std");
pub const safetensors = @import("safetensors.zig");
pub const tensor = @import("tensor.zig");
pub const kernels = @import("kernels.zig");

const bf16 = tensor.bf16;

pub const LayerType = enum {
    sliding_attention,
    full_attention,
};

pub const LayerWeights = struct {
    layer_type: LayerType,
    head_dim: usize,
    rotary_dim: usize,
    q_dim: usize,
    kv_dim: usize,

    input_layernorm: []const bf16,
    q_proj: []const bf16,
    k_proj: []const bf16,
    v_proj: []const bf16,
    o_proj: []const bf16,
    q_norm: []const bf16,
    k_norm: []const bf16,
    post_attention_layernorm: ?[]const bf16 = null,

    pre_feedforward_layernorm: []const bf16,
    gate_proj: []const bf16,
    up_proj: []const bf16,
    down_proj: []const bf16,
    post_feedforward_layernorm: ?[]const bf16 = null,
    layer_scalar: ?f32 = null,

    // Optional Per-Layer Embedding weights (for E2B/E4B)
    per_layer_input_gate: ?[]const bf16 = null,
    per_layer_projection: ?[]const bf16 = null,
    post_per_layer_input_norm: ?[]const bf16 = null,
};

pub const ModelConfig = struct {
    vocab_size: usize = 262144,
    hidden_size: usize = 1536,
    intermediate_size: usize = 6144,
    hidden_size_per_layer_input: usize = 256,
    num_hidden_layers: usize = 35,
    num_attention_heads: usize = 8,
    num_key_value_heads: usize = 1,
    head_dim: usize = 256,
    global_head_dim: usize = 512,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 10000.0,
    rope_theta_full: f32 = 1000000.0,
    sliding_window: usize = 512,
    max_seq_len: usize = 4096,
};

pub const KVCache = struct {
    allocator: std.mem.Allocator,
    k: []f32,
    v: []f32,
    max_seq_len: usize,
    num_layers: usize,
    max_kv_dim: usize,

    pub fn init(allocator: std.mem.Allocator, num_layers: usize, max_seq_len: usize, max_kv_dim: usize) !KVCache {
        const total_elements = num_layers * max_seq_len * max_kv_dim;
        const k_buf = try allocator.alloc(f32, total_elements);
        const v_buf = try allocator.alloc(f32, total_elements);
        @memset(k_buf, 0);
        @memset(v_buf, 0);

        return KVCache{
            .allocator = allocator,
            .k = k_buf,
            .v = v_buf,
            .max_seq_len = max_seq_len,
            .num_layers = num_layers,
            .max_kv_dim = max_kv_dim,
        };
    }

    pub fn deinit(self: *KVCache) void {
        self.allocator.free(self.k);
        self.allocator.free(self.v);
    }

    pub fn getKV(self: *KVCache, layer: usize, pos: usize, kv_dim: usize) struct { k: []f32, v: []f32 } {
        const offset = (layer * self.max_seq_len + pos) * self.max_kv_dim;
        return .{
            .k = self.k[offset .. offset + kv_dim],
            .v = self.v[offset .. offset + kv_dim],
        };
    }
};

pub const ForwardScratch = struct {
    x: []f32,
    normed_x: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn_scores: []f32,
    attn_out: []f32,
    mlp_gate_up: []f32,
    mlp_out: []f32,
    ple_buf_1: []f32,
    ple_buf_2: []f32,
    logits: []f32,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig) !ForwardScratch {
        const max_head_dim = @max(config.head_dim, config.global_head_dim);
        const max_q_dim = config.num_attention_heads * max_head_dim;
        const max_kv_dim = config.num_key_value_heads * max_head_dim;

        return ForwardScratch{
            .x = try allocator.alloc(f32, config.hidden_size),
            .normed_x = try allocator.alloc(f32, config.hidden_size),
            .q = try allocator.alloc(f32, max_q_dim),
            .k = try allocator.alloc(f32, max_kv_dim),
            .v = try allocator.alloc(f32, max_kv_dim),
            .attn_scores = try allocator.alloc(f32, config.max_seq_len),
            .attn_out = try allocator.alloc(f32, @max(max_q_dim, config.hidden_size)),
            .mlp_gate_up = try allocator.alloc(f32, config.intermediate_size),
            .mlp_out = try allocator.alloc(f32, config.hidden_size),
            .ple_buf_1 = try allocator.alloc(f32, config.hidden_size_per_layer_input),
            .ple_buf_2 = try allocator.alloc(f32, config.hidden_size),
            .logits = try allocator.alloc(f32, config.vocab_size),
        };
    }

    pub fn deinit(self: *ForwardScratch, allocator: std.mem.Allocator) void {
        allocator.free(self.x);
        allocator.free(self.normed_x);
        allocator.free(self.q);
        allocator.free(self.k);
        allocator.free(self.v);
        allocator.free(self.attn_scores);
        allocator.free(self.attn_out);
        allocator.free(self.mlp_gate_up);
        allocator.free(self.mlp_out);
        allocator.free(self.ple_buf_1);
        allocator.free(self.ple_buf_2);
        allocator.free(self.logits);
    }
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    config: ModelConfig,
    embed_tokens: []const bf16,
    embed_tokens_per_layer: ?[]const bf16 = null,
    per_layer_projection_norm: ?[]const bf16 = null,
    layers: []LayerWeights,
    final_norm: []const bf16,

    pub fn loadFromSafeTensors(allocator: std.mem.Allocator, st: *const safetensors.SafeTensors, config: ModelConfig) !Model {
        const embed_tv = st.get("model.language_model.embed_tokens.weight") orelse return error.MissingEmbedTokens;
        const norm_tv = st.get("model.language_model.norm.weight") orelse return error.MissingFinalNorm;

        const embed_slice = embed_tv.asSlice(bf16);
        const norm_slice = norm_tv.asSlice(bf16);

        const ple_tv = st.get("model.language_model.embed_tokens_per_layer.weight");
        const ple_slice = if (ple_tv) |p| p.asSlice(bf16) else null;

        const plpn_tv = st.get("model.language_model.per_layer_projection_norm.weight");
        const plpn_slice = if (plpn_tv) |p| p.asSlice(bf16) else null;

        var layers = try allocator.alloc(LayerWeights, config.num_hidden_layers);

        var buf: [128]u8 = undefined;
        for (0..config.num_hidden_layers) |l| {
            const is_full = ((l + 1) % 5 == 0);
            const head_dim = if (is_full) config.global_head_dim else config.head_dim;
            const rotary_dim = if (is_full) head_dim / 4 else head_dim; // Partial RoPE 0.25 on full attention
            const q_dim = config.num_attention_heads * head_dim;
            const kv_dim = config.num_key_value_heads * head_dim;

            const in_norm_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.input_layernorm.weight", .{l});
            const in_norm = (st.get(in_norm_name) orelse return error.MissingLayerWeight).asSlice(bf16);

            const q_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.self_attn.q_proj.weight", .{l});
            const q = (st.get(q_name) orelse return error.MissingLayerWeight).asSlice(bf16);

            const k_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.self_attn.k_proj.weight", .{l});
            const k = (st.get(k_name) orelse return error.MissingLayerWeight).asSlice(bf16);

            const v_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.self_attn.v_proj.weight", .{l});
            const v = (st.get(v_name) orelse return error.MissingLayerWeight).asSlice(bf16);

            const o_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.self_attn.o_proj.weight", .{l});
            const o = (st.get(o_name) orelse return error.MissingLayerWeight).asSlice(bf16);

            const qn_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.self_attn.q_norm.weight", .{l});
            const qn = (st.get(qn_name) orelse return error.MissingLayerWeight).asSlice(bf16);

            const kn_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.self_attn.k_norm.weight", .{l});
            const kn = (st.get(kn_name) orelse return error.MissingLayerWeight).asSlice(bf16);

            const post_attn_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.post_attention_layernorm.weight", .{l});
            const post_attn_tv = st.get(post_attn_name);
            const post_attn = if (post_attn_tv) |p| p.asSlice(bf16) else null;

            const pre_ffn_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.pre_feedforward_layernorm.weight", .{l});
            const pre_ffn = (st.get(pre_ffn_name) orelse return error.MissingLayerWeight).asSlice(bf16);

            const gate_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.gate_proj.weight", .{l});
            const gate = (st.get(gate_name) orelse return error.MissingLayerWeight).asSlice(bf16);

            const up_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.up_proj.weight", .{l});
            const up = (st.get(up_name) orelse return error.MissingLayerWeight).asSlice(bf16);

            const down_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.down_proj.weight", .{l});
            const down = (st.get(down_name) orelse return error.MissingLayerWeight).asSlice(bf16);

            const post_ffn_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.post_feedforward_layernorm.weight", .{l});
            const post_ffn_tv = st.get(post_ffn_name);
            const post_ffn = if (post_ffn_tv) |p| p.asSlice(bf16) else null;

            const scalar_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.layer_scalar", .{l});
            const scalar_tv = st.get(scalar_name);
            const layer_scalar = if (scalar_tv) |s| s.asSlice(bf16)[0].toF32() else null;

            // Optional PLE weights
            const plg_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.per_layer_input_gate.weight", .{l});
            const plg_tv = st.get(plg_name);
            const pl_gate = if (plg_tv) |g| g.asSlice(bf16) else null;

            const plp_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.per_layer_projection.weight", .{l});
            const plp_tv = st.get(plp_name);
            const pl_proj = if (plp_tv) |p| p.asSlice(bf16) else null;

            const pln_name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.post_per_layer_input_norm.weight", .{l});
            const pln_tv = st.get(pln_name);
            const pl_norm = if (pln_tv) |n| n.asSlice(bf16) else null;

            layers[l] = LayerWeights{
                .layer_type = if (is_full) .full_attention else .sliding_attention,
                .head_dim = head_dim,
                .rotary_dim = rotary_dim,
                .q_dim = q_dim,
                .kv_dim = kv_dim,
                .input_layernorm = in_norm,
                .q_proj = q,
                .k_proj = k,
                .v_proj = v,
                .o_proj = o,
                .q_norm = qn,
                .k_norm = kn,
                .post_attention_layernorm = post_attn,
                .pre_feedforward_layernorm = pre_ffn,
                .gate_proj = gate,
                .up_proj = up,
                .down_proj = down,
                .post_feedforward_layernorm = post_ffn,
                .layer_scalar = layer_scalar,
                .per_layer_input_gate = pl_gate,
                .per_layer_projection = pl_proj,
                .post_per_layer_input_norm = pl_norm,
            };
        }

        return Model{
            .allocator = allocator,
            .config = config,
            .embed_tokens = embed_slice,
            .embed_tokens_per_layer = ple_slice,
            .per_layer_projection_norm = plpn_slice,
            .layers = layers,
            .final_norm = norm_slice,
        };
    }

    pub fn deinit(self: *Model) void {
        self.allocator.free(self.layers);
    }

    /// Run forward pass for a single token at position `pos`
    pub fn forwardToken(
        self: *const Model,
        cache: *KVCache,
        scratch: *ForwardScratch,
        token_id: u32,
        pos: usize,
        thread_pool: ?*std.Thread.Pool,
    ) void {
        const H = self.config.hidden_size;
        const ple_dim = self.config.hidden_size_per_layer_input;
        const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));

        // 1. Embedding lookup & scale
        const emb_offset = @as(usize, token_id) * H;
        const emb_row = self.embed_tokens[emb_offset .. emb_offset + H];
        for (scratch.x, emb_row) |*out, e| {
            out.* = e.toF32() * embed_scale;
        }

        // 2. Pass through transformer layers
        for (self.layers, 0..) |l, layer_idx| {
            // Optional Per-Layer Embedding fusion (if present)
            if (self.embed_tokens_per_layer) |ple_table| {
                if (l.per_layer_input_gate != null and l.per_layer_projection != null and l.post_per_layer_input_norm != null) {
                    const ple_row_offset = (@as(usize, token_id) * self.layers.len + layer_idx) * ple_dim;
                    const ple_row = ple_table[ple_row_offset .. ple_row_offset + ple_dim];

                    // gate = sigmoid(gemv(per_layer_input_gate, x))
                    kernels.gemv(scratch.ple_buf_1, scratch.x, l.per_layer_input_gate.?, ple_dim, H);
                    for (scratch.ple_buf_1, ple_row) |*g, p| {
                        g.* = kernels.sigmoid(g.*) * p.toF32();
                    }

                    // proj = gemv(per_layer_projection, gated)
                    kernels.gemv(scratch.ple_buf_2, scratch.ple_buf_1, l.per_layer_projection.?, H, ple_dim);

                    // norm and add
                    kernels.rmsNorm(scratch.ple_buf_2, scratch.ple_buf_2, l.post_per_layer_input_norm.?, self.config.rms_norm_eps);
                    for (scratch.x, scratch.ple_buf_2) |*x_val, p_val| {
                        x_val.* += p_val;
                    }
                }
            }

            // --- Attention Sub-layer ---
            // Input RMSNorm
            kernels.rmsNorm(scratch.normed_x, scratch.x, l.input_layernorm, self.config.rms_norm_eps);

            // Q, K, V Projections
            if (thread_pool) |tp| {
                kernels.gemvParallel(scratch.q[0..l.q_dim], scratch.normed_x, l.q_proj, l.q_dim, H, tp);
                kernels.gemvParallel(scratch.k[0..l.kv_dim], scratch.normed_x, l.k_proj, l.kv_dim, H, tp);
                kernels.gemvParallel(scratch.v[0..l.kv_dim], scratch.normed_x, l.v_proj, l.kv_dim, H, tp);
            } else {
                kernels.gemv(scratch.q[0..l.q_dim], scratch.normed_x, l.q_proj, l.q_dim, H);
                kernels.gemv(scratch.k[0..l.kv_dim], scratch.normed_x, l.k_proj, l.kv_dim, H);
                kernels.gemv(scratch.v[0..l.kv_dim], scratch.normed_x, l.v_proj, l.kv_dim, H);
            }

            // Q-Norm and K-Norm (per head)
            for (0..self.config.num_attention_heads) |h| {
                const head_q = scratch.q[h * l.head_dim .. (h + 1) * l.head_dim];
                kernels.rmsNorm(head_q, head_q, l.q_norm, self.config.rms_norm_eps);
            }
            kernels.rmsNorm(scratch.k[0..l.kv_dim], scratch.k[0..l.kv_dim], l.k_norm, self.config.rms_norm_eps);

            // RoPE on Q and K with partial rotary dimension
            const theta = if (l.layer_type == .full_attention) self.config.rope_theta_full else self.config.rope_theta;
            kernels.applyRopePartial(scratch.q[0..l.q_dim], pos, l.head_dim, l.rotary_dim, theta);
            kernels.applyRopePartial(scratch.k[0..l.kv_dim], pos, l.head_dim, l.rotary_dim, theta);

            // Store K, V in cache at current pos
            const kv = cache.getKV(layer_idx, pos, l.kv_dim);
            @memcpy(kv.k, scratch.k[0..l.kv_dim]);
            @memcpy(kv.v, scratch.v[0..l.kv_dim]);

            // Multi-Head Attention Calculation
            const attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(l.head_dim)));
            const history_len = pos + 1;
            const start_p = if (l.layer_type == .sliding_attention and history_len > self.config.sliding_window)
                history_len - self.config.sliding_window
            else
                0;
            const active_p_count = history_len - start_p;

            for (0..self.config.num_attention_heads) |h| {
                const q_head = scratch.q[h * l.head_dim .. (h + 1) * l.head_dim];
                const head_scores = scratch.attn_scores[0..active_p_count];

                // Compute scores q · k_p
                for (start_p..history_len, 0..) |p, i| {
                    const k_hist = cache.getKV(layer_idx, p, l.kv_dim).k;
                    var dot: f32 = 0.0;
                    for (q_head, k_hist) |q_val, k_val| {
                        dot += q_val * k_val;
                    }
                    head_scores[i] = dot * attn_scale;
                }

                kernels.softmax(head_scores);

                // Accumulate weighted values: sum(score_p * v_p)
                const out_head = scratch.attn_out[h * l.head_dim .. (h + 1) * l.head_dim];
                @memset(out_head, 0);

                for (start_p..history_len, 0..) |p, i| {
                    const weight = head_scores[i];
                    const v_hist = cache.getKV(layer_idx, p, l.kv_dim).v;
                    for (out_head, v_hist) |*o, v_val| {
                        o.* += weight * v_val;
                    }
                }
            }

            // Output projection: gemv(W_o, attn_out) -> mlp_out as temp
            if (thread_pool) |tp| {
                kernels.gemvParallel(scratch.mlp_out, scratch.attn_out[0..l.q_dim], l.o_proj, H, l.q_dim, tp);
            } else {
                kernels.gemv(scratch.mlp_out, scratch.attn_out[0..l.q_dim], l.o_proj, H, l.q_dim);
            }

            // Post-Attention Layernorm
            if (l.post_attention_layernorm) |pal| {
                kernels.rmsNorm(scratch.mlp_out, scratch.mlp_out, pal, self.config.rms_norm_eps);
            }

            // Residual add with scalar
            const scalar = l.layer_scalar orelse 1.0;
            for (scratch.x, scratch.mlp_out) |*x_val, attn_v| {
                x_val.* += attn_v * scalar;
            }

            // --- Feed-Forward (MLP) Sub-layer ---
            kernels.rmsNorm(scratch.normed_x, scratch.x, l.pre_feedforward_layernorm, self.config.rms_norm_eps);
            kernels.gatedMlp(
                scratch.mlp_out,
                scratch.normed_x,
                l.gate_proj,
                l.up_proj,
                l.down_proj,
                H,
                self.config.intermediate_size,
                scratch.mlp_gate_up,
            );

            // Post-Feedforward Layernorm
            if (l.post_feedforward_layernorm) |pfl| {
                kernels.rmsNorm(scratch.mlp_out, scratch.mlp_out, pfl, self.config.rms_norm_eps);
            }

            // Residual add
            for (scratch.x, scratch.mlp_out) |*x_val, ffn_v| {
                x_val.* += ffn_v * scalar;
            }
        }

        // 3. Final RMSNorm
        kernels.rmsNorm(scratch.normed_x, scratch.x, self.final_norm, self.config.rms_norm_eps);

        // 4. Output projection (tied embeddings): logits = W_embed * final_normed_x
        if (thread_pool) |tp| {
            kernels.gemvParallel(scratch.logits, scratch.normed_x, self.embed_tokens, self.config.vocab_size, H, tp);
        } else {
            kernels.gemv(scratch.logits, scratch.normed_x, self.embed_tokens, self.config.vocab_size, H);
        }

        // Gemma final logit softcapping (cap = 30.0)
        const cap: f32 = 30.0;
        const inv_cap: f32 = 1.0 / cap;
        for (scratch.logits) |*logit| {
            logit.* = cap * std.math.tanh(logit.* * inv_cap);
        }
    }
};
