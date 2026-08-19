const std = @import("std");
pub const safetensors = @import("safetensors.zig");
pub const tensor = @import("tensor.zig");
pub const kernels = @import("kernels.zig");

const bf16 = tensor.bf16;

pub const ModelConfig = struct {
    vocab_size: usize = 262144,
    hidden_size: usize = 1536,
    intermediate_size: usize = 6144,
    num_hidden_layers: usize = 35,
    num_attention_heads: usize = 8,
    num_key_value_heads: usize = 1,
    head_dim: usize = 256,
    global_head_dim: usize = 512,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 10000.0,
    max_seq_len: usize = 4096,
};

pub const LayerWeights = struct {
    input_layernorm: []const bf16,
    q_proj: []const bf16,
    k_proj: []const bf16,
    v_proj: []const bf16,
    o_proj: []const bf16,
    q_norm: []const bf16,
    k_norm: []const bf16,
    pre_feedforward_layernorm: []const bf16,
    gate_proj: []const bf16,
    up_proj: []const bf16,
    down_proj: []const bf16,
    post_feedforward_layernorm: ?[]const bf16 = null,
    layer_scalar: ?f32 = null,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    config: ModelConfig,
    embed_tokens: []const bf16,
    layers: []LayerWeights,
    final_norm: []const bf16,

    pub fn loadFromSafeTensors(allocator: std.mem.Allocator, st: *const safetensors.SafeTensors, config: ModelConfig) !Model {
        const embed_tv = st.get("model.language_model.embed_tokens.weight") orelse return error.MissingEmbedTokens;
        const norm_tv = st.get("model.language_model.norm.weight") orelse return error.MissingFinalNorm;

        const embed_slice = embed_tv.asSlice(bf16);
        const norm_slice = norm_tv.asSlice(bf16);

        var layers = try allocator.alloc(LayerWeights, config.num_hidden_layers);

        var buf: [128]u8 = undefined;
        for (0..config.num_hidden_layers) |l| {
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

            layers[l] = LayerWeights{
                .input_layernorm = in_norm,
                .q_proj = q,
                .k_proj = k,
                .v_proj = v,
                .o_proj = o,
                .q_norm = qn,
                .k_norm = kn,
                .pre_feedforward_layernorm = pre_ffn,
                .gate_proj = gate,
                .up_proj = up,
                .down_proj = down,
                .post_feedforward_layernorm = post_ffn,
                .layer_scalar = layer_scalar,
            };
        }

        return Model{
            .allocator = allocator,
            .config = config,
            .embed_tokens = embed_slice,
            .layers = layers,
            .final_norm = norm_slice,
        };
    }

    pub fn deinit(self: *Model) void {
        self.allocator.free(self.layers);
    }
};

pub const KVCache = struct {
    allocator: std.mem.Allocator,
    k: []f32,
    v: []f32,
    max_seq_len: usize,
    num_layers: usize,
    kv_dim: usize,

    pub fn init(allocator: std.mem.Allocator, num_layers: usize, max_seq_len: usize, kv_dim: usize) !KVCache {
        const total_elements = num_layers * max_seq_len * kv_dim;
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
            .kv_dim = kv_dim,
        };
    }

    pub fn deinit(self: *KVCache) void {
        self.allocator.free(self.k);
        self.allocator.free(self.v);
    }

    pub fn getKV(self: *KVCache, layer: usize, pos: usize) struct { k: []f32, v: []f32 } {
        const offset = (layer * self.max_seq_len + pos) * self.kv_dim;
        return .{
            .k = self.k[offset .. offset + self.kv_dim],
            .v = self.v[offset .. offset + self.kv_dim],
        };
    }
};
