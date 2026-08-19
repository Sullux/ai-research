const std = @import("std");
pub const tensor = @import("../tensor.zig");
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
    intermediate_dim: usize,

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
    intermediate_size: usize = 12288,
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
    ple_context: []f32,
    ple_buf_1: []f32,
    ple_buf_2: []f32,
    logits: []f32,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig) !ForwardScratch {
        const max_head_dim = @max(config.head_dim, config.global_head_dim);
        const max_q_dim = config.num_attention_heads * max_head_dim;
        const max_kv_dim = config.num_key_value_heads * max_head_dim;
        const total_ple_dim = config.num_hidden_layers * config.hidden_size_per_layer_input;

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
            .ple_context = try allocator.alloc(f32, total_ple_dim),
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
        allocator.free(self.ple_context);
        allocator.free(self.ple_buf_1);
        allocator.free(self.ple_buf_2);
        allocator.free(self.logits);
    }
};
