const std = @import("std");
pub const types = @import("types.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const kernels = @import("kernels.zig");
pub const descriptors = @import("descriptors.zig");
pub const tensor = @import("../tensor.zig");
pub const model = @import("../model.zig");
pub const model_types = @import("../model/types.zig");
pub const quant = @import("../quant.zig");

pub const GpuLayerWeights = struct {
    input_norm: buffer.GpuBuffer,
    q_norm: buffer.GpuBuffer,
    k_norm: buffer.GpuBuffer,
    q_proj: buffer.GpuBuffer,
    k_proj: buffer.GpuBuffer,
    v_proj: buffer.GpuBuffer,
    o_proj: buffer.GpuBuffer,
    gate_proj: buffer.GpuBuffer, up_proj: buffer.GpuBuffer, down_proj: buffer.GpuBuffer,
    pre_ffn_norm: buffer.GpuBuffer, post_attn_norm: buffer.GpuBuffer, post_ffn_norm: buffer.GpuBuffer,
    buf_k_cache: buffer.GpuBuffer, buf_v_cache: buffer.GpuBuffer, layer_scalar: f32,
    has_post_attn_norm: bool, has_post_ffn_norm: bool, desc: descriptors.LayerDescriptorSets,

    pub fn deinit(self: *GpuLayerWeights) void {
        self.buf_v_cache.deinit(); self.buf_k_cache.deinit();
        self.post_ffn_norm.deinit(); self.post_attn_norm.deinit(); self.pre_ffn_norm.deinit();
        self.down_proj.deinit(); self.up_proj.deinit(); self.gate_proj.deinit();
        self.o_proj.deinit(); self.v_proj.deinit(); self.k_proj.deinit();
        self.q_proj.deinit(); self.k_norm.deinit(); self.q_norm.deinit();
        self.input_norm.deinit();
    }
};

pub const GpuModelContext = struct {
    allocator: std.mem.Allocator,
    ctx: *const context.GpuContext,
    engine: kernels.GpuEngine,
    desc_mgr: descriptors.DescriptorManager,
    layers: []GpuLayerWeights,
    embed_tokens: buffer.GpuBuffer, final_norm: buffer.GpuBuffer,
    desc_logits: types.VkDescriptorSet, desc_final_norm: types.VkDescriptorSet,
    buf_x: buffer.GpuBuffer, buf_normed_x: buffer.GpuBuffer,
    buf_q: buffer.GpuBuffer, buf_k: buffer.GpuBuffer, buf_v: buffer.GpuBuffer,
    buf_attn_out: buffer.GpuBuffer, buf_active_slots: buffer.GpuBuffer,
    buf_gate: buffer.GpuBuffer, buf_up: buffer.GpuBuffer, buf_act: buffer.GpuBuffer,
    buf_mlp_out: buffer.GpuBuffer, buf_logits: buffer.GpuBuffer,

    fn createWeightBuffer(ctx: *const context.GpuContext, src: []const tensor.bf16, rows: usize, cols: usize, mode: quant.QuantMode) !buffer.GpuBuffer {
        if (mode == .none or src.len == 0) {
            const byte_size = @as(u64, @intCast(src.len * @sizeOf(tensor.bf16)));
            var buf = try buffer.GpuBuffer.init(ctx, byte_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            if (src.len > 0) @memcpy(buf.asSlice(u8), std.mem.sliceAsBytes(src));
            return buf;
        }
        const byte_size = @as(u64, @intCast(quant.getQuantizedSizeBytes(rows, cols, mode)));
        var buf = try buffer.GpuBuffer.init(ctx, byte_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        quant.quantizeMatrix(buf.asSlice(u32), @as([]const u16, @ptrCast(src)), rows, cols, mode);
        return buf;
    }

    fn createNormBuffer(ctx: *const context.GpuContext, src: []const tensor.bf16, H: usize) !buffer.GpuBuffer {
        var buf = try buffer.GpuBuffer.init(ctx, H * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const slice = buf.asSlice(f32);
        if (src.len == H) {
            for (slice, src) |*dst, s| dst.* = s.toF32();
        } else {
            @memset(slice, 1.0);
        }
        return buf;
    }

    pub fn init(allocator: std.mem.Allocator, ctx: *const context.GpuContext, m: *const model.Model, config: model_types.ModelConfig, mode: quant.QuantMode) !GpuModelContext {
        var engine = try kernels.GpuEngine.init(ctx, mode);
        errdefer engine.deinit();

        const max_sets: u32 = @intCast(m.layers.len * 16 + 16);
        var desc_mgr = try descriptors.DescriptorManager.init(ctx, max_sets);
        errdefer desc_mgr.deinit();

        const H = config.hidden_size;
        const I = config.intermediate_size;
        const V = config.vocab_size;
        const max_slots = 1024;
        const max_head = @max(config.head_dim, config.global_head_dim);
        const max_q = config.num_attention_heads * max_head;
        const max_kv = @max(config.num_key_value_heads, config.num_global_key_value_heads) * max_head;

        const buf_x = try buffer.GpuBuffer.init(ctx, H * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_normed_x = try buffer.GpuBuffer.init(ctx, H * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_q = try buffer.GpuBuffer.init(ctx, max_q * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_k = try buffer.GpuBuffer.init(ctx, max_kv * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_v = try buffer.GpuBuffer.init(ctx, max_kv * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_attn_out = try buffer.GpuBuffer.init(ctx, max_q * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_active_slots = try buffer.GpuBuffer.init(ctx, max_slots * @sizeOf(u32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_gate = try buffer.GpuBuffer.init(ctx, I * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_up = try buffer.GpuBuffer.init(ctx, I * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_act = try buffer.GpuBuffer.init(ctx, I * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_mlp_out = try buffer.GpuBuffer.init(ctx, H * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_logits = try buffer.GpuBuffer.init(ctx, V * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        var gpu_layers = try allocator.alloc(GpuLayerWeights, m.layers.len);
        for (m.layers, 0..) |l, i| {
            const q_dim = l.q_proj.len / H;
            const kv_dim = l.k_proj.len / H;
            var w = GpuLayerWeights{
                .input_norm = try createNormBuffer(ctx, l.input_layernorm, H),
                .q_norm = try createNormBuffer(ctx, l.q_norm, l.head_dim),
                .k_norm = try createNormBuffer(ctx, l.k_norm, l.head_dim),
                .q_proj = try createWeightBuffer(ctx, l.q_proj, q_dim, H, mode),
                .k_proj = try createWeightBuffer(ctx, l.k_proj, kv_dim, H, mode),
                .v_proj = if (l.v_proj.len > 0) try createWeightBuffer(ctx, l.v_proj, kv_dim, H, mode) else try createWeightBuffer(ctx, l.k_proj, kv_dim, H, mode),
                .o_proj = try createWeightBuffer(ctx, l.o_proj, H, q_dim, mode),
                .gate_proj = try createWeightBuffer(ctx, l.gate_proj, I, H, mode),
                .up_proj = try createWeightBuffer(ctx, l.up_proj, I, H, mode),
                .down_proj = try createWeightBuffer(ctx, l.down_proj, H, I, mode),
                .pre_ffn_norm = try createNormBuffer(ctx, l.pre_feedforward_layernorm, H),
                .post_attn_norm = try createNormBuffer(ctx, if (l.post_attention_layernorm) |p| p else &.{}, H),
                .post_ffn_norm = try createNormBuffer(ctx, if (l.post_feedforward_layernorm) |p| p else &.{}, H),
                .buf_k_cache = try buffer.GpuBuffer.init(ctx, max_slots * max_kv * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                .buf_v_cache = try buffer.GpuBuffer.init(ctx, max_slots * max_kv * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                .layer_scalar = l.layer_scalar orelse 1.0,
                .has_post_attn_norm = (l.post_attention_layernorm != null),
                .has_post_ffn_norm = (l.post_feedforward_layernorm != null),
                .desc = .{
                    .input_norm = try desc_mgr.allocateSet(engine.rmsnorm_pipe.desc_set_layout),
                    .q_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .k_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .v_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .qkv_rope = try desc_mgr.allocateSet(engine.qkv_rope_pipe.desc_set_layout),
                    .attn = try desc_mgr.allocateSet(engine.attn_pipe.desc_set_layout),
                    .o_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .gate_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .up_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .swiglu = try desc_mgr.allocateSet(engine.swiglu_pipe.desc_set_layout),
                    .down_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .gate_up_swiglu = try desc_mgr.allocateSet(engine.gate_up_pipe.desc_set_layout),
                    .pre_ffn_norm = try desc_mgr.allocateSet(engine.add_rmsnorm_pipe.desc_set_layout),
                    .post_attn_norm = try desc_mgr.allocateSet(engine.rmsnorm_pipe.desc_set_layout),
                    .post_ffn_norm = try desc_mgr.allocateSet(engine.rmsnorm_pipe.desc_set_layout),
                    .post_ffn_add = try desc_mgr.allocateSet(engine.add_rmsnorm_pipe.desc_set_layout),
                },
            };
            desc_mgr.bindBuffers(w.desc.input_norm, &.{ &buf_x, &w.input_norm, &buf_normed_x });
            desc_mgr.bindBuffers(w.desc.q_proj, &.{ &w.q_proj, &buf_normed_x, &buf_q });
            desc_mgr.bindBuffers(w.desc.k_proj, &.{ &w.k_proj, &buf_normed_x, &buf_k });
            desc_mgr.bindBuffers(w.desc.v_proj, &.{ &w.v_proj, &buf_normed_x, &buf_v });
            desc_mgr.bindBuffers(w.desc.qkv_rope, &.{ &buf_q, &buf_k, &buf_v, &w.q_norm, &w.k_norm, &buf_q, &w.buf_k_cache, &w.buf_v_cache });
            desc_mgr.bindBuffers(w.desc.attn, &.{ &buf_q, &w.buf_k_cache, &w.buf_v_cache, &buf_active_slots, &buf_attn_out });
            desc_mgr.bindBuffers(w.desc.o_proj, &.{ &w.o_proj, &buf_attn_out, &buf_mlp_out });
            desc_mgr.bindBuffers(w.desc.gate_proj, &.{ &w.gate_proj, &buf_normed_x, &buf_gate });
            desc_mgr.bindBuffers(w.desc.up_proj, &.{ &w.up_proj, &buf_normed_x, &buf_up });
            desc_mgr.bindBuffers(w.desc.swiglu, &.{ &buf_gate, &buf_up, &buf_act });
            desc_mgr.bindBuffers(w.desc.down_proj, &.{ &w.down_proj, &buf_act, &buf_mlp_out });
            desc_mgr.bindBuffers(w.desc.gate_up_swiglu, &.{ &w.gate_proj, &w.up_proj, &buf_normed_x, &buf_act });
            desc_mgr.bindBuffers(w.desc.pre_ffn_norm, &.{ &buf_x, &buf_mlp_out, &w.pre_ffn_norm, &buf_normed_x });
            desc_mgr.bindBuffers(w.desc.post_attn_norm, &.{ &buf_mlp_out, &w.post_attn_norm, &buf_mlp_out });
            desc_mgr.bindBuffers(w.desc.post_ffn_norm, &.{ &buf_mlp_out, &w.post_ffn_norm, &buf_mlp_out });
            desc_mgr.bindBuffers(w.desc.post_ffn_add, &.{ &buf_x, &buf_mlp_out, &w.input_norm, &buf_normed_x });
            gpu_layers[i] = w;
        }

        const embed_mode: quant.QuantMode = if (mode == .q4) .q8 else mode;
        const embed_tokens = try createWeightBuffer(ctx, m.embed_tokens, V, H, embed_mode);
        const final_norm = try createNormBuffer(ctx, m.final_norm, H);
        const desc_logits = try desc_mgr.allocateSet(engine.gemv_logits_pipe.desc_set_layout);
        const desc_final_norm = try desc_mgr.allocateSet(engine.rmsnorm_pipe.desc_set_layout);
        desc_mgr.bindBuffers(desc_final_norm, &.{ &buf_x, &final_norm, &buf_normed_x });
        desc_mgr.bindBuffers(desc_logits, &.{ &embed_tokens, &buf_normed_x, &buf_logits });

        return .{
            .allocator = allocator, .ctx = ctx, .engine = engine, .desc_mgr = desc_mgr,
            .layers = gpu_layers, .embed_tokens = embed_tokens, .final_norm = final_norm,
            .desc_logits = desc_logits, .desc_final_norm = desc_final_norm,
            .buf_x = buf_x, .buf_normed_x = buf_normed_x, .buf_q = buf_q, .buf_k = buf_k,
            .buf_v = buf_v, .buf_attn_out = buf_attn_out, .buf_active_slots = buf_active_slots,
            .buf_gate = buf_gate, .buf_up = buf_up, .buf_act = buf_act, .buf_mlp_out = buf_mlp_out,
            .buf_logits = buf_logits,
        };
    }

    pub fn deinit(self: *GpuModelContext) void {
        self.engine.deinit();
        self.desc_mgr.deinit();
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        self.final_norm.deinit();
        self.embed_tokens.deinit();
        self.buf_logits.deinit(); self.buf_mlp_out.deinit(); self.buf_act.deinit();
        self.buf_up.deinit(); self.buf_gate.deinit(); self.buf_active_slots.deinit();
        self.buf_attn_out.deinit(); self.buf_v.deinit(); self.buf_k.deinit(); self.buf_q.deinit();
        self.buf_normed_x.deinit(); self.buf_x.deinit();
    }
};
