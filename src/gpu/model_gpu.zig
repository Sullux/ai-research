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
    q_proj: buffer.GpuBuffer,
    k_proj: buffer.GpuBuffer,
    v_proj: buffer.GpuBuffer,
    o_proj: buffer.GpuBuffer,
    gate_proj: buffer.GpuBuffer,
    up_proj: buffer.GpuBuffer,
    down_proj: buffer.GpuBuffer,
    desc: descriptors.LayerDescriptorSets,

    pub fn deinit(self: *GpuLayerWeights) void {
        self.down_proj.deinit();
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.o_proj.deinit();
        self.v_proj.deinit();
        self.k_proj.deinit();
        self.q_proj.deinit();
    }
};

pub const GpuModelContext = struct {
    allocator: std.mem.Allocator,
    ctx: *const context.GpuContext,
    engine: kernels.GpuEngine,
    desc_mgr: descriptors.DescriptorManager,
    layers: []GpuLayerWeights,
    embed_tokens: buffer.GpuBuffer,
    desc_logits: types.VkDescriptorSet,
    buf_normed_x: buffer.GpuBuffer,
    buf_q: buffer.GpuBuffer,
    buf_k: buffer.GpuBuffer,
    buf_v: buffer.GpuBuffer,
    buf_attn_out: buffer.GpuBuffer,
    buf_gate: buffer.GpuBuffer,
    buf_up: buffer.GpuBuffer,
    buf_act: buffer.GpuBuffer,
    buf_mlp_out: buffer.GpuBuffer,
    buf_logits: buffer.GpuBuffer,

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

    pub fn init(allocator: std.mem.Allocator, ctx: *const context.GpuContext, m: *const model.Model, config: model_types.ModelConfig, mode: quant.QuantMode) !GpuModelContext {
        var engine = try kernels.GpuEngine.init(ctx, mode);
        errdefer engine.deinit();

        const max_sets: u32 = @intCast(m.layers.len * 8 + 8);
        var desc_mgr = try descriptors.DescriptorManager.init(ctx, max_sets);
        errdefer desc_mgr.deinit();

        const H = config.hidden_size;
        const I = config.intermediate_size;
        const V = config.vocab_size;
        const max_head = @max(config.head_dim, config.global_head_dim);
        const max_q = config.num_attention_heads * max_head;
        const max_kv = @max(config.num_key_value_heads, config.num_global_key_value_heads) * max_head;

        const buf_normed_x = try buffer.GpuBuffer.init(ctx, H * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_q = try buffer.GpuBuffer.init(ctx, max_q * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_k = try buffer.GpuBuffer.init(ctx, max_kv * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_v = try buffer.GpuBuffer.init(ctx, max_kv * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_attn_out = try buffer.GpuBuffer.init(ctx, max_q * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
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
                .q_proj = try createWeightBuffer(ctx, l.q_proj, q_dim, H, mode),
                .k_proj = try createWeightBuffer(ctx, l.k_proj, kv_dim, H, mode),
                .v_proj = if (l.v_proj.len > 0) try createWeightBuffer(ctx, l.v_proj, kv_dim, H, mode) else try createWeightBuffer(ctx, l.k_proj, kv_dim, H, mode),
                .o_proj = try createWeightBuffer(ctx, l.o_proj, H, q_dim, mode),
                .gate_proj = try createWeightBuffer(ctx, l.gate_proj, I, H, mode),
                .up_proj = try createWeightBuffer(ctx, l.up_proj, I, H, mode),
                .down_proj = try createWeightBuffer(ctx, l.down_proj, H, I, mode),
                .desc = .{
                    .q_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .k_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .v_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .o_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .gate_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .up_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .swiglu = try desc_mgr.allocateSet(engine.swiglu_pipe.desc_set_layout),
                    .down_proj = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout),
                    .gate_up_swiglu = try desc_mgr.allocateSet(engine.gate_up_pipe.desc_set_layout),
                },
            };
            desc_mgr.bindBuffers(w.desc.q_proj, &.{ &w.q_proj, &buf_normed_x, &buf_q });
            desc_mgr.bindBuffers(w.desc.k_proj, &.{ &w.k_proj, &buf_normed_x, &buf_k });
            desc_mgr.bindBuffers(w.desc.v_proj, &.{ &w.v_proj, &buf_normed_x, &buf_v });
            desc_mgr.bindBuffers(w.desc.o_proj, &.{ &w.o_proj, &buf_attn_out, &buf_mlp_out });
            desc_mgr.bindBuffers(w.desc.gate_proj, &.{ &w.gate_proj, &buf_normed_x, &buf_gate });
            desc_mgr.bindBuffers(w.desc.up_proj, &.{ &w.up_proj, &buf_normed_x, &buf_up });
            desc_mgr.bindBuffers(w.desc.swiglu, &.{ &buf_gate, &buf_up, &buf_act });
            desc_mgr.bindBuffers(w.desc.down_proj, &.{ &w.down_proj, &buf_act, &buf_mlp_out });
            desc_mgr.bindBuffers(w.desc.gate_up_swiglu, &.{ &w.gate_proj, &w.up_proj, &buf_normed_x, &buf_act });
            gpu_layers[i] = w;
        }

        const embed_tokens = try createWeightBuffer(ctx, m.embed_tokens, V, H, mode);
        const desc_logits = try desc_mgr.allocateSet(engine.gemv_pipe.desc_set_layout);
        desc_mgr.bindBuffers(desc_logits, &.{ &embed_tokens, &buf_normed_x, &buf_logits });

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .engine = engine,
            .desc_mgr = desc_mgr,
            .layers = gpu_layers,
            .embed_tokens = embed_tokens,
            .desc_logits = desc_logits,
            .buf_normed_x = buf_normed_x,
            .buf_q = buf_q,
            .buf_k = buf_k,
            .buf_v = buf_v,
            .buf_attn_out = buf_attn_out,
            .buf_gate = buf_gate,
            .buf_up = buf_up,
            .buf_act = buf_act,
            .buf_mlp_out = buf_mlp_out,
            .buf_logits = buf_logits,
        };
    }

    pub fn deinit(self: *GpuModelContext) void {
        self.buf_logits.deinit();
        self.buf_mlp_out.deinit();
        self.buf_act.deinit();
        self.buf_up.deinit();
        self.buf_gate.deinit();
        self.buf_attn_out.deinit();
        self.buf_v.deinit();
        self.buf_k.deinit();
        self.buf_q.deinit();
        self.buf_normed_x.deinit();
        self.embed_tokens.deinit();
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        self.desc_mgr.deinit();
        self.engine.deinit();
    }
};
