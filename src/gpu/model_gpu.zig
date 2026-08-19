const std = @import("std");
pub const types = @import("types.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const kernels = @import("kernels.zig");
pub const tensor = @import("../tensor.zig");
pub const model = @import("../model.zig");
pub const model_types = @import("../model/types.zig");

pub const GpuLayerWeights = struct {
    q_proj: buffer.GpuBuffer,
    k_proj: buffer.GpuBuffer,
    v_proj: buffer.GpuBuffer,
    o_proj: buffer.GpuBuffer,
    gate_proj: buffer.GpuBuffer,
    up_proj: buffer.GpuBuffer,
    down_proj: buffer.GpuBuffer,

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
    layers: []GpuLayerWeights,
    embed_tokens: buffer.GpuBuffer,
    buf_normed_x: buffer.GpuBuffer,
    buf_q: buffer.GpuBuffer,
    buf_k: buffer.GpuBuffer,
    buf_v: buffer.GpuBuffer,
    buf_gate: buffer.GpuBuffer,
    buf_up: buffer.GpuBuffer,
    buf_act: buffer.GpuBuffer,
    buf_mlp_out: buffer.GpuBuffer,
    buf_logits: buffer.GpuBuffer,

    fn createWeightBuffer(ctx: *const context.GpuContext, src: []const tensor.bf16) !buffer.GpuBuffer {
        const byte_size = @as(u64, @intCast(src.len * @sizeOf(tensor.bf16)));
        var buf = try buffer.GpuBuffer.init(ctx, byte_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        @memcpy(buf.asSlice(u8), std.mem.sliceAsBytes(src));
        return buf;
    }

    pub fn init(allocator: std.mem.Allocator, ctx: *const context.GpuContext, m: *const model.Model, config: model_types.ModelConfig) !GpuModelContext {
        const engine = try kernels.GpuEngine.init(ctx);
        const H = config.hidden_size;
        const I = config.intermediate_size;
        const V = config.vocab_size;
        const max_head = @max(config.head_dim, config.global_head_dim);
        const max_q = config.num_attention_heads * max_head;
        const max_kv = @max(config.num_key_value_heads, config.num_global_key_value_heads) * max_head;

        var gpu_layers = try allocator.alloc(GpuLayerWeights, m.layers.len);
        errdefer allocator.free(gpu_layers);

        for (m.layers, 0..) |l, i| {
            gpu_layers[i] = .{
                .q_proj = try createWeightBuffer(ctx, l.q_proj),
                .k_proj = try createWeightBuffer(ctx, l.k_proj),
                .v_proj = if (l.v_proj.len > 0) try createWeightBuffer(ctx, l.v_proj) else try createWeightBuffer(ctx, l.k_proj),
                .o_proj = try createWeightBuffer(ctx, l.o_proj),
                .gate_proj = try createWeightBuffer(ctx, l.gate_proj),
                .up_proj = try createWeightBuffer(ctx, l.up_proj),
                .down_proj = try createWeightBuffer(ctx, l.down_proj),
            };
        }

        const embed_tokens = try createWeightBuffer(ctx, m.embed_tokens);

        const buf_normed_x = try buffer.GpuBuffer.init(ctx, H * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_q = try buffer.GpuBuffer.init(ctx, max_q * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_k = try buffer.GpuBuffer.init(ctx, max_kv * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_v = try buffer.GpuBuffer.init(ctx, max_kv * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_gate = try buffer.GpuBuffer.init(ctx, I * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_up = try buffer.GpuBuffer.init(ctx, I * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_act = try buffer.GpuBuffer.init(ctx, I * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_mlp_out = try buffer.GpuBuffer.init(ctx, H * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_logits = try buffer.GpuBuffer.init(ctx, V * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .engine = engine,
            .layers = gpu_layers,
            .embed_tokens = embed_tokens,
            .buf_normed_x = buf_normed_x,
            .buf_q = buf_q,
            .buf_k = buf_k,
            .buf_v = buf_v,
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
        self.buf_v.deinit();
        self.buf_k.deinit();
        self.buf_q.deinit();
        self.buf_normed_x.deinit();
        self.embed_tokens.deinit();
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        self.engine.deinit();
    }
};
