const std = @import("std");
pub const types = @import("types.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const kernels = @import("kernels.zig");
pub const model_types = @import("../model/types.zig");

pub const GpuScratch = struct {
    engine: kernels.GpuEngine,
    buf_x: buffer.GpuBuffer,
    buf_gate: buffer.GpuBuffer,
    buf_up: buffer.GpuBuffer,
    buf_out: buffer.GpuBuffer,
    buf_w: buffer.GpuBuffer,

    pub fn init(ctx: *const context.GpuContext, config: model_types.ModelConfig) !GpuScratch {
        const engine = try kernels.GpuEngine.init(ctx);
        const H = config.hidden_size;
        const I = config.intermediate_size;
        const V = config.vocab_size;
        const max_dim = @max(@max(H, I), V);

        // Preallocate shared activation buffers
        const buf_x = try buffer.GpuBuffer.init(ctx, max_dim * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_gate = try buffer.GpuBuffer.init(ctx, max_dim * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_up = try buffer.GpuBuffer.init(ctx, max_dim * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const buf_out = try buffer.GpuBuffer.init(ctx, max_dim * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        // Staging buffer for weights slice
        const buf_w = try buffer.GpuBuffer.init(ctx, max_dim * H * @sizeOf(u16), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        return .{
            .engine = engine,
            .buf_x = buf_x,
            .buf_gate = buf_gate,
            .buf_up = buf_up,
            .buf_out = buf_out,
            .buf_w = buf_w,
        };
    }

    pub fn deinit(self: *GpuScratch) void {
        self.buf_w.deinit();
        self.buf_out.deinit();
        self.buf_up.deinit();
        self.buf_gate.deinit();
        self.buf_x.deinit();
        self.engine.deinit();
    }
};
