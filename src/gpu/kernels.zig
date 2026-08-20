const std = @import("std");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const pipeline = @import("pipeline.zig");
pub const shaders = @import("shaders.zig");
pub const quant = @import("../quant.zig");

pub const GpuEngine = struct {
    ctx: *const context.GpuContext,
    gemv_pipe: pipeline.ComputePipeline,
    swiglu_pipe: pipeline.ComputePipeline,

    pub fn init(ctx: *const context.GpuContext, mode: quant.QuantMode) !GpuEngine {
        const gemv_spirv = switch (mode) {
            .none => &shaders.GEMV_BF16_SPIRV,
            .q8 => &shaders.GEMV_Q8_SPIRV,
            .q4 => &shaders.GEMV_Q4_SPIRV,
        };
        var gemv = try pipeline.ComputePipeline.init(ctx, gemv_spirv, 3, 8);
        errdefer gemv.deinit();

        var swiglu = try pipeline.ComputePipeline.init(ctx, &shaders.FUSED_SWIGLU_SPIRV, 3, 4);
        errdefer swiglu.deinit();

        return .{
            .ctx = ctx,
            .gemv_pipe = gemv,
            .swiglu_pipe = swiglu,
        };
    }

    pub fn deinit(self: *GpuEngine) void {
        self.swiglu_pipe.deinit();
        self.gemv_pipe.deinit();
    }

    pub fn dispatchGemv(
        self: *const GpuEngine,
        w: *const buffer.GpuBuffer,
        x: *const buffer.GpuBuffer,
        y: *const buffer.GpuBuffer,
        m: usize,
        k: usize,
    ) !void {
        const bufs = [_]*const buffer.GpuBuffer{ w, x, y };
        try self.gemv_pipe.bindBuffers(&bufs);
        const pc = [_]u32{ @intCast(m), @intCast(k) };
        const workgroups: u32 = @intCast((m + 63) / 64);
        try self.gemv_pipe.dispatch(std.mem.sliceAsBytes(&pc), workgroups, 1, 1);
    }

    pub fn dispatchSwiGlu(
        self: *const GpuEngine,
        gate: *const buffer.GpuBuffer,
        up: *const buffer.GpuBuffer,
        out: *const buffer.GpuBuffer,
        dim: usize,
    ) !void {
        const bufs = [_]*const buffer.GpuBuffer{ gate, up, out };
        try self.swiglu_pipe.bindBuffers(&bufs);
        const pc = [_]u32{@intCast(dim)};
        const workgroups: u32 = @intCast((dim + 63) / 64);
        try self.swiglu_pipe.dispatch(std.mem.sliceAsBytes(&pc), workgroups, 1, 1);
    }
};
