const std = @import("std");
const model = @import("../model.zig");
const model_types = @import("types.zig");
const gpu = @import("../gpu.zig");

pub fn runGpuBenchmark(m: *const model.Model, config: model_types.ModelConfig, g: *gpu.model_gpu.GpuModelContext, stdout: anytype) !void {
    try stdout.print("\n=== Single-Submission GPU Pipeline Benchmark ({d} layers) ===\n", .{m.layers.len});
    const iter: usize = 20;

    // Warmup
    try recordAndSubmit(m, config, g);

    const t0 = std.time.nanoTimestamp();
    for (0..iter) |_| try recordAndSubmit(m, config, g);
    const t1 = std.time.nanoTimestamp();

    const ms = (@as(f64, @floatFromInt(t1 - t0)) / @as(f64, @floatFromInt(iter))) / 1_000_000.0;
    const tps = 1000.0 / ms;
    try stdout.print("GPU Single-Submission Execution: {d:.2} ms / token\n", .{ms});
    try stdout.print("Measured GPU Throughput:        {d:.1} TOKENS / SEC\n\n", .{tps});
}

fn recordAndSubmit(m: *const model.Model, config: model_types.ModelConfig, g: *gpu.model_gpu.GpuModelContext) !void {
    g.engine.beginBatch();
    for (0..m.layers.len) |l_i| {
        const l = &g.layers[l_i];
        g.engine.recordAddRmsNorm(l.desc.input_norm, config.hidden_size, config.rms_norm_eps);
        g.engine.recordBarrier(&g.buf_normed_x);
        g.engine.recordGemv(l.desc.q_proj, config.hidden_size, config.hidden_size);
        g.engine.recordGemv(l.desc.k_proj, 2048, config.hidden_size);
        g.engine.recordGemv(l.desc.v_proj, 2048, config.hidden_size);
        g.engine.recordBarrier(&g.buf_q);
        g.engine.recordGemv(l.desc.o_proj, config.hidden_size, config.hidden_size);
        g.engine.recordBarrier(&g.buf_mlp_out);
        g.engine.recordAddRmsNorm(l.desc.pre_ffn_norm, config.hidden_size, config.rms_norm_eps);
        g.engine.recordBarrier(&g.buf_normed_x);
        if (g.engine.mode == .q4) {
            g.engine.recordGateUpSwiGlu(l.desc.gate_up_swiglu, config.intermediate_size, config.hidden_size);
            g.engine.recordBarrier(&g.buf_act);
            g.engine.recordGemv(l.desc.down_proj, config.hidden_size, config.intermediate_size);
        }
        g.engine.recordBarrier(&g.buf_mlp_out);
    }
    try g.engine.submitBatch();
}
