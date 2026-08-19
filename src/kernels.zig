const std = @import("std");
pub const tensor = @import("tensor.zig");
const bf16 = tensor.bf16;

/// Fast 8-wide SIMD dot product between bf16 matrix row and f32 activation vector
pub inline fn dotBf16(w_row: []const bf16, x: []const f32) f32 {
    const V = 8;
    const VecF32 = @Vector(V, f32);
    var acc: VecF32 = @splat(0.0);
    var i: usize = 0;
    const len = @min(w_row.len, x.len);
    while (i + V <= len) : (i += V) {
        var wf: [V]f32 = undefined;
        inline for (0..V) |k| wf[k] = w_row[i + k].toF32();
        const wv: VecF32 = wf;
        const xv: VecF32 = x[i..][0..V].*;
        acc += wv * xv;
    }
    var sum = @reduce(.Add, acc);
    while (i < len) : (i += 1) {
        sum += w_row[i].toF32() * x[i];
    }
    return sum;
}

/// Out-of-place or in-place RMSNorm with learnable weight
pub fn rmsNorm(out: []f32, in: []const f32, weight: []const bf16, eps: f32) void {
    std.debug.assert(in.len == out.len and in.len == weight.len);
    var sum_sq: f32 = 0.0;
    for (in) |v| sum_sq += v * v;
    const rsqrt_val = 1.0 / @sqrt((sum_sq / @as(f32, @floatFromInt(in.len))) + eps);
    for (out, in, weight) |*o, i, w| o.* = i * rsqrt_val * w.toF32();
}

/// Unit RMSNorm without learnable weight (used for v_norm in Gemma 4)
pub fn unitRmsNorm(out: []f32, in: []const f32, eps: f32) void {
    std.debug.assert(in.len == out.len);
    var sum_sq: f32 = 0.0;
    for (in) |v| sum_sq += v * v;
    const rsqrt_val = 1.0 / @sqrt((sum_sq / @as(f32, @floatFromInt(in.len))) + eps);
    for (out, in) |*o, i| o.* = i * rsqrt_val;
}

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// Matrix-Vector multiplication: y = W * x (row-major)
pub fn gemv(y: []f32, x: []const f32, w: []const bf16, rows: usize, cols: usize) void {
    std.debug.assert(y.len == rows and x.len == cols and w.len >= rows * cols);
    for (0..rows) |r| {
        y[r] = dotBf16(w[r * cols .. (r + 1) * cols], x);
    }
}

/// Parallel GEMV dividing rows across thread pool workers
pub fn gemvParallel(y: []f32, x: []const f32, w: []const bf16, rows: usize, cols: usize, tp: *std.Thread.Pool) void {
    const num_jobs: usize = 16;
    const chunk_size = (rows + num_jobs - 1) / num_jobs;
    var wg = std.Thread.WaitGroup{};
    var r_start: usize = 0;
    while (r_start < rows) : (r_start += chunk_size) {
        const r_end = @min(r_start + chunk_size, rows);
        tp.spawnWg(&wg, struct {
            fn run(y_s: []f32, x_v: []const f32, w_m: []const bf16, s: usize, e: usize, c: usize) void {
                for (s..e) |r| {
                    y_s[r] = dotBf16(w_m[r * c .. (r + 1) * c], x_v);
                }
            }
        }.run, .{ y, x, w, r_start, r_end, cols });
    }
    tp.waitAndWork(&wg);
}

pub inline fn geluTanh(x: f32) f32 {
    const inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

/// Parallel Gated MLP (SwiGLU / GeGLU style)
pub fn gatedMlp(out: []f32, in: []const f32, gw: []const bf16, uw: []const bf16, dw: []const bf16, h: usize, inter: usize, act: []f32, tp: ?*std.Thread.Pool) void {
    std.debug.assert(act.len >= inter);
    if (tp) |pool| {
        const num_jobs: usize = 16;
        const chunk_size = (inter + num_jobs - 1) / num_jobs;
        var wg = std.Thread.WaitGroup{};
        var r_start: usize = 0;
        while (r_start < inter) : (r_start += chunk_size) {
            const r_end = @min(r_start + chunk_size, inter);
            pool.spawnWg(&wg, struct {
                fn run(a: []f32, in_v: []const f32, g_w: []const bf16, u_w: []const bf16, s: usize, e: usize, c: usize) void {
                    for (s..e) |r| {
                        const g_dot = dotBf16(g_w[r * c .. (r + 1) * c], in_v);
                        const u_dot = dotBf16(u_w[r * c .. (r + 1) * c], in_v);
                        a[r] = geluTanh(g_dot) * u_dot;
                    }
                }
            }.run, .{ act, in, gw, uw, r_start, r_end, h });
        }
        pool.waitAndWork(&wg);
        gemvParallel(out, act[0..inter], dw, h, inter, pool);
    } else {
        for (0..inter) |r| {
            const g_dot = dotBf16(gw[r * h .. (r + 1) * h], in);
            const u_dot = dotBf16(uw[r * h .. (r + 1) * h], in);
            act[r] = geluTanh(g_dot) * u_dot;
        }
        gemv(out, act[0..inter], dw, h, inter);
    }
}

/// Partial RoPE on 2D head planes
pub fn applyRopePartial(vec: []f32, pos: usize, head_dim: usize, rotary_dim: usize, theta: f32) void {
    const num_heads = vec.len / head_dim;
    const half_rot = rotary_dim / 2;
    for (0..num_heads) |h| {
        const offset = h * head_dim;
        for (0..half_rot) |i| {
            const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
            const angle = @as(f32, @floatFromInt(pos)) * freq;
            const cos_v = @cos(angle);
            const sin_v = @sin(angle);
            const idx0 = offset + i;
            const idx1 = offset + i + half_rot;
            const v0 = vec[idx0];
            const v1 = vec[idx1];
            vec[idx0] = v0 * cos_v - v1 * sin_v;
            vec[idx1] = v0 * sin_v + v1 * cos_v;
        }
    }
}

pub fn softmax(logits: []f32) void {
    var max_val: f32 = -std.math.inf(f32);
    for (logits) |v| max_val = @max(max_val, v);
    var sum_exp: f32 = 0.0;
    for (logits) |*v| {
        v.* = @exp(v.* - max_val);
        sum_exp += v.*;
    }
    const inv = 1.0 / sum_exp;
    for (logits) |*v| v.* *= inv;
}

pub fn sampleArgmax(logits: []const f32) u32 {
    var max_idx: u32 = 0;
    var max_val: f32 = logits[0];
    for (logits[1..], 1..) |val, i| {
        if (val > max_val) {
            max_val = val;
            max_idx = @intCast(i);
        }
    }
    return max_idx;
}

test "gemv kernel calculation" {
    const x = [_]f32{ 1.0, 2.0 };
    const w = [_]bf16{ bf16.fromF32(1.0), bf16.fromF32(2.0), bf16.fromF32(3.0), bf16.fromF32(4.0) };
    var y: [2]f32 = undefined;
    gemv(&y, &x, &w, 2, 2);
    try std.testing.expect(std.math.approxEqAbs(f32, y[0], 5.0, 0.01));
    try std.testing.expect(std.math.approxEqAbs(f32, y[1], 11.0, 0.01));
}
