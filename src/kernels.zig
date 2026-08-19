const std = @import("std");
pub const tensor = @import("tensor.zig");
const bf16 = tensor.bf16;

/// In-place or out-of-place RMSNorm:
/// out = (x / sqrt(mean(x^2) + eps)) * weight
pub fn rmsNorm(
    out: []f32,
    in: []const f32,
    weight: []const bf16,
    eps: f32,
) void {
    std.debug.assert(in.len == out.len);
    std.debug.assert(in.len == weight.len);

    var sum_sq: f32 = 0.0;
    for (in) |v| {
        sum_sq += v * v;
    }
    const mean_sq = sum_sq / @as(f32, @floatFromInt(in.len));
    const rsqrt_val = 1.0 / @sqrt(mean_sq + eps);

    for (out, in, weight) |*o, i, w| {
        o.* = i * rsqrt_val * w.toF32();
    }
}

/// Sigmoid activation
pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// Matrix-Vector multiplication: y = W * x
/// W has shape [rows, cols] in row-major order (rows * cols total elements)
/// x has length [cols]
/// y has length [rows]
pub fn gemv(
    y: []f32,
    x: []const f32,
    w: []const bf16,
    rows: usize,
    cols: usize,
) void {
    std.debug.assert(y.len == rows);
    std.debug.assert(x.len == cols);
    std.debug.assert(w.len >= rows * cols);

    for (0..rows) |r| {
        const row_offset = r * cols;
        var dot: f32 = 0.0;
        const row_w = w[row_offset .. row_offset + cols];
        for (row_w, x) |w_val, x_val| {
            dot += w_val.toF32() * x_val;
        }
        y[r] = dot;
    }
}

/// Parallel GEMV dividing rows across threads
pub fn gemvParallel(
    y: []f32,
    x: []const f32,
    w: []const bf16,
    rows: usize,
    cols: usize,
    thread_pool: *std.Thread.Pool,
) void {
    const num_chunks = 16;
    const chunk_size = (rows + num_chunks - 1) / num_chunks;

    var wg = std.Thread.WaitGroup{};
    var r_start: usize = 0;
    while (r_start < rows) : (r_start += chunk_size) {
        const r_end = @min(r_start + chunk_size, rows);
        thread_pool.spawnWg(&wg, struct {
            fn run(y_slice: []f32, x_vec: []const f32, w_mat: []const bf16, start: usize, end: usize, c: usize) void {
                for (start..end) |r| {
                    const row_offset = r * c;
                    var dot: f32 = 0.0;
                    const row_w = w_mat[row_offset .. row_offset + c];
                    for (row_w, x_vec) |w_val, x_val| {
                        dot += w_val.toF32() * x_val;
                    }
                    y_slice[r] = dot;
                }
            }
        }.run, .{ y, x, w, r_start, r_end, cols });
    }
    thread_pool.waitAndWork(&wg);
}

/// GeLU with PyTorch tanh approximation:
/// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub inline fn geluTanh(x: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608;
    const coef: f32 = 0.044715;
    const inner = sqrt_2_over_pi * (x + coef * x * x * x);
    const tanh_val = std.math.tanh(inner);
    return 0.5 * x * (1.0 + tanh_val);
}

/// Gated MLP (SwiGLU / GeGLU style):
pub fn gatedMlp(
    out: []f32,
    in: []const f32,
    gate_w: []const bf16,
    up_w: []const bf16,
    down_w: []const bf16,
    hidden_dim: usize,
    intermediate_dim: usize,
    temp_act: []f32,
) void {
    std.debug.assert(temp_act.len >= intermediate_dim);

    for (0..intermediate_dim) |r| {
        const row_offset = r * hidden_dim;
        var gate_dot: f32 = 0.0;
        var up_dot: f32 = 0.0;
        for (0..hidden_dim) |c| {
            const in_v = in[c];
            gate_dot += gate_w[row_offset + c].toF32() * in_v;
            up_dot += up_w[row_offset + c].toF32() * in_v;
        }
        temp_act[r] = geluTanh(gate_dot) * up_dot;
    }

    gemv(out, temp_act[0..intermediate_dim], down_w, hidden_dim, intermediate_dim);
}

/// Rotary Position Embedding (RoPE) applied to 2D head planes with partial rotary support
pub fn applyRopePartial(
    vec: []f32,
    pos: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
) void {
    std.debug.assert(vec.len % head_dim == 0);
    const num_heads = vec.len / head_dim;

    for (0..num_heads) |h| {
        const head_offset = h * head_dim;
        const half_rot = rotary_dim / 2;

        for (0..half_rot) |i| {
            const freq = 1.0 / std.math.pow(f32, rope_theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(rotary_dim)));
            const angle = @as(f32, @floatFromInt(pos)) * freq;
            const cos_val = @cos(angle);
            const sin_val = @sin(angle);

            const idx0 = head_offset + i;
            const idx1 = head_offset + i + half_rot;

            const v0 = vec[idx0];
            const v1 = vec[idx1];

            vec[idx0] = v0 * cos_val - v1 * sin_val;
            vec[idx1] = v0 * sin_val + v1 * cos_val;
        }
    }
}

pub fn softmax(logits: []f32) void {
    var max_val: f32 = -std.math.inf(f32);
    for (logits) |v| {
        if (v > max_val) max_val = v;
    }

    var sum_exp: f32 = 0.0;
    for (logits) |*v| {
        const exp_v = @exp(v.* - max_val);
        v.* = exp_v;
        sum_exp += exp_v;
    }

    const inv_sum = 1.0 / sum_exp;
    for (logits) |*v| {
        v.* *= inv_sum;
    }
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
    const w = [_]bf16{
        bf16.fromF32(1.0), bf16.fromF32(2.0),
        bf16.fromF32(3.0), bf16.fromF32(4.0),
    };
    var y: [2]f32 = undefined;
    gemv(&y, &x, &w, 2, 2);
    try std.testing.expect(std.math.approxEqAbs(f32, y[0], 5.0, 0.01));
    try std.testing.expect(std.math.approxEqAbs(f32, y[1], 11.0, 0.01));
}
