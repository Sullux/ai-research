const std = @import("std");
const kernels = @import("kernels.zig");

pub const Sampler = struct {
    prng: std.Random.DefaultPrng,
    temp: f32 = 0.7,
    top_p: f32 = 0.95,
    rep_penalty: f32 = 1.05,
    history: [128]u32 = undefined,
    history_len: usize = 0,

    pub fn init(seed: u64, temp: f32, top_p: f32, rep_penalty: f32) Sampler {
        return .{
            .prng = std.Random.DefaultPrng.init(seed),
            .temp = temp,
            .top_p = top_p,
            .rep_penalty = rep_penalty,
        };
    }

    pub fn recordToken(self: *Sampler, tok: u32) void {
        if (self.history_len < self.history.len) {
            self.history[self.history_len] = tok;
            self.history_len += 1;
        } else {
            std.mem.copyForwards(u32, self.history[0 .. self.history.len - 1], self.history[1..self.history.len]);
            self.history[self.history.len - 1] = tok;
        }
    }

    pub fn resetHistory(self: *Sampler) void {
        self.history_len = 0;
    }

    pub fn sample(self: *Sampler, logits: []f32) u32 {
        if (self.rep_penalty != 1.0) {
            for (self.history[0..self.history_len]) |tok| {
                if (tok < logits.len) {
                    if (logits[tok] > 0.0) logits[tok] /= self.rep_penalty else logits[tok] *= self.rep_penalty;
                }
            }
        }

        for (logits) |*l| l.* = 30.0 * std.math.tanh(l.* / 30.0);

        if (self.temp <= 0.0) return kernels.sampleArgmax(logits);
        for (logits) |*l| l.* /= self.temp;
        kernels.softmax(logits);

        const IndexedLogit = struct { id: u32, prob: f32 };
        var top_buf: [128]IndexedLogit = undefined;
        var count: usize = 0;

        for (logits, 0..) |p, i| {
            if (p > 1e-4 and count < top_buf.len) {
                top_buf[count] = .{ .id = @intCast(i), .prob = p };
                count += 1;
            }
        }

        if (count == 0) return kernels.sampleArgmax(logits);

        const slice = top_buf[0..count];
        std.mem.sort(IndexedLogit, slice, {}, struct {
            fn cmp(_: void, a: IndexedLogit, b: IndexedLogit) bool { return a.prob > b.prob; }
        }.cmp);

        var cum_p: f32 = 0.0;
        var cutoff: usize = 0;
        for (slice, 0..) |cand, i| {
            cum_p += cand.prob;
            cutoff = i + 1;
            if (cum_p >= self.top_p) break;
        }

        const r = self.prng.random().float(f32) * cum_p;
        var acc: f32 = 0.0;
        for (slice[0..cutoff]) |cand| {
            acc += cand.prob;
            if (acc >= r) return cand.id;
        }
        return slice[0].id;
    }
};
