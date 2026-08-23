const std = @import("std");
const kernels = @import("kernels.zig");

// Min-heap for keeping track of the top K elements
const IndexedLogit = struct {
    id: u32,
    prob: f32,
};

fn pushTopK(heap: []IndexedLogit, size: *usize, max_k: usize, item: IndexedLogit) void {
    if (size.* < max_k) {
        heap[size.*] = item;
        size.* += 1;
        // Bubble up min-heap
        var idx = size.* - 1;
        while (idx > 0) {
            const parent = (idx - 1) / 2;
            if (heap[idx].prob < heap[parent].prob) {
                const tmp = heap[idx];
                heap[idx] = heap[parent];
                heap[parent] = tmp;
                idx = parent;
            } else break;
        }
    } else if (item.prob > heap[0].prob) {
        heap[0] = item;
        // Sift down
        var idx: usize = 0;
        while (true) {
            const left = 2 * idx + 1;
            const right = 2 * idx + 2;
            var smallest = idx;
            if (left < max_k and heap[left].prob < heap[smallest].prob) smallest = left;
            if (right < max_k and heap[right].prob < heap[smallest].prob) smallest = right;
            if (smallest != idx) {
                const tmp = heap[idx];
                heap[idx] = heap[smallest];
                heap[smallest] = tmp;
                idx = smallest;
            } else break;
        }
    }
}

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
        const suppress = [_]u32{ 0, 258882, 258883, 255999, 256000, 256001, 255995, 255996, 255997, 255998 };
        for (suppress) |sup| {
            if (sup < logits.len) logits[sup] = -1e9;
        }

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

        // Find true top-64 probabilities using min-heap
        var top_heap: [64]IndexedLogit = undefined;
        var heap_size: usize = 0;

        for (logits, 0..) |p, i| {
            if (p > 1e-6) {
                pushTopK(&top_heap, &heap_size, 64, .{ .id = @intCast(i), .prob = p });
            }
        }

        if (heap_size == 0) return kernels.sampleArgmax(logits);

        const candidates = top_heap[0..heap_size];
        std.mem.sort(IndexedLogit, candidates, {}, struct {
            fn cmp(_: void, a: IndexedLogit, b: IndexedLogit) bool { return a.prob > b.prob; }
        }.cmp);

        var cum_p: f32 = 0.0;
        var cutoff: usize = 0;
        for (candidates, 0..) |cand, i| {
            cum_p += cand.prob;
            cutoff = i + 1;
            if (cum_p >= self.top_p) break;
        }

        const r = self.prng.random().float(f32) * cum_p;
        var acc: f32 = 0.0;
        for (candidates[0..cutoff]) |cand| {
            acc += cand.prob;
            if (acc >= r) return cand.id;
        }
        return candidates[0].id;
    }
};
