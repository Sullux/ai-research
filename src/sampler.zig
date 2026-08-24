const std = @import("std");
const kernels = @import("kernels.zig");

const IndexedLogit = struct {
    id: u32,
    val: f32,
};

fn pushTopK(heap: []IndexedLogit, size: *usize, max_k: usize, item: IndexedLogit) void {
    if (size.* < max_k) {
        heap[size.*] = item;
        size.* += 1;
        var idx = size.* - 1;
        while (idx > 0) {
            const parent = (idx - 1) / 2;
            if (heap[idx].val < heap[parent].val) {
                const tmp = heap[idx];
                heap[idx] = heap[parent];
                heap[parent] = tmp;
                idx = parent;
            } else break;
        }
    } else if (item.val > heap[0].val) {
        heap[0] = item;
        var idx: usize = 0;
        while (true) {
            const left = 2 * idx + 1;
            const right = 2 * idx + 2;
            var smallest = idx;
            if (left < max_k and heap[left].val < heap[smallest].val) smallest = left;
            if (right < max_k and heap[right].val < heap[smallest].val) smallest = right;
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

    pub fn init(seed: u64, temp: f32, top_p: f32) Sampler {
        return .{
            .prng = std.Random.DefaultPrng.init(seed),
            .temp = temp,
            .top_p = top_p,
        };
    }

    pub fn sample(self: *Sampler, logits: []f32) u32 {
        const suppress = [_]u32{ 0, 258882, 258883, 255999, 256000, 256001, 255995, 255996, 255997, 255998 };
        for (suppress) |sup| {
            if (sup < logits.len) logits[sup] = -1e9;
        }

        if (self.temp <= 0.0) return kernels.sampleArgmax(logits);

        var top_heap: [64]IndexedLogit = undefined;
        var heap_size: usize = 0;

        for (logits, 0..) |v, i| {
            if (v > -1e8) {
                pushTopK(&top_heap, &heap_size, 64, .{ .id = @intCast(i), .val = v });
            }
        }
        if (heap_size == 0) return kernels.sampleArgmax(logits);

        const candidates = top_heap[0..heap_size];
        std.mem.sort(IndexedLogit, candidates, {}, struct {
            fn cmp(_: void, a: IndexedLogit, b: IndexedLogit) bool { return a.val > b.val; }
        }.cmp);

        // Apply 30.0 tanh soft-capping and temperature scaling to top-64
        var probs: [64]f32 = undefined;
        var max_scaled: f32 = -1e9;
        for (candidates, 0..) |cand, i| {
            const capped = 30.0 * std.math.tanh(cand.val / 30.0);
            const scaled = capped / self.temp;
            probs[i] = scaled;
            if (scaled > max_scaled) max_scaled = scaled;
        }

        // Softmax over top-64
        var sum: f32 = 0.0;
        for (0..heap_size) |i| {
            probs[i] = @exp(probs[i] - max_scaled);
            sum += probs[i];
        }
        for (0..heap_size) |i| probs[i] /= sum;

        // Top-P cumulative selection
        var cum_p: f32 = 0.0;
        var cutoff: usize = 0;
        for (0..heap_size) |i| {
            cum_p += probs[i];
            cutoff = i + 1;
            if (cum_p >= self.top_p) break;
        }

        const r = self.prng.random().float(f32) * cum_p;
        var acc: f32 = 0.0;
        for (0..cutoff) |i| {
            acc += probs[i];
            if (acc >= r) return candidates[i].id;
        }
        return candidates[0].id;
    }
};
