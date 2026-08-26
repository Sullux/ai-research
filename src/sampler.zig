const std = @import("std");
const kernels = @import("kernels.zig");

pub const IndexedLogit = struct {
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
        return;
    }
    if (item.val <= heap[0].val) return;
    heap[0] = item;
    var idx: usize = 0;
    while (true) {
        const left = 2 * idx + 1;
        const right = 2 * idx + 2;
        var smallest = idx;
        if (left < max_k and heap[left].val < heap[smallest].val) smallest = left;
        if (right < max_k and heap[right].val < heap[smallest].val) smallest = right;
        if (smallest == idx) break;
        const tmp = heap[idx];
        heap[idx] = heap[smallest];
        heap[smallest] = tmp;
        idx = smallest;
    }
}

inline fn isExcludedFromRepeatPenalty(tok: u32) bool {
    if (tok < 256) return true;
    return switch (tok) {
        // Punctuation & Quotes
        623, 236743, 236761, 236764, 236768, 236787, 236788, 236789, 236799, 236881 => true,
        // Common Contractions ('m, 's, 't, 'd, 're, 've, 'll)
        500, 560, 859, 236745, 236751, 236753, 236757 => true,
        else => false,
    };
}

pub const Sampler = struct {
    prng: std.Random.DefaultPrng,
    top_k: u32 = 64,
    repeat_penalty: f32 = 1.1,
    top_p: f32 = 0.95,
    min_p: f32 = 0.05,
    temp: f32 = 1.0,

    pub fn init(seed: u64, temp: f32, top_p: f32) Sampler {
        return .{
            .prng = std.Random.DefaultPrng.init(seed),
            .temp = temp,
            .top_p = top_p,
            .top_k = 64,
            .repeat_penalty = 1.1,
            .min_p = 0.05,
        };
    }

    pub fn sample(self: *Sampler, logits: []f32, recent_tokens: ?[]const u32) u32 {
        const suppress = [_]u32{ 0, 258882, 258883, 255999, 256000, 256001, 255995, 255996, 255997, 255998 };
        for (suppress) |sup| {
            if (sup < logits.len) logits[sup] = -1e9;
        }

        if (self.repeat_penalty > 1.0 and recent_tokens != null) {
            for (recent_tokens.?, 0..) |tok, i| {
                if (isExcludedFromRepeatPenalty(tok)) continue;

                var already_seen = false;
                for (recent_tokens.?[0..i]) |prev| {
                    if (prev == tok) { already_seen = true; break; }
                }
                if (already_seen) continue;

                if (tok < logits.len) {
                    if (logits[tok] > 0.0) {
                        logits[tok] /= self.repeat_penalty;
                    } else {
                        logits[tok] *= self.repeat_penalty;
                    }
                }
            }
        }

        if (self.temp <= 0.0) return kernels.sampleArgmax(logits);

        var top_heap: [64]IndexedLogit = undefined;
        var heap_size: usize = 0;
        const max_k = @min(64, @as(usize, self.top_k));

        for (logits, 0..) |v, i| {
            if (v > -1e8) {
                pushTopK(&top_heap, &heap_size, max_k, .{ .id = @intCast(i), .val = v });
            }
        }
        if (heap_size == 0) return kernels.sampleArgmax(logits);

        const candidates = top_heap[0..heap_size];
        std.mem.sort(IndexedLogit, candidates, {}, struct {
            fn cmp(_: void, a: IndexedLogit, b: IndexedLogit) bool { return a.val > b.val; }
        }.cmp);

        var probs: [64]f32 = undefined;
        var max_scaled: f32 = -1e9;
        const t = if (self.temp > 0.0) self.temp else 1.0;
        for (candidates, 0..) |cand, i| {
            const capped = 30.0 * std.math.tanh(cand.val / 30.0);
            const scaled = capped / t;
            probs[i] = scaled;
            if (scaled > max_scaled) max_scaled = scaled;
        }

        var sum: f32 = 0.0;
        for (0..heap_size) |i| {
            probs[i] = @exp(probs[i] - max_scaled);
            sum += probs[i];
        }
        for (0..heap_size) |i| probs[i] /= sum;

        // Min-P filter
        const max_p = probs[0];
        const p_thresh = max_p * self.min_p;
        var valid_k: usize = 0;
        var valid_sum: f32 = 0.0;
        for (0..heap_size) |i| {
            if (probs[i] >= p_thresh) {
                valid_k = i + 1;
                valid_sum += probs[i];
            } else break;
        }
        if (valid_k == 0) valid_k = 1;
        for (0..valid_k) |i| probs[i] /= valid_sum;

        // Top-P filter
        var cum_p: f32 = 0.0;
        var cutoff: usize = 0;
        for (0..valid_k) |i| {
            cum_p += probs[i];
            cutoff = i + 1;
            if (cum_p >= self.top_p) break;
        }

        const r = self.prng.random().float(f32) * cum_p;
        var acc: f32 = 0.0;
        var chosen_id = candidates[0].id;
        for (0..cutoff) |i| {
            acc += probs[i];
            if (acc >= r) {
                chosen_id = candidates[i].id;
                break;
            }
        }

        return chosen_id;
    }
};
