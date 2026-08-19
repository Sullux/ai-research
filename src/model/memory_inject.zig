const std = @import("std");
pub const kernels = @import("../kernels.zig");
pub const diff = @import("../diff.zig");
pub const memory = @import("../memory.zig");
pub const types = @import("types.zig");
pub const loader = @import("loader.zig");
pub const ring_buffer = @import("../ring_buffer.zig");

const Model = loader.Model;
const ForwardScratch = types.ForwardScratch;
const DynamicRingBuffer = ring_buffer.DynamicRingBuffer;

/// A state transition is committed to the diff archive when consecutive final
/// normalized states diverge below this cosine similarity.
const LANDMARK_SIMILARITY_THRESHOLD: f32 = 0.98;

/// Phase 2 (Step 2.1): commit salient state transitions to the diff archive
/// and repopulate the Tier-3 associative recall slots for the next cycle.
pub fn integrateMemory(self: *const Model, mem: *memory.DiffArchive, ring: *DynamicRingBuffer, scratch: *ForwardScratch, clock: usize, H: usize, tp: ?*std.Thread.Pool) void {
    const delta = diff.computeDelta(scratch.delta_x, scratch.normed_x, scratch.prev_normed_x, 0.0);
    const similarity = diff.cosineSimilarity(scratch.prev_normed_x, scratch.normed_x);

    if (similarity < LANDMARK_SIMILARITY_THRESHOLD) {
        mem.append(@intCast(clock), delta.norm, 0, scratch.normed_x);
    }
    @memcpy(scratch.prev_normed_x, scratch.normed_x);

    const recall_count = @min(ring.num_recall, scratch.recall_indices.len);
    if (recall_count == 0) {
        ring.clearRecall();
        return;
    }

    const selected = mem.scan(scratch.normed_x, @intCast(clock), scratch.recall_indices[0..recall_count], recall_count);
    ring.clearRecall();

    for (0..selected) |rank| {
        const mi = scratch.recall_indices[rank];
        const mem_vec = mem.vectors[mi * H .. (mi + 1) * H];
        const mem_ts = mem.metas[mi].timestamp;
        injectRecall(self, ring, scratch, mem_vec, mem_ts, rank, H, tp);
    }
}

/// Project a recalled hidden-state snapshot into each layer's K/V space and
/// write it into that layer's Tier-3 recall slot, RoPE-rotated at its true
/// token clock so attention perceives real temporal distance.
fn injectRecall(self: *const Model, ring: *DynamicRingBuffer, scratch: *ForwardScratch, mem_vec: []const f32, mem_ts: u64, rank: usize, H: usize, tp: ?*std.Thread.Pool) void {
    const first_shared = self.config.num_hidden_layers - self.config.num_kv_shared_layers;

    for (self.layers, 0..) |l, layer_idx| {
        if (self.config.num_kv_shared_layers > 0 and layer_idx >= first_shared) continue;

        kernels.rmsNorm(scratch.normed_x, mem_vec, l.input_layernorm, self.config.rms_norm_eps);

        if (tp) |pool| {
            kernels.gemvParallel(scratch.k[0..l.kv_dim], scratch.normed_x, l.k_proj, l.kv_dim, H, pool);
            if (!l.k_eq_v and l.v_proj.len > 0) kernels.gemvParallel(scratch.v[0..l.kv_dim], scratch.normed_x, l.v_proj, l.kv_dim, H, pool);
        } else {
            kernels.gemv(scratch.k[0..l.kv_dim], scratch.normed_x, l.k_proj, l.kv_dim, H);
            if (!l.k_eq_v and l.v_proj.len > 0) kernels.gemv(scratch.v[0..l.kv_dim], scratch.normed_x, l.v_proj, l.kv_dim, H);
        }
        if (l.k_eq_v or l.v_proj.len == 0) @memcpy(scratch.v[0..l.kv_dim], scratch.k[0..l.kv_dim]);

        for (0..l.num_kv_heads) |kv_h| {
            const head_k = scratch.k[kv_h * l.head_dim .. (kv_h + 1) * l.head_dim];
            kernels.rmsNorm(head_k, head_k, l.k_norm, self.config.rms_norm_eps);
            const head_v = scratch.v[kv_h * l.head_dim .. (kv_h + 1) * l.head_dim];
            kernels.unitRmsNorm(head_v, head_v, self.config.rms_norm_eps);
        }

        const theta = if (l.layer_type == .full_attention) self.config.rope_theta_full else self.config.rope_theta;
        kernels.applyRopePartial(scratch.k[0..l.kv_dim], @intCast(mem_ts), l.head_dim, l.rotary_dim, theta);
        ring.writeRecallKV(layer_idx, rank, scratch.k[0..l.kv_dim], scratch.v[0..l.kv_dim], @intCast(mem_ts));
    }
}
