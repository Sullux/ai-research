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

const LANDMARK_SIMILARITY_THRESHOLD: f32 = 0.98;

/// Phase 2 (Step 2.1): commit salient state transitions to the diff archive
/// and repopulate the Tier-3 associative recall slots from cached KV states.
pub fn integrateMemory(self: *const Model, mem: *memory.DiffArchive, ring: *DynamicRingBuffer, scratch: *ForwardScratch, clock: usize, H: usize, tp: ?*std.Thread.Pool) void {
    _ = self;
    _ = tp;
    _ = H;
    const delta = diff.computeDelta(scratch.delta_x, scratch.normed_x, scratch.prev_normed_x, 0.0);
    const similarity = diff.cosineSimilarity(scratch.prev_normed_x, scratch.normed_x);

    if (similarity < LANDMARK_SIMILARITY_THRESHOLD) {
        const mem_idx = mem.write_head;
        mem.append(@intCast(clock), delta.norm, 0, scratch.normed_x);
        const slot = ring.getSlotIndex(clock);
        mem.copyKVFromRing(mem_idx, ring, slot);
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
        const mem_ts = mem.metas[mi].timestamp;
        mem.copyKVToRing(mi, ring, rank, @intCast(mem_ts));
    }
}
