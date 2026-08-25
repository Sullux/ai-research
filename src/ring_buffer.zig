const std = @import("std");

pub const TOTAL_SLOTS: usize = 4096;
pub const UPPER_RECALL_SLOTS: usize = 128;
pub const SPLIT_LAYER: usize = 16;

/// Fixed 4,096-slot per-layer buffer with asymmetric 3-tier partitioning:
///   Tier 1: [0 .. num_anchors)                           Immutable Dynamic Anchors
///   Tier 2: [num_anchors .. recallStart(l))             Sliding FIFO Ring Window
///   Tier 3: [recallStart(l) .. TOTAL_SLOTS)              Associative Recall (Layers 16..47 only)
pub const DynamicRingBuffer = struct {
    allocator: std.mem.Allocator,
    num_layers: usize,
    max_kv_dim: usize,
    num_anchors: usize,
    total_slots: usize,

    k: []f32,
    v: []f32,
    clocks: []usize,
    active: []bool,
    total_ingested: usize,

    pub fn init(allocator: std.mem.Allocator, num_layers: usize, max_kv_dim: usize, num_anchors: usize, window_size: usize, num_recall: usize) !DynamicRingBuffer {
        _ = window_size; _ = num_recall;
        const slots = TOTAL_SLOTS;
        const total_elements = num_layers * slots * max_kv_dim;

        const k_buf = try allocator.alloc(f32, total_elements);
        const v_buf = try allocator.alloc(f32, total_elements);
        const clocks_buf = try allocator.alloc(usize, num_layers * slots);
        const active_buf = try allocator.alloc(bool, num_layers * slots);

        @memset(k_buf, 0); @memset(v_buf, 0); @memset(clocks_buf, 0); @memset(active_buf, false);

        return .{
            .allocator = allocator, .num_layers = num_layers, .max_kv_dim = max_kv_dim,
            .num_anchors = num_anchors, .total_slots = slots, .k = k_buf, .v = v_buf,
            .clocks = clocks_buf, .active = active_buf, .total_ingested = 0,
        };
    }

    pub fn deinit(self: *DynamicRingBuffer) void {
        self.allocator.free(self.k); self.allocator.free(self.v);
        self.allocator.free(self.clocks); self.allocator.free(self.active);
    }
    pub fn reset(self: *DynamicRingBuffer) void {
        @memset(self.k, 0); @memset(self.v, 0);
        @memset(self.clocks, 0); @memset(self.active, false);
        self.total_ingested = 0;
    }
    pub fn setNumAnchors(self: *DynamicRingBuffer, n: usize) void {
        self.num_anchors = @min(n, self.total_slots - UPPER_RECALL_SLOTS - 64);
    }
    pub inline fn recallSlots(self: *const DynamicRingBuffer, layer: usize) usize {
        return if (layer < SPLIT_LAYER or self.total_slots < UPPER_RECALL_SLOTS * 2) 0 else UPPER_RECALL_SLOTS;
    }
    pub inline fn recallStart(self: *const DynamicRingBuffer, layer: usize) usize {
        return self.total_slots - self.recallSlots(layer);
    }
    pub inline fn windowSize(self: *const DynamicRingBuffer, layer: usize) usize {
        const start = self.recallStart(layer);
        return if (start > self.num_anchors) start - self.num_anchors else 1;
    }

    pub fn getSlotIndex(self: *const DynamicRingBuffer, clock: usize) usize { return self.getSlotIndexLayer(0, clock); }

    pub fn getSlotIndexLayer(self: *const DynamicRingBuffer, layer: usize, clock: usize) usize {
        if (clock < self.num_anchors) return clock;
        return self.num_anchors + ((clock - self.num_anchors) % self.windowSize(layer));
    }

    pub fn activateSlot(self: *DynamicRingBuffer, layer: usize, clock: usize) usize {
        const slot = self.getSlotIndexLayer(layer, clock);
        const slot_idx = layer * self.total_slots + slot;
        self.clocks[slot_idx] = clock;
        self.active[slot_idx] = true;
        if (layer == 0 and clock >= self.total_ingested) self.total_ingested = clock + 1;
        return slot;
    }

    pub fn writeKV(self: *DynamicRingBuffer, layer: usize, clock: usize, k_src: []const f32, v_src: []const f32) void {
        const slot = self.getSlotIndexLayer(layer, clock);
        const slot_idx = layer * self.total_slots + slot;
        const kv_offset = slot_idx * self.max_kv_dim;

        @memcpy(self.k[kv_offset .. kv_offset + k_src.len], k_src);
        @memcpy(self.v[kv_offset .. kv_offset + v_src.len], v_src);
        self.clocks[slot_idx] = clock;
        self.active[slot_idx] = true;
        if (layer == 0 and clock >= self.total_ingested) self.total_ingested = clock + 1;
    }

    pub fn clearRecall(self: *DynamicRingBuffer) void {
        for (SPLIT_LAYER..self.num_layers) |l| {
            const base = l * self.total_slots + self.recallStart(l);
            @memset(self.active[base .. base + self.recallSlots(l)], false);
        }
    }

    pub fn writeRecallKV(self: *DynamicRingBuffer, layer: usize, rank: usize, k_src: []const f32, v_src: []const f32, clock: usize) void {
        if (layer < SPLIT_LAYER) return;
        const slot = self.recallStart(layer) + rank;
        if (slot >= self.total_slots) return;
        const slot_idx = layer * self.total_slots + slot;
        const kv_offset = slot_idx * self.max_kv_dim;
        @memcpy(self.k[kv_offset .. kv_offset + k_src.len], k_src);
        @memcpy(self.v[kv_offset .. kv_offset + v_src.len], v_src);
        self.clocks[slot_idx] = clock; self.active[slot_idx] = true;
    }

    pub fn getActiveSlots(self: *const DynamicRingBuffer, layer: usize, curr_clock: usize, out_slots: []usize) usize {
        return self.getActiveSlotsLayer(layer, curr_clock, false, 1024, out_slots);
    }

    pub fn getActiveSlotsLayer(self: *const DynamicRingBuffer, layer: usize, curr_clock: usize, is_sliding: bool, sliding_window: usize, out_slots: []usize) usize {
        const layer_offset = layer * self.total_slots;
        var count: usize = 0;

        const anchor_limit = @min(curr_clock + 1, self.num_anchors);
        for (0..anchor_limit) |s| {
            if (self.active[layer_offset + s]) {
                out_slots[count] = s;
                count += 1;
            }
        }

        const w_start = self.recallStart(layer);
        const max_cap = self.windowSize(layer);
        const effective_window = if (is_sliding) @min(sliding_window, max_cap) else max_cap;
        if (curr_clock >= self.num_anchors) {
            const min_valid_clock = if (curr_clock >= effective_window) curr_clock - effective_window + 1 else self.num_anchors;
            for (self.num_anchors..w_start) |s| {
                const global_idx = layer_offset + s;
                if (!self.active[global_idx]) continue;
                const slot_clock = self.clocks[global_idx];
                if (slot_clock >= min_valid_clock and slot_clock <= curr_clock) {
                    out_slots[count] = s;
                    count += 1;
                }
            }
        }

        for (w_start..self.total_slots) |s| {
            const global_idx = layer_offset + s;
            if (!self.active[global_idx]) continue;
            if (self.clocks[global_idx] <= curr_clock) {
                out_slots[count] = s;
                count += 1;
            }
        }

        return count;
    }

    pub fn getSlotKV(self: *const DynamicRingBuffer, layer: usize, slot: usize, kv_dim: usize) struct { k: []const f32, v: []const f32, clock: usize } {
        const slot_idx = layer * self.total_slots + slot;
        const offset = slot_idx * self.max_kv_dim;
        return .{
            .k = self.k[offset .. offset + kv_dim],
            .v = self.v[offset .. offset + kv_dim],
            .clock = self.clocks[slot_idx],
        };
    }
};

test "ring buffer tiers and window eviction" {
    var ring = try DynamicRingBuffer.init(std.testing.allocator, 2, 64, 4, 8, 4);
    defer ring.deinit();

    try std.testing.expectEqual(@as(usize, 4096), ring.total_slots);
    var dummy_k: [64]f32 = undefined;
    var dummy_v: [64]f32 = undefined;
    @memset(&dummy_k, 1.0);
    @memset(&dummy_v, 2.0);

    for (0..20) |c| ring.writeKV(0, c, &dummy_k, &dummy_v);

    var slots_buf: [32]usize = undefined;
    const count = ring.getActiveSlots(0, 19, &slots_buf);
    try std.testing.expectEqual(@as(usize, 20), count);
    try std.testing.expectEqual(@as(usize, 0), slots_buf[0]);
    try std.testing.expectEqual(@as(usize, 3), slots_buf[3]);
}

test "recall slots are written, cleared, and included in active set" {
    var ring = try DynamicRingBuffer.init(std.testing.allocator, 32, 8, 2, 4, 3);
    defer ring.deinit();

    var k_src: [8]f32 = undefined;
    var v_src: [8]f32 = undefined;
    @memset(&k_src, 1.0);
    @memset(&v_src, 2.0);

    ring.writeRecallKV(16, 0, &k_src, &v_src, 100);
    ring.writeRecallKV(16, 1, &k_src, &v_src, 50);

    var slots_buf: [16]usize = undefined;
    const count = ring.getActiveSlots(16, 101, &slots_buf);
    try std.testing.expectEqual(@as(usize, 2), count);

    ring.clearRecall();
    const after_clear = ring.getActiveSlots(16, 101, &slots_buf);
    try std.testing.expectEqual(@as(usize, 0), after_clear);
}
