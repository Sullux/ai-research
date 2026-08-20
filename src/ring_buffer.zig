const std = @import("std");

/// Fixed per-layer context allocation split into three zones:
///   [0 .. num_anchors)                       Tier 1: static anchors
///   [num_anchors .. num_anchors+window_size) Tier 2: sliding FIFO window
///   [num_anchors+window_size .. total)       Tier 3: associative recall slots
pub const DynamicRingBuffer = struct {
    allocator: std.mem.Allocator,
    num_layers: usize,
    max_kv_dim: usize,
    num_anchors: usize,
    window_size: usize,
    num_recall: usize,
    total_slots: usize,

    k: []f32,
    v: []f32,
    clocks: []usize,
    active: []bool,
    total_ingested: usize,

    pub fn init(allocator: std.mem.Allocator, num_layers: usize, max_kv_dim: usize, num_anchors: usize, window_size: usize, num_recall: usize) !DynamicRingBuffer {
        const total_slots = num_anchors + window_size + num_recall;
        const total_elements = num_layers * total_slots * max_kv_dim;

        const k_buf = try allocator.alloc(f32, total_elements);
        const v_buf = try allocator.alloc(f32, total_elements);
        const clocks_buf = try allocator.alloc(usize, num_layers * total_slots);
        const active_buf = try allocator.alloc(bool, num_layers * total_slots);

        @memset(k_buf, 0); @memset(v_buf, 0); @memset(clocks_buf, 0); @memset(active_buf, false);

        return .{
            .allocator = allocator, .num_layers = num_layers, .max_kv_dim = max_kv_dim,
            .num_anchors = num_anchors, .window_size = window_size, .num_recall = num_recall,
            .total_slots = total_slots, .k = k_buf, .v = v_buf, .clocks = clocks_buf,
            .active = active_buf, .total_ingested = 0,
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

    inline fn recallStart(self: *const DynamicRingBuffer) usize {
        return self.num_anchors + self.window_size;
    }

    pub fn getSlotIndex(self: *const DynamicRingBuffer, clock: usize) usize {
        if (clock < self.num_anchors) return clock;
        return self.num_anchors + ((clock - self.num_anchors) % self.window_size);
    }

    pub fn activateSlot(self: *DynamicRingBuffer, layer: usize, clock: usize) usize {
        const slot = self.getSlotIndex(clock);
        const slot_idx = layer * self.total_slots + slot;
        self.clocks[slot_idx] = clock;
        self.active[slot_idx] = true;
        if (layer == 0 and clock >= self.total_ingested) self.total_ingested = clock + 1;
        return slot;
    }

    pub fn writeKV(self: *DynamicRingBuffer, layer: usize, clock: usize, k_src: []const f32, v_src: []const f32) void {
        const slot = self.getSlotIndex(clock);
        const slot_idx = layer * self.total_slots + slot;
        const kv_offset = slot_idx * self.max_kv_dim;

        @memcpy(self.k[kv_offset .. kv_offset + k_src.len], k_src);
        @memcpy(self.v[kv_offset .. kv_offset + v_src.len], v_src);
        self.clocks[slot_idx] = clock;
        self.active[slot_idx] = true;
        if (layer == 0 and clock >= self.total_ingested) self.total_ingested = clock + 1;
    }

    pub fn clearRecall(self: *DynamicRingBuffer) void {
        const start = self.recallStart();
        for (0..self.num_layers) |l| {
            const base = l * self.total_slots + start;
            @memset(self.active[base .. base + self.num_recall], false);
        }
    }

    pub fn writeRecallKV(self: *DynamicRingBuffer, layer: usize, rank: usize, k_src: []const f32, v_src: []const f32, clock: usize) void {
        const slot = self.recallStart() + rank;
        const slot_idx = layer * self.total_slots + slot;
        const kv_offset = slot_idx * self.max_kv_dim;

        @memcpy(self.k[kv_offset .. kv_offset + k_src.len], k_src);
        @memcpy(self.v[kv_offset .. kv_offset + v_src.len], v_src);
        self.clocks[slot_idx] = clock;
        self.active[slot_idx] = true;
    }

    pub fn getActiveSlots(self: *const DynamicRingBuffer, layer: usize, curr_clock: usize, out_slots: []usize) usize {
        const layer_offset = layer * self.total_slots;
        var count: usize = 0;

        const anchor_limit = @min(curr_clock + 1, self.num_anchors);
        for (0..anchor_limit) |s| {
            if (self.active[layer_offset + s]) {
                out_slots[count] = s;
                count += 1;
            }
        }

        if (curr_clock >= self.num_anchors) {
            const min_valid_clock = if (curr_clock >= self.window_size) curr_clock - self.window_size + 1 else self.num_anchors;
            for (self.num_anchors..self.recallStart()) |s| {
                const global_idx = layer_offset + s;
                if (!self.active[global_idx]) continue;
                const slot_clock = self.clocks[global_idx];
                if (slot_clock >= min_valid_clock and slot_clock <= curr_clock) {
                    out_slots[count] = s;
                    count += 1;
                }
            }
        }

        for (self.recallStart()..self.total_slots) |s| {
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

    try std.testing.expectEqual(@as(usize, 16), ring.total_slots);
    var dummy_k: [64]f32 = undefined;
    var dummy_v: [64]f32 = undefined;
    @memset(&dummy_k, 1.0);
    @memset(&dummy_v, 2.0);

    for (0..20) |c| ring.writeKV(0, c, &dummy_k, &dummy_v);

    var slots_buf: [32]usize = undefined;
    const count = ring.getActiveSlots(0, 19, &slots_buf);
    try std.testing.expectEqual(@as(usize, 12), count);
    try std.testing.expectEqual(@as(usize, 0), slots_buf[0]);
    try std.testing.expectEqual(@as(usize, 3), slots_buf[3]);
}

test "recall slots are written, cleared, and included in active set" {
    var ring = try DynamicRingBuffer.init(std.testing.allocator, 1, 8, 2, 4, 3);
    defer ring.deinit();

    var k_src: [8]f32 = undefined;
    var v_src: [8]f32 = undefined;
    @memset(&k_src, 1.0);
    @memset(&v_src, 2.0);

    ring.writeRecallKV(0, 0, &k_src, &v_src, 100);
    ring.writeRecallKV(0, 1, &k_src, &v_src, 50);

    var slots_buf: [16]usize = undefined;
    const count = ring.getActiveSlots(0, 101, &slots_buf);
    try std.testing.expectEqual(@as(usize, 2), count);

    ring.clearRecall();
    const after_clear = ring.getActiveSlots(0, 101, &slots_buf);
    try std.testing.expectEqual(@as(usize, 0), after_clear);
}
