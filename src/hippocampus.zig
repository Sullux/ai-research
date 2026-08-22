const std = @import("std");
const memory = @import("memory.zig");
const DiffArchive = memory.DiffArchive;
const DynamicRingBuffer = @import("ring_buffer.zig").DynamicRingBuffer;
const PersistentDiffStore = @import("storage.zig").PersistentDiffStore;

pub const StagedItem = struct {
    timestamp: u64,
    salience_norm: f32,
    layer_id: u8,
    is_interrupted: bool,
    token_id: u32,
    slot_idx: usize,
    vector_offset: usize,
};

pub const Hippocampus = struct {
    allocator: std.mem.Allocator,
    dim: usize,
    capacity: usize,
    count: usize,
    debounce_ms: i64,
    last_activity_ms: i64,

    vectors: []f32,
    items: []StagedItem,

    pub fn init(allocator: std.mem.Allocator, dim: usize, capacity: usize, debounce_ms: i64) !Hippocampus {
        const vectors = try allocator.alloc(f32, capacity * dim);
        errdefer allocator.free(vectors);
        const items = try allocator.alloc(StagedItem, capacity);
        errdefer allocator.free(items);

        return .{
            .allocator = allocator,
            .dim = dim,
            .capacity = capacity,
            .count = 0,
            .debounce_ms = debounce_ms,
            .last_activity_ms = 0,
            .vectors = vectors,
            .items = items,
        };
    }

    pub fn deinit(self: *Hippocampus) void {
        self.allocator.free(self.vectors);
        self.allocator.free(self.items);
    }

    pub fn stage(self: *Hippocampus, vector: []const f32, timestamp: u64, salience: f32, layer: u8, token_id: u32, slot: usize, now_ms: i64) void {
        if (self.count >= self.capacity) return;
        const idx = self.count;
        const v_off = idx * self.dim;
        @memcpy(self.vectors[v_off .. v_off + self.dim], vector);
        self.items[idx] = .{
            .timestamp = timestamp,
            .salience_norm = salience,
            .layer_id = layer,
            .is_interrupted = false,
            .token_id = token_id,
            .slot_idx = slot,
            .vector_offset = v_off,
        };
        self.count += 1;
        self.last_activity_ms = now_ms;
    }

    pub fn markInterrupted(self: *Hippocampus) void {
        for (self.items[0..self.count]) |*item| {
            item.is_interrupted = true;
        }
    }

    pub fn shouldFlush(self: *const Hippocampus, now_ms: i64, force: bool) bool {
        if (self.count == 0) return false;
        if (force or self.count >= self.capacity) return true;
        return (now_ms - self.last_activity_ms >= self.debounce_ms);
    }

    pub fn commit(self: *Hippocampus, archive: *DiffArchive, ring: ?*const DynamicRingBuffer, storage: ?*PersistentDiffStore) usize {
        if (self.count == 0) return 0;
        const n = self.count;

        for (self.items[0..n]) |item| {
            const row = self.vectors[item.vector_offset .. item.vector_offset + self.dim];
            archive.appendWithMeta(item.timestamp, item.salience_norm, item.layer_id, item.is_interrupted, item.token_id, row);
            const head_slot = (archive.write_head + archive.capacity - 1) % archive.capacity;

            if (ring) |r| {
                archive.copyKVFromRing(head_slot, r, item.slot_idx);
            }

            if (storage) |s| {
                s.append(item.timestamp, item.salience_norm, item.layer_id, row);
            }
        }

        self.count = 0;
        return n;
    }
};

test "hippocampus stages and flushes after debounce" {
    var arch = try DiffArchive.init(std.testing.allocator, 4, 8, .{});
    defer arch.deinit();
    var hippo = try Hippocampus.init(std.testing.allocator, 4, 16, 6000);
    defer hippo.deinit();

    hippo.stage(&[_]f32{ 1, 0, 0, 0 }, 100, 0.8, 0, 42, 0, 1000);
    try std.testing.expectEqual(@as(usize, 1), hippo.count);
    try std.testing.expect(!hippo.shouldFlush(3000, false));
    try std.testing.expect(hippo.shouldFlush(7500, false));

    const flushed = hippo.commit(&arch, null, null);
    try std.testing.expectEqual(@as(usize, 1), flushed);
    try std.testing.expectEqual(@as(usize, 0), hippo.count);
    try std.testing.expectEqual(@as(usize, 1), arch.count);
}
