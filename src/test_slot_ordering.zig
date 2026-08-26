const std = @import("std");
const ring_buffer = @import("ring_buffer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    var ring = try ring_buffer.DynamicRingBuffer.init(alloc, 48, 256, 32, 2048, 96);
    defer ring.deinit();

    // Ingest 504 tokens in Turn 1
    for (0..504) |c| {
        _ = ring.activateSlot(0, c);
    }

    var active_slots: [2048]usize = undefined;
    const count = ring.getActiveSlots(0, 503, &active_slots);
    std.debug.print("Turn 1 active slot count at clock 503: {d}\n", .{count});

    // Check if slots are in ascending clock order
    var is_sorted = true;
    var prev_clock: usize = 0;
    for (active_slots[0..count], 0..) |s, i| {
        const slot_clock = ring.clocks[s];
        if (i > 0 and slot_clock < prev_clock) {
            std.debug.print("MISMATCH at index {d}: slot {d} has clock {d} < prev_clock {d}\n", .{ i, s, slot_clock, prev_clock });
            is_sorted = false;
        }
        prev_clock = slot_clock;
    }
    std.debug.print("Are active slots in chronological order? {}\n", .{is_sorted});
}
