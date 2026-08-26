const std = @import("std");
const context = @import("gpu/context.zig");
const types = @import("gpu/types.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    var ctx = try context.GpuContext.init(alloc);
    defer ctx.deinit();

    std.debug.print("Available memory types ({d}):\n", .{ctx.mem_props.memoryTypeCount});
    for (0..ctx.mem_props.memoryTypeCount) |i| {
        const flags = ctx.mem_props.memoryTypes[i].propertyFlags;
        std.debug.print("  Type {d}: flags=0x{x:0>4} [", .{ i, flags });
        if ((flags & types.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) std.debug.print(" DEVICE_LOCAL", .{});
        if ((flags & types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) std.debug.print(" HOST_VISIBLE", .{});
        if ((flags & types.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) std.debug.print(" HOST_COHERENT", .{});
        if ((flags & types.VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) std.debug.print(" HOST_CACHED", .{});
        std.debug.print(" ]\n", .{});
    }
}
