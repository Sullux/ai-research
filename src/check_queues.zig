const std = @import("std");
const gpu = @import("gpu.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    var ctx = try gpu.context.GpuContext.init(alloc);
    defer ctx.deinit();

    std.debug.print("Selected Device: {s}\n", .{ctx.device_name});
    std.debug.print("Selected Queue Family Index: {d}\n", .{ctx.queue_family_index});

    var num_q: u32 = 0;
    ctx.api.vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device, &num_q, null);
    const q_props = try alloc.alloc(gpu.types.VkQueueFamilyProperties, num_q);
    defer alloc.free(q_props);
    ctx.api.vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device, &num_q, q_props.ptr);

    for (q_props, 0..) |qp, idx| {
        std.debug.print("Queue {d}: flags=0x{X} (graphics={}, compute={}, transfer={})\n", .{
            idx,
            qp.queueFlags,
            (qp.queueFlags & gpu.types.VK_QUEUE_GRAPHICS_BIT) != 0,
            (qp.queueFlags & gpu.types.VK_QUEUE_COMPUTE_BIT) != 0,
            (qp.queueFlags & 0x4) != 0,
        });
    }
}
