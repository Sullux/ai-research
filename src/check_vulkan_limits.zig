const std = @import("std");
const context = @import("gpu/context.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var gpu_ctx = try context.GpuContext.init(allocator);
    defer gpu_ctx.deinit();

    var props: @import("gpu/types.zig").VkPhysicalDeviceProperties = undefined;
    gpu_ctx.api.vkGetPhysicalDeviceProperties(gpu_ctx.physical_device, &props);

    std.debug.print("maxStorageBufferRange = {} bytes ({d:.2} GB)\n", .{
        props.limits.maxStorageBufferRange,
        @as(f64, @floatFromInt(props.limits.maxStorageBufferRange)) / 1024.0 / 1024.0 / 1024.0,
    });
    std.debug.print("maxComputeWorkGroupInvocations = {}\n", .{props.limits.maxComputeWorkGroupInvocations});
}
