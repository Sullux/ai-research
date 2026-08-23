const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, "../gemma-4-12B-it/tokenizer.json");
    defer tok.deinit();

    const words = [_][]const u8{
        "don't",
        "Acknowledge",
        "How are you doing today?",
        "I'm doing well, thank you for asking!",
    };

    for (words) |w| {
        const ids = try tok.encode(allocator, w, false);
        defer allocator.free(ids);
        std.debug.print("Word '{s}': ", .{w});
        for (ids) |id| {
            std.debug.print("[{}: '{s}'] ", .{ id, tok.decode(id) });
        }
        std.debug.print("\n", .{});
    }
}
