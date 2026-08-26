const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const model_dir = "../gemma-4-12B-it-qat-q4_0-unquantized";
    var tok = try tokenizer.Tokenizer.loadFromJson(alloc, model_dir ++ "/tokenizer.json");
    defer tok.deinit();

    const test_strings = [_][]const u8{
        "Acknow",
        "Acknowledge",
        "acknowledge",
        "Acknow and respond",
        "Acknowledge and respond",
        "Plan:\n1. Acknowledge and respond to the greeting.",
    };

    for (test_strings) |str| {
        const ids = try tok.encode(alloc, str, false);
        defer alloc.free(ids);
        std.debug.print("String: \"{s}\"\n  tokens: ", .{str});
        for (ids) |id| {
            std.debug.print("{d} ('{s}') ", .{ id, tok.decode(id) });
        }
        std.debug.print("\n", .{});
    }
}
