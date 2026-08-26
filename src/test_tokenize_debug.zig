const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const model_dir = "../gemma-4-12B-it-qat-q4_0-unquantized";
    var tok = try tokenizer.Tokenizer.loadFromJson(alloc, model_dir ++ "/tokenizer.json");
    defer tok.deinit();

    const tests = [_][]const u8{
        "I'm doing well",
        "I don't have feelings",
        "I don'",
        "I don't",
        "don't",
        "don",
        "'t",
        "'s",
        "'m",
    };

    for (tests) |str| {
        const ids = try tok.encode(alloc, str, false);
        defer alloc.free(ids);
        std.debug.print("\"{s}\" -> ", .{str});
        for (ids) |id| {
            std.debug.print("{d} ('{s}') ", .{ id, tok.decode(id) });
        }
        std.debug.print("\n", .{});
    }
}
