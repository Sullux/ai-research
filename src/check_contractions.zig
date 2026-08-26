const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    var tok = try tokenizer.Tokenizer.loadFromJson(alloc, "../gemma-4-12B-it-qat-q4_0-unquantized/tokenizer.json");
    defer tok.deinit();

    const phrases = [_][]const u8{
        "I'm", "I's", "don't", "can't", "it's", "we're", "they're", "you're", "I'll", "I'd", "we've", "I've",
        "m", "s", "t", "d", "re", "ve", "ll", "'m", "'s", "'t", "'d", "'re", "'ve", "'ll",
    };

    for (phrases) |p| {
        const ids = try tok.encode(alloc, p, false);
        defer alloc.free(ids);
        std.debug.print("Phrase \"{s:<8}\": tokens = [", .{p});
        for (ids) |id| std.debug.print(" {d} ('{s}')", .{ id, tok.decode(id) });
        std.debug.print(" ]\n", .{});
    }
}
