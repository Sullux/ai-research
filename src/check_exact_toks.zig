const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const model_dir = "../gemma-4-12B-it-qat-q4_0-unquantized";
    var tok = try tokenizer.Tokenizer.loadFromJson(alloc, model_dir ++ "/tokenizer.json");
    defer tok.deinit();

    const test_chars = [_][]const u8{
        "m", "'m", "s", "'s", "t", "'t", "d", "'d", "re", "'re", "ve", "'ve", "ll", "'ll",
        " ", "  ", "_", "-", "--", "---", "ed", "ge", "or", "don't", "don", "'",
        "acknowledge", "acknowledging", "Acknowledge",
    };

    for (test_chars) |tc| {
        const enc = try tok.encode(alloc, tc, false);
        defer alloc.free(enc);
        std.debug.print("Input: \"{s}\" -> ", .{tc});
        for (enc) |id| {
            std.debug.print("{d} ('{s}') ", .{ id, tok.decode(id) });
        }
        std.debug.print("\n", .{});
    }
}
