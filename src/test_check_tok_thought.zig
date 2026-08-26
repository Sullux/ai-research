const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, "../gemma-4-12B-it-qat-q4_0-unquantized/tokenizer.json");
    defer tok.deinit();

    const p1 = "<|turn>model\n<|channel>thought\n";
    const toks1 = try tok.encode(allocator, p1, false);
    defer allocator.free(toks1);

    std.debug.print("Tokens for '{s}':\n", .{p1});
    for (toks1) |t| {
        std.debug.print("  id={} str='{s}'\n", .{ t, tok.decode(t) });
    }

    const p2 = "<|turn>model\n<|thought|>\n";
    const toks2 = try tok.encode(allocator, p2, false);
    defer allocator.free(toks2);

    std.debug.print("\nTokens for '{s}':\n", .{p2});
    for (toks2) |t| {
        std.debug.print("  id={} str='{s}'\n", .{ t, tok.decode(t) });
    }
}
