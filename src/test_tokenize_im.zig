const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, "../gemma-4-12B-it/tokenizer.json");
    defer tok.deinit();

    const str = "I'm doing well, thank you for asking! I'm ready";
    const tokens = try tok.encode(allocator, str, false);
    defer allocator.free(tokens);

    std.debug.print("Tokens for \"{s}\":\n", .{str});
    for (tokens) |t| {
        std.debug.print("  token {d:6}: '{s}'\n", .{ t, tok.decode(t) });
    }

    const str2 = "<channel|>I'm";
    const tokens2 = try tok.encode(allocator, str2, false);
    defer allocator.free(tokens2);
    std.debug.print("\nTokens for \"{s}\":\n", .{str2});
    for (tokens2) |t| {
        std.debug.print("  token {d:6}: '{s}'\n", .{ t, tok.decode(t) });
    }
}
