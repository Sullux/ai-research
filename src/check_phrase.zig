const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, "../gemma-4-12B-it/tokenizer.json");
    defer tok.deinit();

    // Let's inspect tokens in "The user is asking..."
    const phrase = "The user is asking \"How are you doing today?\". This is a standard or friendly-opener question. As an AI, I don't have feelings, but I should respond in a helpful and polite manner.";
    const ids = try tok.encode(allocator, phrase, false);
    defer allocator.free(ids);
    std.debug.print("Ground truth tokens:\n", .{});
    for (ids) |id| {
        std.debug.print("[{}: '{s}'] ", .{ id, tok.decode(id) });
    }
    std.debug.print("\n", .{});
}
