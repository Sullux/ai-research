const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, "../gemma-4-12B-it/tokenizer.json");
    defer tok.deinit();

    const file = try std.fs.cwd().openFile("tui/PROMPT_KERNEL.md", .{});
    defer file.close();
    const txt = try file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(txt);

    const tokens = try tok.encode(allocator, txt, false);
    defer allocator.free(tokens);

    for (tokens) |t| {
        std.debug.print("[{s}]", .{tok.decode(t)});
    }
    std.debug.print("\n", .{});
}
