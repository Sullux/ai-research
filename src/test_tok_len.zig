const std = @import("std");
const model = @import("model.zig");
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

    var full_prompt = std.ArrayList(u8).init(allocator);
    defer full_prompt.deinit();
    try full_prompt.appendSlice("<|turn>system\n");
    try full_prompt.appendSlice(txt);
    try full_prompt.appendSlice("\n<turn|>\n<|turn>user\nHow are you doing today?\n<turn|>\n<|turn>model\n");

    const tokens = try tok.encode(allocator, full_prompt.items, true);
    defer allocator.free(tokens);

    std.debug.print("Full prompt token count: {}\n", .{tokens.len});
    for (tokens[0..@min(20, tokens.len)]) |t| {
        std.debug.print("{} ('{s}') ", .{ t, tok.decode(t) });
    }
    std.debug.print("\n", .{});
}
