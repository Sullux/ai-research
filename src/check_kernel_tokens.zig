const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    var tok = try tokenizer.Tokenizer.loadFromJson(alloc, "../gemma-4-12B-it-qat-q4_0-unquantized/tokenizer.json");
    defer tok.deinit();

    const file = try std.fs.cwd().openFile("tui/PROMPT_KERNEL.md", .{});
    defer file.close();
    const size = try file.getEndPos();
    const kbuf = try alloc.alloc(u8, size);
    defer alloc.free(kbuf);
    _ = try file.readAll(kbuf);

    const tokens = try tok.encode(alloc, kbuf, false);
    defer alloc.free(tokens);

    std.debug.print("Kernel token count: {d}\n", .{tokens.len});
    var tool_count: usize = 0;
    for (tokens) |t| {
        if (t == 46 or t == 47 or t == 48 or t == 49 or t == 98 or t == 100 or t == 101 or t == 105 or t == 106) {
            std.debug.print("Special token: id={d} ({s})\n", .{ t, tok.decode(t) });
            tool_count += 1;
        }
    }
    std.debug.print("Total special tool/turn tokens found: {d}\n", .{tool_count});
}
