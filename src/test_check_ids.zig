const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, "../gemma-4-12B-it-qat-q4_0-unquantized/tokenizer.json");
    defer tok.deinit();

    const check = [_]u32{ 236751, 236757, 236780, 236777, 236789, 680, 775, 235248 };
    for (check) |id| {
        std.debug.print("id={}: str='{s}' hex=", .{ id, tok.decode(id) });
        for (tok.decode(id)) |b| std.debug.print("{x:0>2} ", .{b});
        std.debug.print("\n", .{});
    }

    const test_str = "I'm doing well";
    const enc = try tok.encode(allocator, test_str, false);
    defer allocator.free(enc);
    std.debug.print("\nEncoding for '{s}':\n", .{test_str});
    for (enc) |t| {
        std.debug.print("  id={} str='{s}'\n", .{ t, tok.decode(t) });
    }
}
