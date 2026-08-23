const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tok_path_buf: [512]u8 = undefined;
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{"../gemma-4-12B-it"}));
    defer tok.deinit();

    const t1 = try tok.encode(allocator, "thought", false);
    defer allocator.free(t1);
    const t2 = try tok.encode(allocator, "\nthought\n", false);
    defer allocator.free(t2);
    const t4 = try tok.encode(allocator, "<|channel>thought\n", false);
    defer allocator.free(t4);
    std.debug.print("encode('<|channel>thought\\n'): {any}\n", .{t4});
    for (t4) |id| std.debug.print("  {}: \"{s}\"\n", .{ id, tok.decode(id) });
}
