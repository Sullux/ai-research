const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, "../gemma-4-12B-it/tokenizer.json");
    defer tok.deinit();

    const str1 = "<|turn>system\n<|think|>\nkernel\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n<|channel>thought\n";
    const tok1 = try tok.encode(allocator, str1, true);
    defer allocator.free(tok1);

    const str2 = "<|turn>user\nIn your context, what tools do you see that you have available?<turn|>\n<|turn>model\n<|channel>thought\n";
    const tok2 = try tok.encode(allocator, str2, false);
    defer allocator.free(tok2);

    std.debug.print("tok1 tail 8 tokens:\n", .{});
    for (tok1[tok1.len - 8 ..]) |t| std.debug.print("  token {} ('{s}')\n", .{ t, tok.decode(t) });

    std.debug.print("tok2 tail 8 tokens:\n", .{});
    for (tok2[tok2.len - 8 ..]) |t| std.debug.print("  token {} ('{s}')\n", .{ t, tok.decode(t) });
}
