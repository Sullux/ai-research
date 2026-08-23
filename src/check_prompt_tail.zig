const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const model_dir = "../gemma-4-12B-it";
    var tok_path_buf: [512]u8 = undefined;
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{model_dir}));
    defer tok.deinit();

    var kernel_buf: [4096]u8 = undefined;
    const kernel_bytes = try std.fs.cwd().readFile("tui/PROMPT_KERNEL.md", &kernel_buf);
    var full_prompt_buf: [8192]u8 = undefined;
    const prompt = try std.fmt.bufPrint(&full_prompt_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHello, how are you today?<turn|>\n<|turn>model\n", .{kernel_bytes});
    const all_tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(all_tokens);

    std.debug.print("Total tokens: {}\n", .{all_tokens.len});
    std.debug.print("Last 10 tokens:\n", .{});
    const start = if (all_tokens.len > 10) all_tokens.len - 10 else 0;
    for (all_tokens[start..], start..) |t, i| {
        std.debug.print("  [{}] {} ('{s}')\n", .{ i, t, tok.decode(t) });
    }
}
