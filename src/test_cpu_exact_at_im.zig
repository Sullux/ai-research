const std = @import("std");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const model_dir = "../gemma-4-12B-it";
    var config_path_buf: [512]u8 = undefined;
    const config = try model.ModelConfig.loadFromJson(allocator, try std.fmt.bufPrint(&config_path_buf, "{s}/config.json", .{model_dir}));
    var tok_path_buf: [512]u8 = undefined;
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{model_dir}));
    defer tok.deinit();

    var st = try safetensors.SafeTensors.openDir(allocator, model_dir);
    defer st.deinit();
    var m = try model.Model.loadFromSafeTensors(allocator, &st, config);
    defer m.deinit();

    var kernel_buf: [4096]u8 = undefined;
    const kernel_bytes = try std.fs.cwd().readFile("tui/PROMPT_KERNEL.md", &kernel_buf);

    var p1_buf: [8192]u8 = undefined;
    const p1 = try std.fmt.bufPrint(&p1_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n<|channel>thought\nThe user is asking \"How are you doing today?\".\nThis is a standard social greeting.\nI should respond politely and informatively, acknowledging my nature as an AI.<channel|>I'm doing well, thank you for asking! I'", .{kernel_bytes});
    const tokens = try tok.encode(allocator, p1, true);
    defer allocator.free(tokens);

    std.debug.print("Total tokens: {}\n", .{tokens.len});
    std.debug.print("Last 3 tokens: {} ('{s}'), {} ('{s}'), {} ('{s}')\n", .{
        tokens[tokens.len - 3], tok.decode(tokens[tokens.len - 3]),
        tokens[tokens.len - 2], tok.decode(tokens[tokens.len - 2]),
        tokens[tokens.len - 1], tok.decode(tokens[tokens.len - 1]),
    });

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 384, 512, 96);
    defer ring.deinit();

    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    var thread_pool: std.Thread.Pool = undefined;
    try thread_pool.init(.{ .allocator = allocator, .n_jobs = 16 });
    defer thread_pool.deinit();

    std.debug.print("Running CPU BF16 forward (parallel 16 threads)...\n", .{});
    const start = std.time.milliTimestamp();
    for (tokens, 0..) |t, i| {
        const is_last = (i == tokens.len - 1);
        _ = m.forwardToken(&ring, &scratch, t, i, &thread_pool, null, null, null, is_last);
        if ((i + 1) % 50 == 0 or is_last) {
            std.debug.print("  [{d:3}/{d:3}] elapsed: {d}ms\n", .{ i + 1, tokens.len, std.time.milliTimestamp() - start });
        }
    }

    std.debug.print("\n=== CPU BF16 Ground Truth Top 5 Candidates after 'I'': ===\n", .{});
    var top5: [5]struct { id: u32, val: f32 } = undefined;
    for (&top5) |*t| t.* = .{ .id = 0, .val = -1e9 };
    for (scratch.logits, 0..) |val, id| {
        if (id == 0 or id == 2 or id == 258882 or id == 258883) continue;
        for (0..5) |j| {
            if (val > top5[j].val) {
                var k: usize = 4;
                while (k > j) : (k -= 1) top5[k] = top5[k - 1];
                top5[j] = .{ .id = @intCast(id), .val = val };
                break;
            }
        }
    }
    for (top5) |t| std.debug.print("  token {d:6} ('{s}'): {d:.3}\n", .{ t.id, tok.decode(t.id), t.val });
}
