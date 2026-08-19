const std = @import("std");
pub const safetensors = @import("safetensors.zig");
pub const tensor = @import("tensor.zig");
pub const kernels = @import("kernels.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const model = @import("model.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var thread_pool: std.Thread.Pool = undefined;
    try thread_pool.init(.{ .allocator = allocator });
    defer thread_pool.deinit();

    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();

    const model_path = "../gemma-4-E2B/model.safetensors";
    const tokenizer_path = "../gemma-4-E2B/tokenizer.json";

    try stdout.print("Loading Gemma engine...\n", .{});
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, tokenizer_path);
    defer tok.deinit();

    var st = try safetensors.SafeTensors.open(allocator, model_path);
    defer st.deinit();

    const config = model.ModelConfig{
        .vocab_size = 262144,
        .hidden_size = 1536,
        .intermediate_size = 12288,
        .hidden_size_per_layer_input = 256,
        .num_hidden_layers = 35,
        .num_attention_heads = 8,
        .num_key_value_heads = 1,
        .head_dim = 256,
        .global_head_dim = 512,
        .max_seq_len = 2048,
    };

    var m = try model.Model.loadFromSafeTensors(allocator, &st, config);
    defer m.deinit();

    var cache = try model.KVCache.init(allocator, config.num_hidden_layers, config.max_seq_len, config.global_head_dim);
    defer cache.deinit();

    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len > 1) {
        var prompt_buf = std.ArrayList(u8).init(allocator);
        defer prompt_buf.deinit();
        for (args[1..], 0..) |arg, i| {
            if (i > 0) try prompt_buf.append(' ');
            try prompt_buf.appendSlice(arg);
        }
        try runInference(&m, &tok, &cache, &scratch, prompt_buf.items, &thread_pool, stdout, allocator);
        return;
    }

    try stdout.print("\n=== Gemma 4 Interactive Inference REPL ===\n", .{});
    try stdout.print("Type your prompt and press Enter. Type 'exit' or Ctrl+C to quit.\n\n", .{});

    var line_buf: [4096]u8 = undefined;
    while (true) {
        try stdout.print(">>> ", .{});
        const maybe_line = try stdin.readUntilDelimiterOrEof(&line_buf, '\n');
        const line = maybe_line orelse break;
        const trimmed = std.mem.trim(u8, line, " \r\t");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) break;

        try runInference(&m, &tok, &cache, &scratch, trimmed, &thread_pool, stdout, allocator);
        try stdout.print("\n\n", .{});
    }
}

fn runInference(
    m: *const model.Model,
    tok: *const tokenizer.Tokenizer,
    cache: *model.KVCache,
    scratch: *model.ForwardScratch,
    prompt: []const u8,
    thread_pool: *std.Thread.Pool,
    stdout: anytype,
    allocator: std.mem.Allocator,
) !void {
    const prompt_tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(prompt_tokens);

    // Reset KV cache state
    @memset(cache.k, 0);
    @memset(cache.v, 0);

    for (prompt_tokens, 0..) |t, pos| {
        m.forwardToken(cache, scratch, t, pos, thread_pool);
    }

    // Inspect top 5 logits after prefill
    var top_ids: [5]u32 = undefined;
    var top_vals: [5]f32 = [_]f32{-1e9} ** 5;
    for (scratch.logits, 0..) |v, i| {
        for (0..5) |k| {
            if (v > top_vals[k]) {
                var shift: usize = 4;
                while (shift > k) : (shift -= 1) {
                    top_vals[shift] = top_vals[shift - 1];
                    top_ids[shift] = top_ids[shift - 1];
                }
                top_vals[k] = v;
                top_ids[k] = @intCast(i);
                break;
            }
        }
    }
    try stdout.print("[Top predictions: ", .{});
    for (top_ids, top_vals) |id, val| {
        try stdout.print("'{s}' ({d:.2}) ", .{ tok.decode(id), val });
    }
    try stdout.print("]\n", .{});

    var current_token = kernels.sampleArgmax(scratch.logits);
    var pos = prompt_tokens.len;
    const max_new_tokens: usize = 32;

    for (0..max_new_tokens) |_| {
        const token_str = tok.decode(current_token);
        try stdout.print("{s}", .{token_str});

        if (current_token == tok.eos_token_id) break;

        m.forwardToken(cache, scratch, current_token, pos, thread_pool);
        current_token = kernels.sampleArgmax(scratch.logits);
        pos += 1;
    }
}

test {
    _ = @import("safetensors.zig");
    _ = @import("tensor.zig");
    _ = @import("kernels.zig");
    _ = @import("tokenizer.zig");
    _ = @import("model.zig");
}
