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

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var model_dir: []const u8 = "../gemma-4-E2B";
    var max_tokens: usize = 128;
    var prompt_buf = std.ArrayList(u8).init(allocator);
    defer prompt_buf.deinit();

    var arg_idx: usize = 1;
    while (arg_idx < args.len) {
        const arg = args[arg_idx];
        if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            if (arg_idx + 1 < args.len) {
                model_dir = args[arg_idx + 1];
                arg_idx += 2;
                continue;
            }
        } else if (std.mem.eql(u8, arg, "--max-tokens") or std.mem.eql(u8, arg, "-n")) {
            if (arg_idx + 1 < args.len) {
                max_tokens = std.fmt.parseInt(usize, args[arg_idx + 1], 10) catch 128;
                arg_idx += 2;
                continue;
            }
        } else {
            if (prompt_buf.items.len > 0) try prompt_buf.append(' ');
            try prompt_buf.appendSlice(arg);
        }
        arg_idx += 1;
    }

    try stdout.print("Loading model from: {s}...\n", .{model_dir});

    var config_path_buf: [512]u8 = undefined;
    const config_path = try std.fmt.bufPrint(&config_path_buf, "{s}/config.json", .{model_dir});
    const config = try model.ModelConfig.loadFromJson(allocator, config_path);

    var tok_path_buf: [512]u8 = undefined;
    const tok_path = try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{model_dir});
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, tok_path);
    defer tok.deinit();

    var st = try safetensors.SafeTensors.openDir(allocator, model_dir);
    defer st.deinit();

    var m = try model.Model.loadFromSafeTensors(allocator, &st, config);
    defer m.deinit();

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * config.num_key_value_heads;
    var cache = try model.KVCache.init(allocator, config.num_hidden_layers, config.max_seq_len, max_kv_dim);
    defer cache.deinit();

    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    if (prompt_buf.items.len > 0) {
        try runInference(&m, &tok, &cache, &scratch, prompt_buf.items, max_tokens, &thread_pool, stdout, allocator);
        try stdout.print("\n", .{});
        return;
    }

    try stdout.print("\n=== Gemma 4 Interactive REPL ({s}) ===\n", .{model_dir});
    try stdout.print("Type your prompt and press Enter. Type 'exit' or Ctrl+C to quit.\n\n", .{});

    var line_buf: [4096]u8 = undefined;
    while (true) {
        try stdout.print(">>> ", .{});
        const maybe_line = try stdin.readUntilDelimiterOrEof(&line_buf, '\n');
        const line = maybe_line orelse break;
        const trimmed = std.mem.trim(u8, line, " \r\t");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) break;

        try runInference(&m, &tok, &cache, &scratch, trimmed, max_tokens, &thread_pool, stdout, allocator);
        try stdout.print("\n\n", .{});
    }
}

fn printToken(stdout: anytype, token_str: []const u8) !void {
    var i: usize = 0;
    while (i < token_str.len) {
        if (i + 2 < token_str.len and token_str[i] == 0xE2 and token_str[i + 1] == 0x96 and token_str[i + 2] == 0x81) {
            try stdout.print(" ", .{});
            i += 3;
        } else {
            try stdout.print("{c}", .{token_str[i]});
            i += 1;
        }
    }
}

fn runInference(
    m: *const model.Model,
    tok: *const tokenizer.Tokenizer,
    cache: *model.KVCache,
    scratch: *model.ForwardScratch,
    prompt: []const u8,
    max_tokens: usize,
    thread_pool: *std.Thread.Pool,
    stdout: anytype,
    allocator: std.mem.Allocator,
) !void {
    const prompt_tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(prompt_tokens);

    @memset(cache.k, 0);
    @memset(cache.v, 0);

    for (prompt_tokens, 0..) |t, pos| {
        m.forwardToken(cache, scratch, t, pos, thread_pool);
    }

    var current_token = kernels.sampleArgmax(scratch.logits);
    var pos = prompt_tokens.len;

    for (0..max_tokens) |_| {
        const token_str = tok.decode(current_token);
        try printToken(stdout, token_str);

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
}
