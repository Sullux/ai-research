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
    try stdout.print("====================================================\n", .{});
    try stdout.print("    Hierarchical LLM Inference Engine (Zig)        \n", .{});
    try stdout.print("====================================================\n\n", .{});

    const model_path = "../gemma-4-E2B/model.safetensors";
    const tokenizer_path = "../gemma-4-E2B/tokenizer.json";

    try stdout.print("[1/3] Loading tokenizer from: {s}...\n", .{tokenizer_path});
    var timer = try std.time.Timer.start();
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, tokenizer_path);
    defer tok.deinit();
    try stdout.print("      Tokenizer loaded ({d} tokens) in {d:.2} ms.\n\n", .{
        tok.id_to_token.len,
        @as(f64, @floatFromInt(timer.read())) / 1_000_000.0,
    });

    try stdout.print("[2/3] Mapping weights from: {s}...\n", .{model_path});
    timer.reset();
    var st = try safetensors.SafeTensors.open(allocator, model_path);
    defer st.deinit();
    try stdout.print("      Memory-mapped {d} tensors in {d:.2} ms.\n\n", .{
        st.tensors.count(),
        @as(f64, @floatFromInt(timer.read())) / 1_000_000.0,
    });

    const config = model.ModelConfig{
        .vocab_size = 262144,
        .hidden_size = 1536,
        .intermediate_size = 6144,
        .hidden_size_per_layer_input = 256,
        .num_hidden_layers = 35,
        .num_attention_heads = 8,
        .num_key_value_heads = 1,
        .head_dim = 256,
        .global_head_dim = 512,
        .max_seq_len = 1024,
    };

    try stdout.print("[3/3] Binding 35-layer Gemma 4 architecture...\n", .{});
    timer.reset();
    var m = try model.Model.loadFromSafeTensors(allocator, &st, config);
    defer m.deinit();
    try stdout.print("      Layers initialized in {d:.2} ms.\n\n", .{
        @as(f64, @floatFromInt(timer.read())) / 1_000_000.0,
    });

    // Allocate KV Cache and forward scratch buffers
    var cache = try model.KVCache.init(allocator, config.num_hidden_layers, config.max_seq_len, config.global_head_dim);
    defer cache.deinit();

    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    const prompt = "The capital of France is";
    const prompt_tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(prompt_tokens);

    try stdout.print("Prompt: \"{s}\"\n", .{prompt});
    try stdout.print("Token IDs: {any}\n", .{prompt_tokens});
    try stdout.print("Prefilling KV cache with {d} prompt tokens...\n", .{prompt_tokens.len});

    var prefill_timer = try std.time.Timer.start();
    for (prompt_tokens, 0..) |t, pos| {
        m.forwardToken(&cache, &scratch, t, pos, &thread_pool);
    }
    const prefill_ms = @as(f64, @floatFromInt(prefill_timer.read())) / 1_000_000.0;
    try stdout.print("Prefill complete in {d:.2} ms.\n\n", .{prefill_ms});

    try stdout.print("Generating continuation:\n{s}", .{prompt});

    var current_token = kernels.sampleArgmax(scratch.logits);
    var pos = prompt_tokens.len;
    const max_new_tokens: usize = 12;

    var gen_timer = try std.time.Timer.start();
    for (0..max_new_tokens) |_| {
        const token_str = tok.decode(current_token);
        // Clean display of SentencePiece space
        for (token_str) |b| {
            if (b == 0x81 and token_str.len >= 3) {
                // If it's part of utf-8   (\xe2\x96\x81)
            }
        }
        try stdout.print("{s}", .{token_str});

        if (current_token == tok.eos_token_id) break;

        m.forwardToken(&cache, &scratch, current_token, pos, &thread_pool);
        current_token = kernels.sampleArgmax(scratch.logits);
        pos += 1;
    }

    const elapsed_gen_ms = @as(f64, @floatFromInt(gen_timer.read())) / 1_000_000.0;
    try stdout.print("\n\nGenerated {d} tokens in {d:.2} ms ({d:.2} tokens/sec on multi-threaded CPU).\n", .{
        max_new_tokens,
        elapsed_gen_ms,
        @as(f64, @floatFromInt(max_new_tokens)) / (elapsed_gen_ms / 1000.0),
    });
}

test {
    _ = @import("safetensors.zig");
    _ = @import("tensor.zig");
    _ = @import("kernels.zig");
    _ = @import("tokenizer.zig");
    _ = @import("model.zig");
}
