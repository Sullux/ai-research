const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const model_types = @import("model/types.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const sampler = @import("sampler.zig");
const batch_dispatch = @import("gpu/batch_dispatch.zig");

const loader = @import("model/loader.zig");

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

    var gpu_ctx = try context.GpuContext.init(allocator);
    defer gpu_ctx.deinit();

    var gpu_model = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .mixed, 0.001);
    defer gpu_model.deinit();

    var kernel_buf: [4096]u8 = undefined;
    const kernel_bytes = try std.fs.cwd().readFile("tui/PROMPT_KERNEL.md", &kernel_buf);
    var full_prompt_buf: [8192]u8 = undefined;
    const prompt = try std.fmt.bufPrint(&full_prompt_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHello, how are you today?<turn|>\n<|turn>model\n", .{kernel_bytes});
    const tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(tokens);

    std.debug.print("Tokens len: {}\n", .{tokens.len});
    for (tokens) |t| std.debug.print(" {}: '{s}'", .{ t, tok.decode(t) });
    std.debug.print("\n", .{});

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring1 = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring1.deinit();

    var slots = try allocator.alloc(u32, tokens.len);
    defer allocator.free(slots);
    for (tokens, 0..) |_, i| slots[i] = @intCast(ring1.activateSlot(0, i));

    const batch_logits = try allocator.alloc(f32, config.vocab_size);
    defer allocator.free(batch_logits);

    const bp = gpu_model.batch_prefill_ctx.?;
    const test_tokens = tokens[0..@min(tokens.len, 128)];
    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, &config, m.layers, test_tokens, m.embed_tokens, slots[0..test_tokens.len], 0, 0, batch_logits);

    var s = sampler.Sampler.init(1337, 0.0, 0.95);
    const top_tok = s.sample(batch_logits);
    std.debug.print("Batched prefill top token: {} '{s}'\n", .{ top_tok, tok.decode(top_tok) });

    // Print top 5 logits
    var top5: [5]struct { id: u32, val: f32 } = undefined;
    for (&top5) |*t| t.* = .{ .id = 0, .val = -1e9 };
    for (batch_logits, 0..) |val, i| {
        for (0..5) |j| {
            if (val > top5[j].val) {
                var k: usize = 4;
                while (k > j) : (k -= 1) top5[k] = top5[k - 1];
                top5[j] = .{ .id = @intCast(i), .val = val };
                break;
            }
        }
    }
    std.debug.print("Top 5 logits from batch prefill:\n", .{});
    for (top5) |t| std.debug.print("  token {} ('{s}'): {d:.3}\n", .{ t.id, tok.decode(t.id), t.val });
}
