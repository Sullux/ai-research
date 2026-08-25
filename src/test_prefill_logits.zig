const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const batch_dispatch = @import("gpu/batch_dispatch.zig");

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

    var kernel_buf: [4096]u8 = undefined;
    const kernel_bytes = try std.fs.cwd().readFile("tui/PROMPT_KERNEL.md", &kernel_buf);

    var gpu_model = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .q8, 0.0);
    defer gpu_model.deinit();

    var p1_buf: [8192]u8 = undefined;
    const p1 = try std.fmt.bufPrint(&p1_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n<|channel>thought\nThe user is asking \"How are you doing today?\".\nThis is a standard social greeting.\nI should respond politely and informatively, acknowledging my nature as an AI.<channel|>I'm doing well, thank you for asking! I'", .{kernel_bytes});
    const tokens = try tok.encode(allocator, p1, true);
    defer allocator.free(tokens);

    std.debug.print("Total prefix tokens: {}\n", .{tokens.len});
    std.debug.print("Last 3 tokens: {} ('{s}'), {} ('{s}'), {} ('{s}')\n", .{
        tokens[tokens.len - 3], tok.decode(tokens[tokens.len - 3]),
        tokens[tokens.len - 2], tok.decode(tokens[tokens.len - 2]),
        tokens[tokens.len - 1], tok.decode(tokens[tokens.len - 1]),
    });

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 384, 512, 96);
    defer ring.deinit();

    var sys_len: usize = 0;
    for (tokens, 0..) |t, i| {
        if (t == 106) { sys_len = i + 1; break; }
    }
    ring.setNumAnchors(if (sys_len > 0) sys_len else 384);

    const clock: usize = 0;
    const bp = gpu_model.batch_prefill_ctx.?;
    var slots1 = try allocator.alloc(u32, tokens.len);
    defer allocator.free(slots1);
    for (tokens, 0..) |_, i| {
        const c = clock + i;
        slots1[i] = @intCast(ring.getSlotIndex(c));
        for (0..config.num_hidden_layers) |l| _ = ring.activateSlot(l, c);
    }
    const logits1 = try allocator.alloc(f32, config.vocab_size);
    defer allocator.free(logits1);

    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, &config, m.layers, tokens, m.embed_tokens, slots1, clock, 0, logits1, null, null);

    std.debug.print("\n=== Top 10 Candidates after '... thank you for asking! I' (prefix prefill): ===\n", .{});
    var top10: [10]struct { id: u32, val: f32 } = undefined;
    for (&top10) |*t| t.* = .{ .id = 0, .val = -1e9 };
    for (logits1, 0..) |val, id| {
        if (id == 0 or id == 2 or id == 258882 or id == 258883) continue;
        for (0..10) |j| {
            if (val > top10[j].val) {
                var k: usize = 9;
                while (k > j) : (k -= 1) top10[k] = top10[k - 1];
                top10[j] = .{ .id = @intCast(id), .val = val };
                break;
            }
        }
    }
    for (top10, 0..) |t, rank| std.debug.print("  #{d:2}: token {d:6} ('{s}'): {d:.3}\n", .{ rank + 1, t.id, tok.decode(t.id), t.val });
}
