const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const batch_dispatch = @import("gpu/batch_dispatch.zig");
const model_dispatch = @import("gpu/model_dispatch.zig");

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
    var full_prompt_buf: [8192]u8 = undefined;
    const prompt = try std.fmt.bufPrint(&full_prompt_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n<|channel>thought\n", .{kernel_bytes});
    const tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(tokens);

    std.debug.print("Prompt tokens len: {}\n", .{tokens.len});

    const H = config.hidden_size;
    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));
    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);

    // 1. Run Batched Prefill
    var gpu_model_batch = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .q4, 0.0);
    defer gpu_model_batch.deinit();

    var ring_batch = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 384, 512, 96);
    defer ring_batch.deinit();

    var slots_batch = try allocator.alloc(u32, tokens.len);
    defer allocator.free(slots_batch);
    for (tokens, 0..) |_, i| slots_batch[i] = @intCast(ring_batch.activateSlot(0, i));

    const logits_batch = try allocator.alloc(f32, config.vocab_size);
    defer allocator.free(logits_batch);

    const bp = gpu_model_batch.batch_prefill_ctx.?;
    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model_batch, &config, m.layers, tokens, m.embed_tokens, slots_batch, 0, 0, logits_batch, null, null);

    // 2. Run Serial Forward Passes
    var gpu_model_serial = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .q4, 0.0);
    defer gpu_model_serial.deinit();

    var ring_serial = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 384, 512, 96);
    defer ring_serial.deinit();

    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    for (tokens, 0..) |t, i| {
        const emb_offset = @as(usize, t) * H;
        for (scratch.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
        const slot_idx = ring_serial.activateSlot(0, i);
        const active_count = ring_serial.getActiveSlots(0, i, scratch.active_slots);
        const is_last = (i == tokens.len - 1);
        _ = model_dispatch.gpuDispatchForwardToken(&gpu_model_serial, &config, m.layers, scratch.x, if (is_last) scratch.logits else scratch.logits[0..0], i, slot_idx, scratch.active_slots[0..active_count]);
    }

    // Compare logits
    std.debug.print("\n=== Top 5 Logits: Batched Prefill ===\n", .{});
    var top5_b: [5]struct { id: u32, val: f32 } = undefined;
    for (&top5_b) |*t| t.* = .{ .id = 0, .val = -1e9 };
    for (logits_batch, 0..) |val, id| {
        for (0..5) |j| {
            if (val > top5_b[j].val) {
                var k: usize = 4;
                while (k > j) : (k -= 1) top5_b[k] = top5_b[k - 1];
                top5_b[j] = .{ .id = @intCast(id), .val = val };
                break;
            }
        }
    }
    for (top5_b) |t| std.debug.print("  token {} ('{s}'): {d:.3}\n", .{ t.id, tok.decode(t.id), t.val });

    std.debug.print("\n=== Top 5 Logits: Serial Forward ===\n", .{});
    var top5_s: [5]struct { id: u32, val: f32 } = undefined;
    for (&top5_s) |*t| t.* = .{ .id = 0, .val = -1e9 };
    for (gpu_model_serial.buf_logits.asSlice(f32), 0..) |val, id| {
        for (0..5) |j| {
            if (val > top5_s[j].val) {
                var k: usize = 4;
                while (k > j) : (k -= 1) top5_s[k] = top5_s[k - 1];
                top5_s[j] = .{ .id = @intCast(id), .val = val };
                break;
            }
        }
    }
    for (top5_s) |t| std.debug.print("  token {} ('{s}'): {d:.3}\n", .{ t.id, tok.decode(t.id), t.val });
}
