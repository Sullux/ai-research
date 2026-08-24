const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const sampler = @import("sampler.zig");
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

    var gpu_model = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .mixed, 0.001);
    defer gpu_model.deinit();

    var kernel_buf: [4096]u8 = undefined;
    const kernel_bytes = try std.fs.cwd().readFile("tui/PROMPT_KERNEL.md", &kernel_buf);
    var full_prompt_buf: [8192]u8 = undefined;
    const prompt = try std.fmt.bufPrint(&full_prompt_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHello, how are you today?<turn|>\n<|turn>model\n", .{kernel_bytes});
    const tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(tokens);

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring.deinit();

    var slots = try allocator.alloc(u32, tokens.len);
    defer allocator.free(slots);
    for (tokens, 0..) |_, i| {
        const c = i;
        slots[i] = @intCast(ring.getSlotIndex(c));
        for (0..config.num_hidden_layers) |l| _ = ring.activateSlot(l, c);
    }

    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    const bp = gpu_model.batch_prefill_ctx.?;
    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, &config, m.layers, tokens, m.embed_tokens, slots, 0, 0, scratch.logits);

    var s = sampler.Sampler.init(1337, 0.0, 0.95);
    const top_tok0 = s.sample(scratch.logits);
    std.debug.print("Prefill top token: {} ('{s}')\n", .{ top_tok0, tok.decode(top_tok0) });

    // Now run serial single-token forward for all 390 tokens and then token 100
    var ring_s = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring_s.deinit();
    var sc_s = try model.ForwardScratch.init(allocator, config);
    defer sc_s.deinit(allocator);

    const H = config.hidden_size;
    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));
    for (tokens, 0..) |t, i| {
        const emb_offset = @as(usize, t) * H;
        for (sc_s.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
        const slot_idx = ring_s.activateSlot(0, i);
        const active_count = ring_s.getActiveSlots(0, i, sc_s.active_slots);
        _ = @import("gpu/model_dispatch.zig").gpuDispatchForwardToken(&gpu_model, &config, m.layers, sc_s.x, sc_s.logits[0..0], i, slot_idx, sc_s.active_slots[0..active_count]);
    }

    // Now forward token 100 in serial
    const emb_100 = @as(usize, 100) * H;
    for (sc_s.x, m.embed_tokens[emb_100 .. emb_100 + H]) |*out, e| out.* = e.toF32() * embed_scale;
    const slot_100 = ring_s.activateSlot(0, tokens.len);
    const active_cnt_100 = ring_s.getActiveSlots(0, tokens.len, sc_s.active_slots);
    _ = @import("gpu/model_dispatch.zig").gpuDispatchForwardToken(&gpu_model, &config, m.layers, sc_s.x, sc_s.logits, tokens.len, slot_100, sc_s.active_slots[0..active_cnt_100]);
    @memcpy(sc_s.logits, gpu_model.buf_logits.asSlice(f32)[0..config.vocab_size]);

    const top_serial_100 = s.sample(sc_s.logits);
    std.debug.print("Serial top token after 100: {} ('{s}')\n", .{ top_serial_100, tok.decode(top_serial_100) });

    var top5_s: [5]struct { id: u32, val: f32 } = undefined;
    for (&top5_s) |*t| t.* = .{ .id = 0, .val = -1e9 };
    for (sc_s.logits, 0..) |val, i| {
        for (0..5) |j| {
            if (val > top5_s[j].val) {
                var k: usize = 4;
                while (k > j) : (k -= 1) top5_s[k] = top5_s[k - 1];
                top5_s[j] = .{ .id = @intCast(i), .val = val };
                break;
            }
        }
    }
    std.debug.print("Top 5 serial logits after token 100:\n", .{});
    for (top5_s) |t| std.debug.print("  token {} ('{s}'): {d:.3}\n", .{ t.id, tok.decode(t.id), t.val });
    var clock: usize = tokens.len;
    _ = m.forwardToken(&ring, &scratch, 100, clock, null, null, null, &gpu_model, false);
    clock += 1;
    _ = m.forwardToken(&ring, &scratch, 3305, clock, null, null, null, &gpu_model, false);
    clock += 1;
    const thought_tok = m.forwardToken(&ring, &scratch, 107, clock, null, null, null, &gpu_model, true);
    clock += 1;

    std.debug.print("\nToken after <|channel>thought\\n: {} ('{s}')\n", .{ thought_tok, tok.decode(thought_tok) });

    var top5: [5]struct { id: u32, val: f32 } = undefined;
    for (&top5) |*t| t.* = .{ .id = 0, .val = -1e9 };
    for (scratch.logits, 0..) |val, i| {
        for (0..5) |j| {
            if (val > top5[j].val) {
                var k: usize = 4;
                while (k > j) : (k -= 1) top5[k] = top5[k - 1];
                top5[j] = .{ .id = @intCast(i), .val = val };
                break;
            }
        }
    }
    std.debug.print("Top 5 logits inside thought channel:\n", .{});
    for (top5) |t| std.debug.print("  token {} ('{s}'): {d:.3}\n", .{ t.id, tok.decode(t.id), t.val });

    // Generate 40 tokens of thought
    std.debug.print("\nGenerated thought text:\n", .{});
    var cur_t = thought_tok;
    for (0..40) |_| {
        std.debug.print("{s}", .{tok.decode(cur_t)});
        if (cur_t == 101 or cur_t == 106) break;
        cur_t = m.forwardToken(&ring, &scratch, cur_t, clock, null, null, null, &gpu_model, true);
        clock += 1;
    }
    std.debug.print("\n", .{});
}
