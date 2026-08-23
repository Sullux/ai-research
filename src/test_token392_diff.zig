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

    var gpu_model = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .q8, 0.001);
    defer gpu_model.deinit();

    const H = config.hidden_size;
    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));

    var kernel_buf: [4096]u8 = undefined;
    const kernel_bytes = try std.fs.cwd().readFile("tui/PROMPT_KERNEL.md", &kernel_buf);
    var full_prompt_buf: [8192]u8 = undefined;
    const prompt = try std.fmt.bufPrint(&full_prompt_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n<|channel>thought\n", .{kernel_bytes});
    const tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(tokens);

    // 1. Run batch prefill on 393 tokens
    var ring1 = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring1.deinit();

    var slots = try allocator.alloc(u32, tokens.len);
    defer allocator.free(slots);
    for (tokens, 0..) |_, i| {
        const c = i;
        slots[i] = @intCast(ring1.getSlotIndex(c));
        for (0..config.num_hidden_layers) |l| _ = ring1.activateSlot(l, c);
    }

    const bp = gpu_model.batch_prefill_ctx.?;
    const batch_logits = try allocator.alloc(f32, config.vocab_size);
    defer allocator.free(batch_logits);
    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, &config, m.layers, tokens, m.embed_tokens, slots, batch_logits);

    // Now forward token 3048 ("The") in batch mode
    var sc1 = try model.ForwardScratch.init(allocator, config);
    defer sc1.deinit(allocator);
    const next_batch_tok = m.forwardToken(&ring1, &sc1, 3048, tokens.len, null, null, null, &gpu_model, true);
    std.debug.print("Next token after 'The' (after BATCH prefill): {} ('{s}')\n", .{ next_batch_tok, tok.decode(next_batch_tok) });

    // 2. Run serial prefill on 393 tokens
    var ring2 = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring2.deinit();
    var sc2 = try model.ForwardScratch.init(allocator, config);
    defer sc2.deinit(allocator);

    for (tokens, 0..) |t, i| {
        const emb_offset = @as(usize, t) * H;
        for (sc2.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
        const slot_idx = ring2.activateSlot(0, i);
        const active_count = ring2.getActiveSlots(0, i, sc2.active_slots);
        _ = @import("gpu/model_dispatch.zig").gpuDispatchForwardToken(&gpu_model, &config, m.layers, sc2.x, sc2.logits[0..0], i, slot_idx, sc2.active_slots[0..active_count]);
    }

    // Now forward token 3048 ("The") in serial mode
    const next_serial_tok = m.forwardToken(&ring2, &sc2, 3048, tokens.len, null, null, null, &gpu_model, true);
    std.debug.print("Next token after 'The' (after SERIAL prefill): {} ('{s}')\n", .{ next_serial_tok, tok.decode(next_serial_tok) });

    var max_logit_diff: f32 = 0.0;
    for (sc1.logits, sc2.logits) |b, s| {
        const diff = @abs(b - s);
        if (diff > max_logit_diff) max_logit_diff = diff;
    }
    std.debug.print("Logits max diff on token after 'The': {d:.6}\n", .{max_logit_diff});
}
