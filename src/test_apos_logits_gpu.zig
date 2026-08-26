const std = @import("std");
const model = @import("model/types.zig");
const loader = @import("model/loader.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const gpu = @import("gpu.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const model_dir = "../gemma-4-12B-it-qat-q4_0-unquantized";
    const config = try model.ModelConfig.loadFromJson(alloc, model_dir ++ "/config.json");
    var tok = try tokenizer.Tokenizer.loadFromJson(alloc, model_dir ++ "/tokenizer.json");
    defer tok.deinit();

    var st = try safetensors.SafeTensors.openDir(alloc, model_dir);
    defer st.deinit();

    var m = try loader.Model.loadFromSafeTensors(alloc, &st, config);
    defer m.deinit();

    var gctx = try gpu.context.GpuContext.init(alloc);
    defer gctx.deinit();

    var gpu_ctx = try gpu.model_gpu.GpuModelContext.init(alloc, &gctx, &m, config, .q4, 0.0);
    defer gpu_ctx.deinit();

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(alloc, config.num_hidden_layers, max_kv_dim, 32, 2048, 96);
    defer ring.deinit();

    var scratch = try model.ForwardScratch.init(alloc, config);
    defer scratch.deinit(alloc);

    const prompt_kernel = try std.fs.cwd().readFileAlloc(alloc, "tui/PROMPT_KERNEL.md", 1024 * 1024);
    defer alloc.free(prompt_kernel);

    var pb = std.ArrayList(u8).init(alloc);
    defer pb.deinit();

    try pb.appendSlice("<|turn>system\n<|think|>\n");
    try pb.appendSlice(std.mem.trim(u8, prompt_kernel, " \t\r\n"));
    try pb.appendSlice("\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n");
    try pb.appendSlice("<|channel>thought\nThe user is asking \"How are you doing today?\". This is a standard social greeting. I should respond politely and helpfully, acknowledging my nature as an AI.\n\nPlan:\n1. Acknowledge and respond to the greeting.\n2. State that I'm doing well and ready to assist.\n3. Ask how I can help the user.<channel|>");
    try pb.appendSlice("I'm doing well, thank you for asking! As an AI, I don'");

    const tokens = try tok.encode(alloc, pb.items, true);
    defer alloc.free(tokens);

    std.debug.print("Prompt tokens count: {d}\n", .{tokens.len});
    std.debug.print("Last 3 tokens: {d} ('{s}'), {d} ('{s}'), {d} ('{s}')\n", .{
        tokens[tokens.len - 3], tok.decode(tokens[tokens.len - 3]),
        tokens[tokens.len - 2], tok.decode(tokens[tokens.len - 2]),
        tokens[tokens.len - 1], tok.decode(tokens[tokens.len - 1]),
    });

    const slots = try alloc.alloc(u32, tokens.len);
    defer alloc.free(slots);
    for (slots, 0..) |*s_val, i| {
        s_val.* = @intCast(ring.getSlotIndex(i));
        for (0..config.num_hidden_layers) |l| _ = ring.activateSlot(l, i);
    }

    const logits_batch = try alloc.alloc(f32, config.vocab_size);
    defer alloc.free(logits_batch);

    try gpu.batch_dispatch.gpuDispatchPrefillBatch(
        gpu_ctx.batch_prefill_ctx.?,
        &gpu_ctx,
        &config,
        m.layers,
        tokens,
        m.embed_tokens,
        slots,
        0,
        0,
        logits_batch,
        null,
        null,
    );

    const t_id: u32 = 236745; // 't'
    const s_id: u32 = 236751; // 's'
    const m_id: u32 = 236757; // 'm'
    const or_id: u32 = 504;   // 'or'

    std.debug.print("\nLogits after \"I don'\":\n", .{});
    std.debug.print("  't'  (id {d}): {d:.4}\n", .{ t_id, logits_batch[t_id] });
    std.debug.print("  's'  (id {d}): {d:.4}\n", .{ s_id, logits_batch[s_id] });
    std.debug.print("  'm'  (id {d}): {d:.4}\n", .{ m_id, logits_batch[m_id] });
    std.debug.print("  'or' (id {d}): {d:.4}\n", .{ or_id, logits_batch[or_id] });

    // Top 10 tokens
    var top10: [10]struct { id: u32, val: f32 } = undefined;
    for (&top10) |*item| item.* = .{ .id = 0, .val = -1e9 };
    for (logits_batch, 0..) |v, id| {
        if (id == 0 or id == 258882 or id == 258883) continue;
        for (0..10) |k| {
            if (v > top10[k].val) {
                var r: usize = 9;
                while (r > k) : (r -= 1) top10[r] = top10[r - 1];
                top10[k] = .{ .id = @intCast(id), .val = v };
                break;
            }
        }
    }
    std.debug.print("\nTop 10 Logits:\n", .{});
    for (top10, 0..) |item, rank| {
        std.debug.print("  #{d}: id={d:<8} val={d:<8.4} piece='{s}'\n", .{ rank + 1, item.id, item.val, tok.decode(item.id) });
    }
}
