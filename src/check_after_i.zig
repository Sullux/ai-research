const std = @import("std");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const tokenizer = @import("tokenizer.zig");
const ring_buffer = @import("ring_buffer.zig");
const gpu = @import("gpu.zig");

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

    var gpu_ctx = try gpu.context.GpuContext.init(allocator);
    defer gpu_ctx.deinit();
    var gpu_model = try gpu.model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .q4, 0.001);
    defer gpu_model.deinit();

    const H = config.hidden_size;
    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring.deinit();

    const file = try std.fs.cwd().openFile("tui/PROMPT_KERNEL.md", .{});
    defer file.close();
    const txt = try file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(txt);

    var full_prompt = std.ArrayList(u8).init(allocator);
    defer full_prompt.deinit();
    try full_prompt.appendSlice("<|turn>system\n<|think|>\n");
    try full_prompt.appendSlice(txt);
    try full_prompt.appendSlice("\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n");

    const tokens = try tok.encode(allocator, full_prompt.items, true);
    defer allocator.free(tokens);

    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));
    var clock: usize = 0;
    for (tokens) |t| {
        const emb_offset = @as(usize, t) * H;
        for (scratch.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
        const slot_idx = ring.activateSlot(0, clock);
        const active_count = ring.getActiveSlots(0, clock, scratch.active_slots);
        _ = gpu.model_dispatch.gpuDispatchForwardToken(&gpu_model, &config, m.layers, scratch.x, scratch.logits[0..0], clock, slot_idx, scratch.active_slots[0..active_count]);
        clock += 1;
    }

    const gen_prefix = [_]u32{
        100, 45518, 107, 818, 2430, 563, 10980, 623, 3910, 659, 611, 3490, 3124, 126584, 1174, 563, 496, 4077, 653, 10841, 236772, 136809, 2934, 236761, 1773, 614, 12498, 236764, 564,
    };

    for (gen_prefix, 0..) |t, i| {
        const emb_offset = @as(usize, t) * H;
        for (scratch.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
        const slot_idx = ring.activateSlot(0, clock);
        const active_count = ring.getActiveSlots(0, clock, scratch.active_slots);
        _ = gpu.model_dispatch.gpuDispatchForwardToken(&gpu_model, &config, m.layers, scratch.x, if (i == gen_prefix.len - 1) scratch.logits else scratch.logits[0..0], clock, slot_idx, scratch.active_slots[0..active_count]);
        clock += 1;
    }

    const gpu_logits = gpu_model.buf_logits.asSlice(f32)[0..config.vocab_size];
    @memcpy(scratch.logits, gpu_logits);

    const Indexed = struct { id: u32, logit: f32 };
    var all = try allocator.alloc(Indexed, config.vocab_size);
    defer allocator.free(all);
    for (scratch.logits, 0..) |l, i| all[i] = .{ .id = @intCast(i), .logit = l };
    std.mem.sort(Indexed, all, {}, struct {
        fn cmp(_: void, a: Indexed, b: Indexed) bool { return a.logit > b.logit; }
    }.cmp);

    std.debug.print("Top 10 logits after \"... As an AI, I\":\n", .{});
    for (all[0..10]) |item| {
        std.debug.print("  [{}: '{s}'] -> logit = {d:.4}\n", .{ item.id, tok.decode(item.id), item.logit });
    }
}
