const std = @import("std");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const tokenizer = @import("tokenizer.zig");
const ring_buffer = @import("ring_buffer.zig");
const quant = @import("quant.zig");
const gpu = @import("gpu.zig");
const kernels = @import("kernels.zig");

fn sampleTopP(logits: []f32, temp: f32, top_p: f32, recent_tokens: []const u32, rep_penalty: f32, rng: std.Random) u32 {
    // 1. Repetition penalty
    if (rep_penalty != 1.0) {
        for (recent_tokens) |tok| {
            if (tok < logits.len) {
                if (logits[tok] > 0.0) {
                    logits[tok] /= rep_penalty;
                } else {
                    logits[tok] *= rep_penalty;
                }
            }
        }
    }

    // 2. Softcapping (30.0)
    for (logits) |*l| {
        l.* = 30.0 * std.math.tanh(l.* / 30.0);
    }

    // 3. Temperature scaling
    if (temp > 0.0) {
        for (logits) |*l| l.* /= temp;
    } else {
        return kernels.sampleArgmax(logits);
    }

    // 4. Softmax
    kernels.softmax(logits);

    // 5. Top-P cumulative thresholding
    const IndexedLogit = struct { id: u32, prob: f32 };
    var top_candidates = std.ArrayList(IndexedLogit).init(std.heap.page_allocator);
    defer top_candidates.deinit();

    for (logits, 0..) |p, i| {
        if (p > 1e-5) {
            top_candidates.append(.{ .id = @intCast(i), .prob = p }) catch break;
        }
    }

    std.mem.sort(IndexedLogit, top_candidates.items, {}, struct {
        fn cmp(_: void, a: IndexedLogit, b: IndexedLogit) bool {
            return a.prob > b.prob;
        }
    }.cmp);

    if (top_candidates.items.len == 0) return 0;

    var cum_p: f32 = 0.0;
    var cutoff: usize = 0;
    for (top_candidates.items, 0..) |cand, i| {
        cum_p += cand.prob;
        cutoff = i + 1;
        if (cum_p >= top_p) break;
    }

    const r = rng.float(f32) * cum_p;
    var acc: f32 = 0.0;
    for (top_candidates.items[0..cutoff]) |cand| {
        acc += cand.prob;
        if (acc >= r) return cand.id;
    }
    return top_candidates.items[0].id;
}

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

    var cur: u32 = 0;
    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));
    var clock: usize = 0;
    const prefill_start = std.time.milliTimestamp();
    for (tokens, 0..) |t, i| {
        const emb_offset = @as(usize, t) * H;
        for (scratch.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
        const slot_idx = ring.activateSlot(0, clock);
        const active_count = ring.getActiveSlots(0, clock, scratch.active_slots);
        _ = gpu.model_dispatch.gpuDispatchForwardToken(&gpu_model, &config, m.layers, scratch.x, if (i == tokens.len - 1) scratch.logits else scratch.logits[0..0], clock, slot_idx, scratch.active_slots[0..active_count]);
        clock += 1;
    }
    const prefill_elapsed = std.time.milliTimestamp() - prefill_start;
    std.debug.print("Prefill {} tokens in {}ms ({d:.1} tok/s)\n", .{ tokens.len, prefill_elapsed, (@as(f32, @floatFromInt(tokens.len)) / @as(f32, @floatFromInt(prefill_elapsed))) * 1000.0 });

    var prng = std.Random.DefaultPrng.init(1337);
    const rng = prng.random();
    var history = std.ArrayList(u32).init(allocator);
    defer history.deinit();

    // Copy logits from host-visible GPU buffer
    const gpu_logits = gpu_model.buf_logits.asSlice(f32)[0..config.vocab_size];
    @memcpy(scratch.logits, gpu_logits);
    cur = sampleTopP(scratch.logits, 0.7, 0.95, history.items, 1.05, rng);

    std.debug.print("\nGenerated: ", .{});
    for (0..100) |_| {
        try history.append(cur);
        std.debug.print("{s}", .{tok.decode(cur)});
        const emb_offset = @as(usize, cur) * H;
        for (scratch.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
        const slot_idx = ring.activateSlot(0, clock);
        const active_count = ring.getActiveSlots(0, clock, scratch.active_slots);
        _ = gpu.model_dispatch.gpuDispatchForwardToken(&gpu_model, &config, m.layers, scratch.x, scratch.logits, clock, slot_idx, scratch.active_slots[0..active_count]);
        clock += 1;
        @memcpy(scratch.logits, gpu_model.buf_logits.asSlice(f32)[0..config.vocab_size]);
        const start_hist = if (history.items.len > 64) history.items.len - 64 else 0;
        cur = sampleTopP(scratch.logits, 0.7, 0.95, history.items[start_hist..], 1.05, rng);
        if (cur == tok.eos_token_id or cur == 106) break;
    }
    std.debug.print("\n", .{});
}
