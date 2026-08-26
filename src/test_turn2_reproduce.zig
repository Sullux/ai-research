const std = @import("std");
const model = @import("model/types.zig");
const loader = @import("model/loader.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const quiescence = @import("quiescence.zig");
const tokenizer = @import("tokenizer.zig");
const gpu = @import("gpu.zig");
const sampler_mod = @import("sampler.zig");

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

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(alloc, config.num_hidden_layers, max_kv_dim, 32, 2048, 96);
    defer ring.deinit();

    var q_tracker = quiescence.QuiescenceTracker.init(.{ .threshold = 0.0 }, config.num_hidden_layers);
    var scratch = try model.ForwardScratch.init(alloc, config);
    defer scratch.deinit(alloc);

    var gctx = try gpu.context.GpuContext.init(alloc);
    defer gctx.deinit();

    var gpu_ctx = try gpu.model_gpu.GpuModelContext.init(alloc, &gctx, &m, config, .q4, 0.0);
    defer gpu_ctx.deinit();

    var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = alloc });
    defer pool.deinit();

    var sampler = sampler_mod.Sampler.init(42, 0.7, 0.95);
    sampler.repeat_penalty = 1.1;

    const kernel_file = try std.fs.cwd().openFile("tui/PROMPT_KERNEL.md", .{});
    defer kernel_file.close();
    const ksize = try kernel_file.getEndPos();
    const kbuf = try alloc.alloc(u8, ksize);
    defer alloc.free(kbuf);
    _ = try kernel_file.readAll(kbuf);

    var clock: usize = 0;

    // --- TURN 1 ---
    std.debug.print("================ TURN 1 ================\n", .{});
    var pb1 = std.ArrayList(u8).init(alloc);
    defer pb1.deinit();
    try pb1.appendSlice("<|turn>system\n<|think|>\n");
    try pb1.appendSlice(std.mem.trim(u8, kbuf, " \t\r\n"));
    try pb1.appendSlice("\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n");

    const tokens1 = try tok.encode(alloc, pb1.items, true);
    defer alloc.free(tokens1);

    for (tokens1, 0..) |t, i| {
        _ = m.forwardToken(&ring, &scratch, t, clock, &pool, null, &q_tracker, &gpu_ctx, i + 1 == tokens1.len);
        clock += 1;
    }

    var recent_buf: [64]u32 = undefined;
    var recent_count: usize = 0;

    var cur = sampler.sample(scratch.logits, null);
    for (0..200) |_| {
        if (cur == tok.eos_token_id or cur == 106) {
            _ = m.forwardToken(&ring, &scratch, cur, clock, &pool, null, &q_tracker, &gpu_ctx, false);
            clock += 1;
            break;
        }

        if (recent_count < 64) {
            recent_buf[recent_count] = cur;
            recent_count += 1;
        } else {
            for (0..63) |i| recent_buf[i] = recent_buf[i + 1];
            recent_buf[63] = cur;
        }
        if (cur == 100 or cur == 101) recent_count = 0;

        std.debug.print("{s}", .{tok.decode(cur)});
        _ = m.forwardToken(&ring, &scratch, cur, clock, &pool, null, &q_tracker, &gpu_ctx, true);
        clock += 1;
        cur = sampler.sample(scratch.logits, recent_buf[0..recent_count]);
    }
    std.debug.print("\n\nTurn 1 ended at clock={d}\n", .{clock});

    // --- TURN 2 ---
    std.debug.print("================ TURN 2 ================\n", .{});
    var pb2 = std.ArrayList(u8).init(alloc);
    defer pb2.deinit();
    try pb2.appendSlice("<|turn>user\nWhat tools do you have?<turn|>\n<|turn>model\n");

    const tokens2 = try tok.encode(alloc, pb2.items, false);
    defer alloc.free(tokens2);

    for (tokens2, 0..) |t, i| {
        _ = m.forwardToken(&ring, &scratch, t, clock, &pool, null, &q_tracker, &gpu_ctx, i + 1 == tokens2.len);
        clock += 1;
    }

    recent_count = 0;
    cur = sampler.sample(scratch.logits, null);
    for (0..300) |_| {
        if (cur == tok.eos_token_id or cur == 106) {
            _ = m.forwardToken(&ring, &scratch, cur, clock, &pool, null, &q_tracker, &gpu_ctx, false);
            clock += 1;
            break;
        }

        if (recent_count < 64) {
            recent_buf[recent_count] = cur;
            recent_count += 1;
        } else {
            for (0..63) |i| recent_buf[i] = recent_buf[i + 1];
            recent_buf[63] = cur;
        }
        if (cur == 100 or cur == 101) recent_count = 0;

        std.debug.print("{s}", .{tok.decode(cur)});
        _ = m.forwardToken(&ring, &scratch, cur, clock, &pool, null, &q_tracker, &gpu_ctx, true);
        clock += 1;
        cur = sampler.sample(scratch.logits, recent_buf[0..recent_count]);
    }
    std.debug.print("\n\nTurn 2 ended at clock={d}\n", .{clock});
}
