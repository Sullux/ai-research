const std = @import("std");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const tokenizer = @import("tokenizer.zig");
const ring_buffer = @import("ring_buffer.zig");
const gpu = @import("gpu.zig");
const sampler = @import("sampler.zig");

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
    var gpu_model = try gpu.model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .q8, 0.001);
    defer gpu_model.deinit();

    const H = config.hidden_size;
    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring.deinit();

    var kernel_buf: [4096]u8 = undefined;
    const kernel_bytes = try std.fs.cwd().readFile("tui/PROMPT_KERNEL.md", &kernel_buf);
    var full_prompt_buf: [8192]u8 = undefined;
    const prompt = try std.fmt.bufPrint(&full_prompt_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n<|channel>thought\n", .{kernel_bytes});
    const tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(tokens);

    var cur: u32 = 0;
    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));
    var clock: usize = 0;
    for (tokens, 0..) |t, i| {
        const emb_offset = @as(usize, t) * H;
        for (scratch.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
        const slot_idx = ring.activateSlot(0, clock);
        const active_count = ring.getActiveSlots(0, clock, scratch.active_slots);
        _ = gpu.model_dispatch.gpuDispatchForwardToken(&gpu_model, &config, m.layers, scratch.x, if (i == tokens.len - 1) scratch.logits else scratch.logits[0..0], clock, slot_idx, scratch.active_slots[0..active_count]);
        clock += 1;
    }

    var s = sampler.Sampler.init(1337, 0.0, 0.95, 1.0);
    @memcpy(scratch.logits, gpu_model.buf_logits.asSlice(f32)[0..config.vocab_size]);
    var top5_dbg: [5]struct { id: u32, val: f32 } = undefined;
    for (&top5_dbg) |*t| t.* = .{ .id = 0, .val = -1e9 };
    for (scratch.logits, 0..) |val, i| {
        for (0..5) |j| {
            if (val > top5_dbg[j].val) {
                var k: usize = 4;
                while (k > j) : (k -= 1) top5_dbg[k] = top5_dbg[k - 1];
                top5_dbg[j] = .{ .id = @intCast(i), .val = val };
                break;
            }
        }
    }
    std.debug.print("Raw unpenalized top 5 logits after prompt:\n", .{});
    for (top5_dbg) |t| std.debug.print("  token {} ('{s}'): {d:.3}\n", .{ t.id, tok.decode(t.id), t.val });

    cur = s.sample(scratch.logits);
    s.recordToken(cur);

    std.debug.print("Serial generation first 10 tokens:\n", .{});
    for (0..10) |_| {
        std.debug.print("  token {} ('{s}')\n", .{ cur, tok.decode(cur) });
        const emb_offset = @as(usize, cur) * H;
        for (scratch.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
        const slot_idx = ring.activateSlot(0, clock);
        const active_count = ring.getActiveSlots(0, clock, scratch.active_slots);
        _ = gpu.model_dispatch.gpuDispatchForwardToken(&gpu_model, &config, m.layers, scratch.x, scratch.logits, clock, slot_idx, scratch.active_slots[0..active_count]);
        clock += 1;
        @memcpy(scratch.logits, gpu_model.buf_logits.asSlice(f32)[0..config.vocab_size]);
        cur = s.sample(scratch.logits);
        s.recordToken(cur);
        if (cur == tok.eos_token_id or cur == 106) break;
    }
}
