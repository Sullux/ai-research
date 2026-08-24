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

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);

    var kernel_buf: [4096]u8 = undefined;
    const kernel_bytes = try std.fs.cwd().readFile("tui/PROMPT_KERNEL.md", &kernel_buf);
    var full_prompt_buf: [8192]u8 = undefined;
    const prompt = try std.fmt.bufPrint(&full_prompt_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n<|channel>thought\n", .{kernel_bytes});
    const tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(tokens);

    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring.deinit();

    var slots = try allocator.alloc(u32, tokens.len);
    defer allocator.free(slots);
    for (tokens, 0..) |_, i| {
        const c = i;
        slots[i] = @intCast(ring.getSlotIndex(c));
        for (0..config.num_hidden_layers) |l| _ = ring.activateSlot(l, c);
    }

    std.debug.print("config.vocab_size = {}\n", .{config.vocab_size});
    const bp = gpu_model.batch_prefill_ctx.?;
    const batch_logits = try allocator.alloc(f32, config.vocab_size);
    defer allocator.free(batch_logits);
    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, &config, m.layers, tokens, m.embed_tokens, slots, 0, 0, batch_logits);

    const raw_gpu_logits = gpu_model.buf_logits.asSlice(f32);
    std.debug.print("raw_gpu_logits first 8: {any}\n", .{raw_gpu_logits[0..8]});

    var max_val: f32 = -1e9;
    var min_val: f32 = 1e9;
    for (batch_logits) |v| {
        if (v > max_val) max_val = v;
        if (v < min_val) min_val = v;
    }
    std.debug.print("batch_logits range: min = {d:.3}, max = {d:.3}\n", .{ min_val, max_val });

    var sc = try model.ForwardScratch.init(allocator, config);
    defer sc.deinit(allocator);

    var s = sampler.Sampler.init(1337, 0.7, 0.95);
    var cur = s.sample(batch_logits);
    std.debug.print("Prefill top token: {} ('{s}')\n", .{ cur, tok.decode(cur) });

    std.debug.print("\nGenerated tokens:\n", .{});
    var clock: usize = tokens.len;
    for (0..120) |_| {
        const decoded = tok.decode(cur);
        for (decoded) |b| {
            if (b == 0xe2) continue;
            std.debug.print("{c}", .{b});
        }
        if (cur == 106 or cur == 1) break;
        cur = m.forwardToken(&ring, &sc, cur, clock, null, null, null, &gpu_model, true);
        clock += 1;
        cur = s.sample(sc.logits);
    }
    std.debug.print("\n", .{});
}
