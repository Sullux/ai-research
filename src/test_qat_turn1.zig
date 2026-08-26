const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const batch_dispatch = @import("gpu/batch_dispatch.zig");
const sampler = @import("sampler.zig");

pub fn quantizeRowQ4_ZeroCentered(dst_words: []u32, src_bf16: []const u16) void {
    const num_blocks = src_bf16.len / 32;
    var b: usize = 0;
    while (b < num_blocks) : (b += 1) {
        const src_blk = src_bf16[b * 32 .. (b + 1) * 32];
        const dst_blk = dst_words[b * 5 .. (b + 1) * 5];

        var amax: f32 = 0.0;
        var vals: [32]f32 = undefined;
        for (src_blk, 0..) |w, i| {
            const v = @as(f32, @bitCast(@as(u32, w) << 16));
            vals[i] = v;
            const abs_v = @abs(v);
            if (abs_v > amax) amax = abs_v;
        }

        const d = amax / 8.0;
        const id = if (d > 0.0) 1.0 / d else 0.0;
        const scale_f16: f16 = @floatCast(d);
        const min_f16: f16 = @floatCast(-8.0 * d);
        const scale_u16: u16 = @bitCast(scale_f16);
        const min_u16: u16 = @bitCast(min_f16);

        dst_blk[0] = @as(u32, scale_u16) | (@as(u32, min_u16) << 16);

        var nibbles: [32]u8 = undefined;
        for (vals, 0..) |v, i| {
            const xi = std.math.clamp(@as(i32, @intFromFloat(std.math.round(v * id + 8.0))), 0, 15);
            nibbles[i] = @intCast(xi);
        }

        for (0..4) |w| {
            var word_val: u32 = 0;
            for (0..8) |n| {
                const nib = @as(u32, nibbles[w * 8 + n]);
                word_val |= (nib << @intCast(n * 4));
            }
            dst_blk[1 + w] = word_val;
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const model_dir = "../gemma-4-12B-it-qat-q4_0-unquantized";
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

    var gpu_model = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .q4, 0.0);
    defer gpu_model.deinit();

    // Requantize gate and up with Zero-Centered Superblock
    for (gpu_model.layers, 0..) |*l_gpu, i| {
        const l_cpu = m.layers[i];
        quantizeRowQ4_ZeroCentered(l_gpu.gate_proj.asSlice(u32), @as([]const u16, @ptrCast(l_cpu.gate_proj)));
        quantizeRowQ4_ZeroCentered(l_gpu.up_proj.asSlice(u32), @as([]const u16, @ptrCast(l_cpu.up_proj)));
    }

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 384, 512, 96);
    defer ring.deinit();

    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    var samp = sampler.Sampler.init(42, 0.0, 1.0);

    var p1_buf: [8192]u8 = undefined;
    const p1 = try std.fmt.bufPrint(&p1_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHow are you today?<turn|>\n<|turn>model\n", .{kernel_bytes});
    const tok1 = try tok.encode(allocator, p1, true);
    defer allocator.free(tok1);

    var sys_len: usize = 0;
    for (tok1, 0..) |t, i| {
        if (t == 106) { sys_len = i + 1; break; }
    }
    ring.setNumAnchors(if (sys_len > 0) sys_len else 384);

    var clock: usize = 0;
    const bp = gpu_model.batch_prefill_ctx.?;
    var slots1 = try allocator.alloc(u32, tok1.len);
    defer allocator.free(slots1);
    for (tok1, 0..) |_, i| {
        const c = clock + i;
        slots1[i] = @intCast(ring.getSlotIndex(c));
        for (0..config.num_hidden_layers) |l| _ = ring.activateSlot(l, c);
    }
    const logits1 = try allocator.alloc(f32, config.vocab_size);
    defer allocator.free(logits1);

    const start_prefill = std.time.milliTimestamp();
    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, &config, m.layers, tok1, m.embed_tokens, slots1, clock, 0, logits1, null, null);
    const prefill_time = std.time.milliTimestamp() - start_prefill;
    clock += tok1.len;

    std.debug.print("Prefill completed in {d} ms ({} tokens)\n", .{ prefill_time, tok1.len });

    var cur = samp.sample(logits1);
    std.debug.print("\n=== Turn 1 Output on QAT Weights ===\n", .{});
    const start_gen = std.time.milliTimestamp();
    var num_tokens: usize = 0;
    for (0..150) |_| {
        if (cur == tok.eos_token_id or cur == 106) {
            _ = m.forwardToken(&ring, &scratch, cur, clock, null, null, null, &gpu_model, false);
            clock += 1;
            break;
        }
        std.debug.print("{s}", .{tok.decode(cur)});
        _ = m.forwardToken(&ring, &scratch, cur, clock, null, null, null, &gpu_model, true);
        clock += 1;
        num_tokens += 1;
        cur = samp.sample(scratch.logits);
    }
    const gen_time = std.time.milliTimestamp() - start_gen;
    std.debug.print("\n\nGenerated {} tokens in {d} ms ({d:.2} tok/s)\n", .{ num_tokens, gen_time, @as(f64, @floatFromInt(num_tokens)) / (@as(f64, @floatFromInt(gen_time)) / 1000.0) });
}
