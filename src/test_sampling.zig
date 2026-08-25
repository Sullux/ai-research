const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const batch_dispatch = @import("gpu/batch_dispatch.zig");
const sampler = @import("sampler.zig");

fn testSampling(temp: f32, top_p: f32, seed: u64, label: []const u8) !void {
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

    var gpu_model = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .q4, 0.0);
    defer gpu_model.deinit();

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 384, 512, 96);
    defer ring.deinit();

    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    var samp = sampler.Sampler.init(seed, temp, top_p);

    var p1_buf: [8192]u8 = undefined;
    const p1 = try std.fmt.bufPrint(&p1_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n", .{kernel_bytes});
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
    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, &config, m.layers, tok1, m.embed_tokens, slots1, clock, 0, logits1, null, null);
    clock += tok1.len;

    var cur = samp.sample(logits1);
    std.debug.print("\n=== {s} (temp={d:.1}, top_p={d:.2}) ===\n", .{ label, temp, top_p });
    for (0..120) |_| {
        if (cur == tok.eos_token_id or cur == 106) {
            _ = m.forwardToken(&ring, &scratch, cur, clock, null, null, null, &gpu_model, false);
            clock += 1;
            break;
        }
        std.debug.print("{s}", .{tok.decode(cur)});
        _ = m.forwardToken(&ring, &scratch, cur, clock, null, null, null, &gpu_model, true);
        clock += 1;
        cur = samp.sample(scratch.logits);
    }
    std.debug.print("\n", .{});
}

pub fn main() !void {
    try testSampling(0.7, 0.95, 1234, "Canonical Gemma 4 Sampling");
}
