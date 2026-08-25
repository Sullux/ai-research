const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const batch_dispatch = @import("gpu/batch_dispatch.zig");
const sampler = @import("sampler.zig");

fn testPrompt(
    allocator: std.mem.Allocator,
    gpu_ctx: *context.GpuContext,
    m: *model.Model,
    config: *const model.ModelConfig,
    tok: *tokenizer.Tokenizer,
    kernel: []const u8,
    label: []const u8,
) !void {
    var gpu_model = try model_gpu.GpuModelContext.init(allocator, gpu_ctx, m, config.*, .q8, 0.0);
    defer gpu_model.deinit();

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 384, 512, 96);
    defer ring.deinit();

    var scratch = try model.ForwardScratch.init(allocator, config.*);
    defer scratch.deinit(allocator);

    var samp = sampler.Sampler.init(42, 0.0, 1.0);

    var p1_buf: [8192]u8 = undefined;
    const p1 = try std.fmt.bufPrint(&p1_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n", .{kernel});
    const tokens = try tok.encode(allocator, p1, true);
    defer allocator.free(tokens);

    var sys_len: usize = 0;
    for (tokens, 0..) |t, i| {
        if (t == 106) { sys_len = i + 1; break; }
    }
    ring.setNumAnchors(if (sys_len > 0) sys_len else 384);

    var clock: usize = 0;
    const bp = gpu_model.batch_prefill_ctx.?;
    var slots1 = try allocator.alloc(u32, tokens.len);
    defer allocator.free(slots1);
    for (tokens, 0..) |_, i| {
        const c = clock + i;
        slots1[i] = @intCast(ring.getSlotIndex(c));
        for (0..config.num_hidden_layers) |l| _ = ring.activateSlot(l, c);
    }
    const logits1 = try allocator.alloc(f32, config.vocab_size);
    defer allocator.free(logits1);

    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, config, m.layers, tokens, m.embed_tokens, slots1, clock, 0, logits1, null, null);
    clock += tokens.len;

    var cur = samp.sample(logits1);
    std.debug.print("\n=== {s} ===\n", .{label});
    for (0..100) |_| {
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

    // 1. Text only
    try testPrompt(allocator, &gpu_ctx, &m, &config, &tok, "You are a helpful, respectful and honest AI assistant operating in an AMD Ryzen computing runtime.", "1. Text Only (no tools)");

    // 2. Text + newline + recall tool
    const k2 = "You are a helpful, respectful and honest AI assistant operating in an AMD Ryzen computing runtime.\n<|tool>declaration:recall{description:<|\"|>Search hippocampal memory archive<|\"|>,parameters:{properties:{query:{description:<|\"|>Search query<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>query<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>";
    try testPrompt(allocator, &gpu_ctx, &m, &config, &tok, k2, "2. Text + newline + recall tool");

    // 3. Just tools (no text)
    const k3 = "<|tool>declaration:recall{description:<|\"|>Search hippocampal memory archive<|\"|>,parameters:{properties:{query:{description:<|\"|>Search query<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>query<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>";
    try testPrompt(allocator, &gpu_ctx, &m, &config, &tok, k3, "3. Tools only (no text)");
}
