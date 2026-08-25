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

    var kernel_buf: [4096]u8 = undefined;
    const kernel_bytes = try std.fs.cwd().readFile("tui/PROMPT_KERNEL.md", &kernel_buf);

    var gpu_model = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .q8, 0.0);
    defer gpu_model.deinit();

    const prompt_prefix = try std.fmt.allocPrint(allocator, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n<|channel>thought\nThe user is asking \"How are you doing today?\".\nThis is a standard social greeting.\nI should respond politely and informatively, acknowledging my nature as an AI.<channel|>I'm doing well, thank you for asking! I", .{kernel_bytes});
    defer allocator.free(prompt_prefix);

    const prefix_tokens = try tok.encode(allocator, prompt_prefix, true);
    defer allocator.free(prefix_tokens);

    std.debug.print("Prefix length: {} tokens\n", .{prefix_tokens.len});

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 384, 512, 96);
    defer ring.deinit();

    var sys_len: usize = 0;
    for (prefix_tokens, 0..) |t, i| {
        if (t == 106) { sys_len = i + 1; break; }
    }
    ring.setNumAnchors(if (sys_len > 0) sys_len else 384);

    var slots = try allocator.alloc(u32, prefix_tokens.len);
    defer allocator.free(slots);
    for (prefix_tokens, 0..) |_, i| {
        slots[i] = @intCast(ring.getSlotIndex(i));
        for (0..config.num_hidden_layers) |l| _ = ring.activateSlot(l, i);
    }
    const prefill_logits = try allocator.alloc(f32, config.vocab_size);
    defer allocator.free(prefill_logits);

    const bp = gpu_model.batch_prefill_ctx.?;
    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, &config, m.layers, prefix_tokens, m.embed_tokens, slots, 0, 0, prefill_logits, null, null);

    // Now forward token 236789 (apostrophe ''') at clock = prefix_tokens.len
    const apos_token: u32 = 236789;
    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    const clock = prefix_tokens.len;
    _ = m.forwardToken(&ring, &scratch, apos_token, clock, null, null, null, &gpu_model, true);

    std.debug.print("\n=== Logits after feeding apostrophe (GPU Decode): ===\n", .{});
    std.debug.print("Logit for 's' (236751): {d:.3}\n", .{scratch.logits[236751]});
    std.debug.print("Logit for 'm' (236757): {d:.3}\n", .{scratch.logits[236757]});
    std.debug.print("Logit for 'd' (236753): {d:.3}\n", .{scratch.logits[236753]});
    std.debug.print("Logit for 'll' (864): {d:.3}\n", .{scratch.logits[864]});
    std.debug.print("Logit for 've' (560): {d:.3}\n", .{scratch.logits[560]});
    std.debug.print("Logit for 're' (655): {d:.3}\n", .{scratch.logits[655]});
}
