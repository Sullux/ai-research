const std = @import("std");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const tokenizer = @import("tokenizer.zig");
const ring_buffer = @import("ring_buffer.zig");
const quant = @import("quant.zig");
const gpu = @import("gpu.zig");
const kernels = @import("kernels.zig");

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

    // Canonical Gemma 4 System Prompt with <|tool> definitions
    const full_prompt =
        "<|turn>system\n" ++
        "You are an autonomous cognitive assistant operating in a continuous streaming runtime.\n" ++
        "<|tool>declaration:recall{description:<|\"|>Search memory archive<|\"|>,parameters:{properties:{query:{type:<|\"|>STRING<|\"|>}},required:[<|\"|>query<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>" ++
        "<|tool>declaration:terminal_write{description:<|\"|>Write shell command<|\"|>,parameters:{properties:{input:{type:<|\"|>STRING<|\"|>}},required:[<|\"|>input<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>" ++
        "<turn|>\n" ++
        "<|turn>user\nHow are you doing today?<turn|>\n" ++
        "<|turn>model\n";

    const tokens = try tok.encode(allocator, full_prompt, true);
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
        cur = gpu.model_dispatch.gpuDispatchForwardToken(&gpu_model, &config, m.layers, scratch.x, if (i == tokens.len - 1) scratch.logits else scratch.logits[0..0], clock, slot_idx, scratch.active_slots[0..active_count]);
        clock += 1;
    }
    const prefill_elapsed = std.time.milliTimestamp() - prefill_start;
    std.debug.print("Prefill {} tokens in {}ms ({d:.1} tok/s)\n", .{ tokens.len, prefill_elapsed, (@as(f32, @floatFromInt(tokens.len)) / @as(f32, @floatFromInt(prefill_elapsed))) * 1000.0 });

    std.debug.print("\nGenerated: ", .{});
    for (0..40) |_| {
        std.debug.print("[{s}]", .{tok.decode(cur)});
        const emb_offset = @as(usize, cur) * H;
        for (scratch.x, m.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;
        const slot_idx = ring.activateSlot(0, clock);
        const active_count = ring.getActiveSlots(0, clock, scratch.active_slots);
        cur = gpu.model_dispatch.gpuDispatchForwardToken(&gpu_model, &config, m.layers, scratch.x, scratch.logits, clock, slot_idx, scratch.active_slots[0..active_count]);
        clock += 1;
        if (cur == tok.eos_token_id or cur == 106) break;
    }
    std.debug.print("\n", .{});
}
