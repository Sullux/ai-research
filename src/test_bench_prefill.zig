const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const batch_dispatch = @import("gpu/batch_dispatch.zig");

fn progressCallback(layer_idx: usize, total: usize, ctx: ?*anyopaque) void {
    const timer_ptr: *i64 = @ptrCast(@alignCast(ctx.?));
    const now = std.time.milliTimestamp();
    const elapsed = now - timer_ptr.*;
    timer_ptr.* = now;
    std.debug.print("Completed layers {}/{} in {d} ms\n", .{ layer_idx, total, elapsed });
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

    const p1 = try std.fmt.allocPrint(allocator, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHow are you today?<turn|>\n<|turn>model\n", .{kernel_bytes});
    defer allocator.free(p1);
    const tokens = try tok.encode(allocator, p1, true);
    defer allocator.free(tokens);

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 384, 512, 96);
    defer ring.deinit();

    var slots = try allocator.alloc(u32, tokens.len);
    defer allocator.free(slots);
    for (tokens, 0..) |_, i| {
        slots[i] = @intCast(ring.getSlotIndex(i));
        for (0..config.num_hidden_layers) |l| _ = ring.activateSlot(l, i);
    }
    const logits = try allocator.alloc(f32, config.vocab_size);
    defer allocator.free(logits);

    const bp = gpu_model.batch_prefill_ctx.?;

    var last_time = std.time.milliTimestamp();
    const start = last_time;
    try batch_dispatch.gpuDispatchPrefillBatch(bp, &gpu_model, &config, m.layers, tokens, m.embed_tokens, slots, 0, 0, logits, progressCallback, &last_time);
    const total_time = std.time.milliTimestamp() - start;

    std.debug.print("Total Prefill Time: {d} ms for {} tokens ({d:.2} tok/s)\n", .{ total_time, tokens.len, @as(f64, @floatFromInt(tokens.len)) / (@as(f64, @floatFromInt(total_time)) / 1000.0) });
}
