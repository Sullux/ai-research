const std = @import("std");
pub const safetensors = @import("safetensors.zig");
pub const tensor = @import("tensor.zig");
pub const kernels = @import("kernels.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const model = @import("model.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("=== AI Research Inference Engine (Zig) ===\n", .{});

    const model_path = "../gemma-4-E2B/model.safetensors";
    try stdout.print("Loading model weights from: {s}...\n", .{model_path});

    var timer = try std.time.Timer.start();
    var st = try safetensors.SafeTensors.open(allocator, model_path);
    defer st.deinit();
    const map_time_ns = timer.read();

    try stdout.print("Mapped {d} tensors in {d:.2} ms.\n", .{
        st.tensors.count(),
        @as(f64, @floatFromInt(map_time_ns)) / 1_000_000.0,
    });

    timer.reset();
    const config = model.ModelConfig{
        .vocab_size = 262144,
        .hidden_size = 1536,
        .intermediate_size = 6144,
        .num_hidden_layers = 35,
        .num_attention_heads = 8,
        .num_key_value_heads = 1,
        .head_dim = 256,
    };

    var m = try model.Model.loadFromSafeTensors(allocator, &st, config);
    defer m.deinit();
    const bind_time_ns = timer.read();

    try stdout.print("Bound {d} transformer layers in {d:.2} ms!\n", .{
        m.layers.len,
        @as(f64, @floatFromInt(bind_time_ns)) / 1_000_000.0,
    });

    try stdout.print("\nModel Architecture Summary:\n", .{});
    try stdout.print("  Vocab Size:          {d}\n", .{m.config.vocab_size});
    try stdout.print("  Hidden Dimension:    {d}\n", .{m.config.hidden_size});
    try stdout.print("  Intermediate Size:   {d}\n", .{m.config.intermediate_size});
    try stdout.print("  Layers:              {d}\n", .{m.layers.len});
    try stdout.print("  Attention Heads:     {d} (GQA: {d} KV head)\n", .{ m.config.num_attention_heads, m.config.num_key_value_heads });
    try stdout.print("  Head Dimension:      {d}\n", .{m.config.head_dim});
    try stdout.print("  Embed Table Size:    {d} elements ({d} MB)\n", .{
        m.embed_tokens.len,
        (m.embed_tokens.len * @sizeOf(tensor.bf16)) / (1024 * 1024),
    });
}

test {
    _ = @import("safetensors.zig");
    _ = @import("tensor.zig");
    _ = @import("kernels.zig");
    _ = @import("tokenizer.zig");
    _ = @import("model.zig");
}

