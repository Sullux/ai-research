const std = @import("std");
const safetensors = @import("safetensors.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var st = try safetensors.SafeTensors.openDir(allocator, "../gemma-4-12B-it");
    defer st.deinit();

    var has_post_attn = false;
    var has_post_ffn = false;
    var has_pre_ffn = false;
    var has_input_norm = false;

    if (st.get("model.language_model.layers.0.post_attention_layernorm.weight") != null) has_post_attn = true;
    if (st.get("model.language_model.layers.0.post_feedforward_layernorm.weight") != null) has_post_ffn = true;
    if (st.get("model.language_model.layers.0.pre_feedforward_layernorm.weight") != null) has_pre_ffn = true;
    if (st.get("model.language_model.layers.0.input_layernorm.weight") != null) has_input_norm = true;

    std.debug.print("gemma-4-12B-it Layer 0 norms:\n", .{});
    std.debug.print("  input_layernorm: {}\n", .{has_input_norm});
    std.debug.print("  post_attention_layernorm: {}\n", .{has_post_attn});
    std.debug.print("  pre_feedforward_layernorm: {}\n", .{has_pre_ffn});
    std.debug.print("  post_feedforward_layernorm: {}\n", .{has_post_ffn});
}
