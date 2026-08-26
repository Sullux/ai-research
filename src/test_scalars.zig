const std = @import("std");
const safetensors = @import("safetensors.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const model_dir = "../gemma-4-12B-it-qat-q4_0-unquantized";
    var st = try safetensors.SafeTensors.openDir(allocator, model_dir);
    defer st.deinit();

    std.debug.print("Checking layer scalars...\n", .{});
    for (0..48) |l| {
        var name_buf: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "model.layers.{}.layer_scalar", .{l});
        if (st.get(name)) |t| {
            const val = @as(f32, @bitCast(@as(u32, t.data[0]) | (@as(u32, t.data[1]) << 8) | (@as(u32, t.data[2]) << 16) | (@as(u32, t.data[3]) << 24)));
            std.debug.print("Layer {}: scalar={d:.6}\n", .{ l, val });
        } else {
            // not present
        }
    }

    std.debug.print("Checking embed scale...\n", .{});
    if (st.get("model.embed_tokens.embed_scale")) |t| {
        const val = @as(f32, @bitCast(@as(u32, t.data[0]) | (@as(u32, t.data[1]) << 8) | (@as(u32, t.data[2]) << 16) | (@as(u32, t.data[3]) << 24)));
        std.debug.print("embed_scale={d:.6}\n", .{val});
    } else {
        std.debug.print("No explicit embed_scale tensor found\n", .{});
    }
}
