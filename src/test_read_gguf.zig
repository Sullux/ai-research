const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const gguf_path = "/home/charles/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf";
    const file = try std.fs.cwd().openFile(gguf_path, .{});
    defer file.close();

    const reader = file.reader();
    const magic = try reader.readInt(u32, .little);
    const version = try reader.readInt(u32, .little);
    const tensor_count = try reader.readInt(u64, .little);
    const kv_count = try reader.readInt(u64, .little);

    std.debug.print("GGUF Magic: 0x{X}, Version: {d}, Tensors: {d}, KV pairs: {d}\n", .{ magic, version, tensor_count, kv_count });

    // Read some metadata keys
    for (0..kv_count) |_| {
        const key_len = try reader.readInt(u64, .little);
        const key = try alloc.alloc(u8, key_len);
        defer alloc.free(key);
        _ = try reader.readAll(key);

        const val_type = try reader.readInt(u32, .little);
        // Skip or print relevant keys
        if (std.mem.indexOf(u8, key, "template") != null or std.mem.indexOf(u8, key, "quant") != null or std.mem.indexOf(u8, key, "rope") != null or std.mem.indexOf(u8, key, "layer") != null) {
            std.debug.print("  Key: {s} (type {d})\n", .{ key, val_type });
        }

        // Skip value based on type
        switch (val_type) {
            0 => _ = try reader.readByte(), // u8
            1 => _ = try reader.readByte(), // i8
            2 => _ = try reader.readInt(u16, .little),
            3 => _ = try reader.readInt(u16, .little),
            4 => _ = try reader.readInt(u32, .little),
            5 => _ = try reader.readInt(u32, .little),
            6 => _ = try reader.readInt(u32, .little),
            7 => _ = try reader.readByte(), // bool
            8 => {
                const s_len = try reader.readInt(u64, .little);
                try file.seekBy(@intCast(s_len));
            },
            9 => {
                const arr_type = try reader.readInt(u32, .little);
                const arr_len = try reader.readInt(u64, .little);
                if (arr_type == 8) {
                    for (0..arr_len) |_| {
                        const s_len = try reader.readInt(u64, .little);
                        try file.seekBy(@intCast(s_len));
                    }
                } else if (arr_type == 4 or arr_type == 5 or arr_type == 6) {
                    try file.seekBy(@intCast(arr_len * 4));
                } else if (arr_type == 7 or arr_type == 0 or arr_type == 1) {
                    try file.seekBy(@intCast(arr_len));
                } else {
                    try file.seekBy(@intCast(arr_len * 8));
                }
            },
            10 => _ = try reader.readInt(u64, .little),
            11 => _ = try reader.readInt(u64, .little),
            12 => _ = try reader.readInt(u64, .little),
            else => {
                std.debug.print("Unknown val_type {d} for key {s}\n", .{ val_type, key });
                break;
            },
        }
    }
}
