const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const gguf_path = "/home/charles/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf";
    const file = try std.fs.cwd().openFile(gguf_path, .{});
    defer file.close();

    const reader = file.reader();
    _ = try reader.readInt(u32, .little);
    _ = try reader.readInt(u32, .little);
    _ = try reader.readInt(u64, .little);
    const kv_count = try reader.readInt(u64, .little);

    for (0..kv_count) |_| {
        const key_len = try reader.readInt(u64, .little);
        const key = try alloc.alloc(u8, key_len);
        defer alloc.free(key);
        _ = try reader.readAll(key);

        const val_type = try reader.readInt(u32, .little);
        switch (val_type) {
            4 => {
                const v = try reader.readInt(u32, .little);
                std.debug.print("  {s} = (u32) {d}\n", .{ key, v });
            },
            6 => {
                const v_u = try reader.readInt(u32, .little);
                const v_f: f32 = @bitCast(v_u);
                std.debug.print("  {s} = (f32) {d}\n", .{ key, v_f });
            },
            8 => {
                const s_len = try reader.readInt(u64, .little);
                const s = try alloc.alloc(u8, s_len);
                defer alloc.free(s);
                _ = try reader.readAll(s);
                if (std.mem.indexOf(u8, key, "template") != null or key_len < 30) {
                    std.debug.print("  {s} = (str) \"{s}\"\n", .{ key, s });
                }
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
            else => {
                if (val_type == 0 or val_type == 1 or val_type == 7) _ = try reader.readByte()
                else if (val_type == 2 or val_type == 3) _ = try reader.readInt(u16, .little)
                else if (val_type == 10 or val_type == 11 or val_type == 12) _ = try reader.readInt(u64, .little);
            },
        }
    }
}
