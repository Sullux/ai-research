const std = @import("std");

pub const QuantMode = enum {
    none,
    q8,
    q4,

    pub fn fromString(str: []const u8) QuantMode {
        if (std.mem.eql(u8, str, "q8") or std.mem.eql(u8, str, "q8_0")) return .q8;
        if (std.mem.eql(u8, str, "q4") or std.mem.eql(u8, str, "q4_0")) return .q4;
        return .none;
    }
};

/// Block size for quantization (32 weights per block)
pub const QK: usize = 32;

/// Quantize a row of BF16 weights into Q8_0 packed u32 array
/// Each block of 32 elements takes 9 u32 words (1 float scale d + 8 words of 32 i8 values)
pub fn quantizeRowQ8_0(dst_words: []u32, src_bf16: []const u16) void {
    const num_blocks = src_bf16.len / QK;
    var b: usize = 0;
    while (b < num_blocks) : (b += 1) {
        const src_blk = src_bf16[b * QK .. (b + 1) * QK];
        const dst_blk = dst_words[b * 9 .. (b + 1) * 9];

        var amax: f32 = 0.0;
        var vals: [QK]f32 = undefined;
        for (src_blk, 0..) |w, i| {
            const v = @as(f32, @bitCast(@as(u32, w) << 16));
            vals[i] = v;
            const abs_v = @abs(v);
            if (abs_v > amax) amax = abs_v;
        }

        const d = amax / 127.0;
        const id = if (amax > 0.0) 127.0 / amax else 0.0;
        dst_blk[0] = @bitCast(d);

        var byte_buf: [32]u8 = undefined;
        for (vals, 0..) |v, i| {
            const q = std.math.clamp(@as(i32, @intFromFloat(std.math.round(v * id))), -128, 127);
            byte_buf[i] = @bitCast(@as(i8, @intCast(q)));
        }

        for (0..8) |w| {
            const w_bytes = byte_buf[w * 4 .. (w + 1) * 4];
            dst_blk[1 + w] = std.mem.readInt(u32, w_bytes[0..4], .little);
        }
    }
}

/// Quantize a row of BF16 weights into Q4_0 packed u32 array
/// Each block of 32 elements takes 5 u32 words (1 float scale d + 4 words of 32 nibbles)
pub fn quantizeRowQ4_0(dst_words: []u32, src_bf16: []const u16) void {
    const num_blocks = src_bf16.len / QK;
    var b: usize = 0;
    while (b < num_blocks) : (b += 1) {
        const src_blk = src_bf16[b * QK .. (b + 1) * QK];
        const dst_blk = dst_words[b * 5 .. (b + 1) * 5];

        var amax: f32 = 0.0;
        var vals: [QK]f32 = undefined;
        for (src_blk, 0..) |w, i| {
            const v = @as(f32, @bitCast(@as(u32, w) << 16));
            vals[i] = v;
            const abs_v = @abs(v);
            if (abs_v > amax) amax = abs_v;
        }

        const d = amax / 7.0;
        const id = if (amax > 0.0) 7.0 / amax else 0.0;
        dst_blk[0] = @bitCast(d);

        var nibbles: [32]u8 = undefined;
        for (vals, 0..) |v, i| {
            const q = std.math.clamp(@as(i32, @intFromFloat(std.math.round(v * id))), -8, 7);
            nibbles[i] = @intCast(q + 8); // Offset by +8 into [0..15]
        }

        for (0..4) |w| {
            var word_val: u32 = 0;
            for (0..8) |n| {
                const nib = @as(u32, nibbles[w * 8 + n]);
                word_val |= (nib << @intCast(n * 4));
            }
            dst_blk[1 + w] = word_val;
        }
    }
}

pub fn quantizeMatrix(dst_words: []u32, src_bf16: []const u16, rows: usize, cols: usize, mode: QuantMode) void {
    if (rows == 0 or cols == 0 or src_bf16.len == 0) return;
    const row_bf16_len = cols;
    const row_words_len = getQuantizedRowWords(cols, mode);

    const num_threads: usize = @min(std.Thread.getCpuCount() catch 16, 16);
    if (rows < num_threads * 4) {
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            const src_row = src_bf16[r * row_bf16_len .. (r + 1) * row_bf16_len];
            const dst_row = dst_words[r * row_words_len .. (r + 1) * row_words_len];
            switch (mode) {
                .q8 => quantizeRowQ8_0(dst_row, src_row),
                .q4 => quantizeRowQ4_0(dst_row, src_row),
                .none => unreachable,
            }
        }
        return;
    }

    const Task = struct {
        dst: []u32,
        src: []const u16,
        r_start: usize,
        r_end: usize,
        row_b_len: usize,
        row_w_len: usize,
        m: QuantMode,

        fn run(self: @This()) void {
            var r = self.r_start;
            while (r < self.r_end) : (r += 1) {
                const src_row = self.src[r * self.row_b_len .. (r + 1) * self.row_b_len];
                const dst_row = self.dst[r * self.row_w_len .. (r + 1) * self.row_w_len];
                switch (self.m) {
                    .q8 => quantizeRowQ8_0(dst_row, src_row),
                    .q4 => quantizeRowQ4_0(dst_row, src_row),
                    .none => unreachable,
                }
            }
        }
    };

    var threads: [16]std.Thread = undefined;
    const chunk = (rows + num_threads - 1) / num_threads;
    var spawned: usize = 0;

    for (0..num_threads) |t| {
        const r_start = t * chunk;
        if (r_start >= rows) break;
        const r_end = @min(r_start + chunk, rows);
        const task = Task{
            .dst = dst_words,
            .src = src_bf16,
            .r_start = r_start,
            .r_end = r_end,
            .row_b_len = row_bf16_len,
            .row_w_len = row_words_len,
            .m = mode,
        };
        threads[spawned] = std.Thread.spawn(.{}, Task.run, .{task}) catch {
            task.run();
            continue;
        };
        spawned += 1;
    }

    for (0..spawned) |t| {
        threads[t].join();
    }
}

pub fn getQuantizedRowWords(cols: usize, mode: QuantMode) usize {
    const num_blocks = (cols + QK - 1) / QK;
    return switch (mode) {
        .none => cols / 2,
        .q8 => num_blocks * 9,
        .q4 => num_blocks * 5,
    };
}

pub fn getQuantizedSizeBytes(rows: usize, cols: usize, mode: QuantMode) usize {
    return rows * getQuantizedRowWords(cols, mode) * @sizeOf(u32);
}
