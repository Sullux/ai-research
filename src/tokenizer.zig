const std = @import("std");

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    id_to_token: [][]const u8,
    token_to_id: std.StringHashMap(u32),
    raw_json: []u8,
    parsed: std.json.Parsed(std.json.Value),
    bos_token_id: u32 = 2,
    eos_token_id: u32 = 1,

    pub fn loadFromJson(allocator: std.mem.Allocator, path: []const u8) !Tokenizer {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const file_size = (try file.stat()).size;
        const raw_json = try allocator.alloc(u8, file_size);
        errdefer allocator.free(raw_json);
        _ = try file.readAll(raw_json);

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw_json, .{});
        errdefer parsed.deinit();

        const root = parsed.value;
        const model_obj = root.object.get("model") orelse return error.InvalidTokenizerJson;
        const vocab_obj = model_obj.object.get("vocab") orelse return error.InvalidTokenizerJson;

        var token_to_id = std.StringHashMap(u32).init(allocator);
        try token_to_id.ensureTotalCapacity(@intCast(vocab_obj.object.count()));

        const id_to_token = try allocator.alloc([]const u8, vocab_obj.object.count());
        @memset(id_to_token, "");

        var iter = vocab_obj.object.iterator();
        while (iter.next()) |entry| {
            const token_str = entry.key_ptr.*;
            const id: u32 = @intCast(entry.value_ptr.*.integer);
            if (id < id_to_token.len) {
                id_to_token[id] = token_str;
            }
            token_to_id.putAssumeCapacity(token_str, id);
        }

        return Tokenizer{
            .allocator = allocator,
            .id_to_token = id_to_token,
            .token_to_id = token_to_id,
            .raw_json = raw_json,
            .parsed = parsed,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.token_to_id.deinit();
        self.allocator.free(self.id_to_token);
        self.parsed.deinit();
        self.allocator.free(self.raw_json);
    }

    pub fn decode(self: *const Tokenizer, token_id: u32) []const u8 {
        if (token_id >= self.id_to_token.len) return "";
        return self.id_to_token[token_id];
    }

    /// Encode input text into a list of token IDs
    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, add_bos: bool) ![]u32 {
        var token_ids = std.ArrayList(u32).init(allocator);
        errdefer token_ids.deinit();

        if (add_bos) {
            try token_ids.append(self.bos_token_id);
        }

        if (text.len == 0) {
            return token_ids.toOwnedSlice();
        }

        // 1. Replace spaces with   (U+2581: \xe2\x96\x81) and add leading space prefix if needed
        var norm = std.ArrayList(u8).init(allocator);
        defer norm.deinit();

        if (text[0] != ' ') {
            try norm.appendSlice("\xe2\x96\x81");
        }

        for (text) |byte| {
            if (byte == ' ') {
                try norm.appendSlice("\xe2\x96\x81");
            } else {
                try norm.append(byte);
            }
        }

        const norm_bytes = norm.items;
        var cursor: usize = 0;

        // 2. Greedy longest-matching prefix against vocabulary
        while (cursor < norm_bytes.len) {
            var max_match_len: usize = 0;
            var matched_id: ?u32 = null;

            const max_search = @min(norm_bytes.len, cursor + 64);
            var end = max_search;
            while (end > cursor) : (end -= 1) {
                const candidate = norm_bytes[cursor..end];
                if (self.token_to_id.get(candidate)) |id| {
                    max_match_len = end - cursor;
                    matched_id = id;
                    break;
                }
            }

            if (matched_id) |id| {
                try token_ids.append(id);
                cursor += max_match_len;
            } else {
                // Byte fallback: try <0xNN>
                var hex_buf: [6]u8 = undefined;
                const hex_str = std.fmt.bufPrint(&hex_buf, "<0x{X:0>2}>", .{norm_bytes[cursor]}) catch "<0x00>";
                if (self.token_to_id.get(hex_str)) |id| {
                    try token_ids.append(id);
                }
                cursor += 1;
            }
        }

        return token_ids.toOwnedSlice();
    }
};

test "tokenizer encode and decode" {
    const testing = std.testing;
    const tokenizer_path = "../gemma-4-E2B/tokenizer.json";

    var tok = Tokenizer.loadFromJson(testing.allocator, tokenizer_path) catch return;
    defer tok.deinit();

    const text = "The capital of France";
    const tokens = try tok.encode(testing.allocator, text, true);
    defer testing.allocator.free(tokens);

    try testing.expect(tokens.len >= 4);
    try testing.expectEqual(tokens[0], 2); // BOS
}
