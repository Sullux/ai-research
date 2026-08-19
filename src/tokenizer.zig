const std = @import("std");

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    id_to_token: [][]const u8,
    token_to_id: std.StringHashMap(u32),
    bos_token_id: u32 = 2,
    eos_token_id: u32 = 1,
    pad_token_id: u32 = 0,

    pub fn loadFromJson(allocator: std.mem.Allocator, path: []const u8) !Tokenizer {
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer file.close();

        const file_size = try file.getEndPos();
        const json_buf = try allocator.alloc(u8, file_size);
        defer allocator.free(json_buf);

        _ = try file.readAll(json_buf);

        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_buf, .{
            .max_value_len = 100 * 1024 * 1024,
        });
        defer parsed.deinit();

        const root = parsed.value;
        const model_val = root.object.get("model") orelse return error.MissingModelField;
        const vocab_val = model_val.object.get("vocab") orelse return error.MissingVocabField;

        const vocab_count = vocab_val.object.count();
        var id_to_tok = try allocator.alloc([]const u8, vocab_count);
        var tok_to_id = std.StringHashMap(u32).init(allocator);

        var it = vocab_val.object.iterator();
        while (it.next()) |entry| {
            const token_str = entry.key_ptr.*;
            const id: u32 = @intCast(entry.value_ptr.integer);
            const owned_str = try allocator.dupe(u8, token_str);

            if (id < vocab_count) {
                id_to_tok[id] = owned_str;
            }
            try tok_to_id.put(owned_str, id);
        }

        return Tokenizer{
            .allocator = allocator,
            .id_to_token = id_to_tok,
            .token_to_id = tok_to_id,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        for (self.id_to_token) |tok| {
            self.allocator.free(tok);
        }
        self.allocator.free(self.id_to_token);
        self.token_to_id.deinit();
    }

    pub fn decode(self: *const Tokenizer, token_id: u32) []const u8 {
        if (token_id < self.id_to_token.len) {
            return self.id_to_token[token_id];
        }
        return "<unk>";
    }

    pub fn encodeToken(self: *const Tokenizer, piece: []const u8) ?u32 {
        return self.token_to_id.get(piece);
    }
};

test "tokenizer basic initialization" {
    const allocator = std.testing.allocator;
    var id_to_tok = try allocator.alloc([]const u8, 3);
    id_to_tok[0] = try allocator.dupe(u8, "<pad>");
    id_to_tok[1] = try allocator.dupe(u8, "<eos>");
    id_to_tok[2] = try allocator.dupe(u8, "<bos>");

    var tok_to_id = std.StringHashMap(u32).init(allocator);
    try tok_to_id.put(id_to_tok[0], 0);
    try tok_to_id.put(id_to_tok[1], 1);
    try tok_to_id.put(id_to_tok[2], 2);

    var tok = Tokenizer{
        .allocator = allocator,
        .id_to_token = id_to_tok,
        .token_to_id = tok_to_id,
    };
    defer tok.deinit();

    try std.testing.expectEqualStrings("<bos>", tok.decode(2));
    try std.testing.expectEqual(@as(?u32, 1), tok.encodeToken("<eos>"));
}
