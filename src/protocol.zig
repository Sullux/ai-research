const std = @import("std");

pub const MAGIC: u32 = 0x53554C58; // 'SULX'
pub const VERSION: u16 = 1;

pub const OP_STREAM_INPUT: u16 = 0x0001;
pub const OP_ABORT: u16 = 0x0002;
pub const OP_MEM_QUERY: u16 = 0x0003;
pub const OP_SET_CONFIG: u16 = 0x0004;
pub const OP_TOOL_RETURN: u16 = 0x0005;
pub const OP_MEM_COMMIT: u16 = 0x0006;
pub const OP_PING: u16 = 0x000E;
pub const OP_SHUTDOWN: u16 = 0x000F;

pub const OP_STREAM_CONTENT: u16 = 0x0101;
pub const OP_STREAM_THOUGHT: u16 = 0x0102;
pub const OP_TURN_COMPLETE: u16 = 0x0103;
pub const OP_TOOL_CALL: u16 = 0x0104;
pub const OP_MEM_RESPONSE: u16 = 0x0105;
pub const OP_STATUS: u16 = 0x0106;
pub const OP_PONG: u16 = 0x010E;
pub const OP_ERROR: u16 = 0x01FF;

pub const MODE_TEXT: u8 = 0x00;
pub const MODE_TOKENS: u8 = 0x01;
pub const MODE_SOFT_VECTORS: u8 = 0x02;
pub const MODE_AUDIO_PCM: u8 = 0x03;
pub const MODE_RAW_IMAGE: u8 = 0x04;
pub const MODE_ENCODED_IMAGE: u8 = 0x05;
pub const MODE_VIDEO_FRAME: u8 = 0x06;

pub const TOKEN_TYPE_TEXT: u8 = 0x00;
pub const TOKEN_TYPE_AUDIO: u8 = 0x01;
pub const TOKEN_TYPE_IMAGE: u8 = 0x02;
pub const TOKEN_TYPE_CONTROL: u8 = 0x03;

pub const STOP_END_OF_TURN: u8 = 0x00;
pub const STOP_MAX_TOKENS: u8 = 0x01;
pub const STOP_ABORTED: u8 = 0x02;
pub const STOP_TOOL_CALL: u8 = 0x03;

pub const QUERY_KEYWORDS: u8 = 0x00;
pub const QUERY_FULLTEXT: u8 = 0x01;
pub const QUERY_TEMPORAL: u8 = 0x02;
pub const QUERY_PINNED: u8 = 0x03;

pub const Header = extern struct {
    magic: u32 = MAGIC,
    version: u16 = VERSION,
    msg_id: u16 = 0,
    opcode: u16,
    reserved: u16 = 0,
    payload_len: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Header) == 16);
}

pub fn readHeader(reader: anytype) !Header {
    var hdr: Header = undefined;
    const bytes = try reader.readBytesNoEof(16);
    hdr = @bitCast(bytes);
    if (hdr.magic != MAGIC) return error.InvalidMagic;
    if (hdr.version != VERSION) return error.UnsupportedVersion;
    return hdr;
}

pub fn writeHeader(writer: anytype, hdr: Header) !void {
    const bytes: [16]u8 = @bitCast(hdr);
    try writer.writeAll(&bytes);
}

pub fn writeToken(writer: anytype, msg_id: u16, opcode: u16, token_id: u32, clock: u64, active_mask: u64, token_type: u8, text: []const u8) !void {
    const text_len: u16 = @intCast(@min(text.len, std.math.maxInt(u16)));
    const payload_len = 4 + 8 + 8 + 1 + 1 + 2 + @as(u32, text_len);
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = opcode, .payload_len = payload_len });
    try writer.writeInt(u32, token_id, .little);
    try writer.writeInt(u64, clock, .little);
    try writer.writeInt(u64, active_mask, .little);
    try writer.writeByte(token_type);
    try writer.writeByte(0); // reserved
    try writer.writeInt(u16, text_len, .little);
    if (text_len > 0) try writer.writeAll(text[0..text_len]);
}

pub fn writeTurnComplete(writer: anytype, msg_id: u16, tokens: u32, elapsed_ms: u32, tok_sec: f32, reason: u8) !void {
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_TURN_COMPLETE, .payload_len = 16 });
    try writer.writeInt(u32, tokens, .little);
    try writer.writeInt(u32, elapsed_ms, .little);
    try writer.writeInt(u32, @bitCast(tok_sec), .little);
    try writer.writeByte(reason);
    try writer.writeAll(&[_]u8{ 0, 0, 0 });
}

pub fn writeMemResponse(writer: anytype, msg_id: u16, count: u8, status: u8, cursor: u16, total_tokens: u32, timestamps: []const u64) !void {
    const ts_len: u32 = @intCast(timestamps.len * 8);
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_MEM_RESPONSE, .payload_len = 8 + ts_len });
    try writer.writeByte(count);
    try writer.writeByte(status);
    try writer.writeInt(u16, cursor, .little);
    try writer.writeInt(u32, total_tokens, .little);
    for (timestamps) |ts| try writer.writeInt(u64, ts, .little);
}

pub fn writeStatus(writer: anytype, msg_id: u16, tok_sec: f32, active_slots: u16, archived_diffs: u16, active_mask: u64, vram_mb: u32) !void {
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_STATUS, .payload_len = 20 });
    try writer.writeInt(u32, @bitCast(tok_sec), .little);
    try writer.writeInt(u16, active_slots, .little);
    try writer.writeInt(u16, archived_diffs, .little);
    try writer.writeInt(u64, active_mask, .little);
    try writer.writeInt(u32, vram_mb, .little);
}

pub fn writeError(writer: anytype, msg_id: u16, msg: []const u8) !void {
    const len: u32 = @intCast(msg.len);
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_ERROR, .payload_len = len });
    if (len > 0) try writer.writeAll(msg);
}

pub fn writePong(writer: anytype, msg_id: u16) !void {
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_PONG, .payload_len = 0 });
}

test "protocol header roundtrip serialization" {
    var buf: [128]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    const hdr = Header{ .msg_id = 42, .opcode = OP_STREAM_INPUT, .payload_len = 100 };
    try writeHeader(stream.writer(), hdr);
    stream.pos = 0;
    const parsed = try readHeader(stream.reader());
    try std.testing.expectEqual(hdr.magic, parsed.magic);
    try std.testing.expectEqual(hdr.version, parsed.version);
    try std.testing.expectEqual(hdr.msg_id, parsed.msg_id);
    try std.testing.expectEqual(hdr.opcode, parsed.opcode);
    try std.testing.expectEqual(hdr.payload_len, parsed.payload_len);
}

test "protocol write token frame" {
    var buf: [128]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try writeToken(stream.writer(), 1, OP_STREAM_CONTENT, 12345, 999, 0xFFFFFFFFFFFF, TOKEN_TYPE_TEXT, "hello");
    stream.pos = 0;
    const parsed_hdr = try readHeader(stream.reader());
    try std.testing.expectEqual(OP_STREAM_CONTENT, parsed_hdr.opcode);
    try std.testing.expectEqual(@as(u32, 24 + 5), parsed_hdr.payload_len);
}
