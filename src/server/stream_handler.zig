const std = @import("std");
const protocol = @import("../protocol.zig");
const template_state = @import("template_state.zig");
const Server = @import("../server.zig").Server;

pub fn parseTokens(self: *Server, p: []const u8) ![]u32 {
    if (p[0] == protocol.MODE_TEXT) {
        return self.tok.encode(self.allocator, p[8..], self.clock == 0);
    }
    const count = std.mem.readInt(u16, p[2..4], .little);
    const slice: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, p[8 .. 8 + count * 4]));
    const copy = try self.allocator.alloc(u32, slice.len);
    @memcpy(copy, slice);
    return copy;
}

pub fn handleStreamInput(self: *Server, msg_id: u16, payload: []const u8, writer: anytype) !void {
    if (payload.len < 8) return;
    self.is_aborted.store(false, .seq_cst);
    self.last_yield_token = null;
    self.ring.markBoundary(self.clock, .user, 1.0);
    const tokens = try self.parseTokens(payload);
    defer self.allocator.free(tokens);
    if (self.clock == 0 and tokens.len > 0) {
        var sys_len: usize = 0;
        for (tokens, 0..) |t, i| {
            if (t == template_state.TOK_TURN_CLOSE) {
                sys_len = i + 1;
                break;
            }
        }
        self.ring.setNumAnchors(if (sys_len > 0) sys_len else @min(tokens.len, 512));
    }

    if (self.turn_open and tokens.len > 0 and tokens[0] != template_state.TOK_TURN_CLOSE) {
        _ = self.m.forwardToken(self.ring, self.scratch, template_state.TOK_TURN_CLOSE, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, false);
        self.clock += 1;
        self.ring.markBoundary(self.clock, .turn_end, 1.0);
        self.turn_open = false;
    }

    const cur = try self.prefillTokens(msg_id, tokens, writer, false);
    if (self.is_aborted.load(.monotonic)) return;

    const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
    const diff_count: u16 = if (self.archive) |a| @intCast(a.count) else 0;
    self.sampler.suppress_thinking = false;
    try self.decodeResponse(msg_id, cur, writer, diff_count, is_gpu);
}

pub fn handleResume(self: *Server, msg_id: u16, writer: anytype) !void {
    self.is_aborted.store(false, .seq_cst);
    const last_token = self.last_yield_token orelse {
        try protocol.writeTurnComplete(writer, msg_id, 0, 0, 0.0, protocol.STOP_END_OF_TURN);
        const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
        const diff_count: u16 = if (self.archive) |a| @intCast(a.count) else 0;
        try protocol.writeStatus(writer, msg_id, protocol.STATUS_IDLE, 0.0, self.slots(), diff_count, 0, 0, is_gpu, self.statusFlags());
        writer.flush();
        return;
    };
    self.last_yield_token = null;
    self.sampler.suppress_thinking = !self.in_thinking_channel;

    const cur = self.advanceToken(last_token, &.{});
    const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
    const diff_count: u16 = if (self.archive) |a| @intCast(a.count) else 0;
    try self.decodeResponse(msg_id, cur, writer, diff_count, is_gpu);
}

pub fn handleToolReturn(self: *Server, msg_id: u16, payload: []const u8, writer: anytype) !void {
    if (payload.len < 6) return;
    self.is_aborted.store(false, .seq_cst);
    self.last_yield_token = null;
    self.ring.markBoundary(self.clock, .tool_result, 1.0);
    const name_len = std.mem.readInt(u16, payload[4..6], .little);
    if (payload.len < 6 + name_len) return;
    const tool_name = payload[6 .. 6 + name_len];
    const result_json = payload[6 + name_len ..];

    const formatted = try self.formatGemmaToolResponse(tool_name, result_json);
    defer self.allocator.free(formatted);

    const tokens = try self.tok.encode(self.allocator, formatted, false);
    defer self.allocator.free(tokens);

    const cur = try self.prefillTokens(msg_id, tokens, writer, false);
    if (self.is_aborted.load(.monotonic)) return;

    const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
    const diff_count: u16 = if (self.archive) |a| @intCast(a.count) else 0;
    self.sampler.suppress_thinking = false;
    try self.decodeResponse(msg_id, cur, writer, diff_count, is_gpu);
}
