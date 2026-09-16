const std = @import("std");
const protocol = @import("../protocol.zig");
const Server = @import("../server.zig").Server;

pub fn handleReadStreamOpen(self: *Server, msg_id: u16, p: []const u8, writer: anytype) !void {
    if (p.len < 12) return;
    const task_id = std.mem.readInt(u16, p[0..2][0..2], .little);
    const file_offset = std.mem.readInt(u64, p[2..10][0..8], .little);
    const path_len = std.mem.readInt(u16, p[10..12][0..2], .little);
    if (12 + path_len > p.len) return;
    const file_path = p[12 .. 12 + path_len];

    const file = std.fs.cwd().openFile(file_path, .{}) catch {
        try protocol.writeReadStreamStatus(writer, msg_id, task_id, protocol.READ_STATUS_ERROR, 0, file_offset);
        writer.flush();
        return;
    };
    defer file.close();

    file.seekTo(file_offset) catch {
        try protocol.writeReadStreamStatus(writer, msg_id, task_id, protocol.READ_STATUS_ERROR, 0, file_offset);
        writer.flush();
        return;
    };

    var buf: [512]u8 = undefined;
    const bytes_read = file.read(&buf) catch {
        try protocol.writeReadStreamStatus(writer, msg_id, task_id, protocol.READ_STATUS_ERROR, 0, file_offset);
        writer.flush();
        return;
    };

    const new_offset = file_offset + bytes_read;
    const is_eof = (bytes_read < buf.len);
    const status: u8 = if (is_eof) protocol.READ_STATUS_EOF else protocol.READ_STATUS_CHUNK;

    if (bytes_read > 0) {
        var chunk_buf: [1024]u8 = undefined;
        const chunk_text = try std.fmt.bufPrint(&chunk_buf, "\n[Stream Task {d} chunk]:\n{s}\n", .{ task_id, buf[0..bytes_read] });
        const tokens = try self.tok.encode(self.allocator, chunk_text, false);
        defer self.allocator.free(tokens);
        if (tokens.len > 0) {
            _ = try self.prefillTokens(msg_id, tokens, writer, false);
        }
    }

    try protocol.writeReadStreamStatus(writer, msg_id, task_id, status, @intCast(bytes_read), new_offset);
    writer.flush();
}

pub fn handleReadStreamClose(self: *Server, p: []const u8) void {
    _ = self;
    _ = p;
}
