const std = @import("std");
pub const protocol = @import("protocol.zig");

pub const InboundMsg = struct {
    hdr: protocol.Header,
    payload: []u8,
};

pub const MessageQueue = struct {
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    list: std.ArrayList(InboundMsg),
    closed: bool = false,

    pub fn init(allocator: std.mem.Allocator) MessageQueue {
        return .{
            .allocator = allocator,
            .list = std.ArrayList(InboundMsg).init(allocator),
        };
    }

    pub fn deinit(self: *MessageQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (self.list.items) |msg| {
            if (msg.payload.len > 0) self.allocator.free(msg.payload);
        }
        self.list.deinit();
    }

    pub fn push(self: *MessageQueue, msg: InboundMsg) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.list.append(msg) catch return;
        self.cond.signal();
    }

    pub fn pop(self: *MessageQueue) ?InboundMsg {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.list.items.len == 0 and !self.closed) {
            self.cond.wait(&self.mutex);
        }
        if (self.list.items.len == 0) return null;
        return self.list.orderedRemove(0);
    }

    pub fn close(self: *MessageQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.closed = true;
        self.cond.broadcast();
    }
};

pub const OutboundQueue = struct {
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    list: std.ArrayList([]u8),
    closed: bool = false,

    pub fn init(allocator: std.mem.Allocator) OutboundQueue {
        return .{
            .allocator = allocator,
            .list = std.ArrayList([]u8).init(allocator),
        };
    }

    pub fn deinit(self: *OutboundQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (self.list.items) |buf| self.allocator.free(buf);
        self.list.deinit();
    }

    pub fn push(self: *OutboundQueue, buf: []const u8) void {
        const copy = self.allocator.alloc(u8, buf.len) catch return;
        @memcpy(copy, buf);
        self.mutex.lock();
        defer self.mutex.unlock();
        self.list.append(copy) catch {
            self.allocator.free(copy);
            return;
        };
        self.cond.signal();
    }

    pub fn pop(self: *OutboundQueue) ?[]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.list.items.len == 0 and !self.closed) {
            self.cond.wait(&self.mutex);
        }
        if (self.list.items.len == 0) return null;
        return self.list.orderedRemove(0);
    }

    pub fn close(self: *OutboundQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.closed = true;
        self.cond.broadcast();
    }
};

pub const AsyncWriter = struct {
    queue: *OutboundQueue,
    buf: std.ArrayList(u8),

    pub fn init(queue: *OutboundQueue, allocator: std.mem.Allocator) AsyncWriter {
        return .{ .queue = queue, .buf = std.ArrayList(u8).init(allocator) };
    }

    pub fn deinit(self: *AsyncWriter) void {
        self.flush();
        self.buf.deinit();
    }

    pub fn writeAll(self: *AsyncWriter, bytes: []const u8) !void {
        try self.buf.appendSlice(bytes);
    }

    pub fn writeByte(self: *AsyncWriter, b: u8) !void {
        try self.buf.append(b);
    }

    pub fn writeInt(self: *AsyncWriter, comptime T: type, val: T, endian: std.builtin.Endian) !void {
        var bytes: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &bytes, val, endian);
        try self.buf.appendSlice(&bytes);
    }

    pub fn flush(self: *AsyncWriter) void {
        if (self.buf.items.len > 0) {
            self.queue.push(self.buf.items);
            self.buf.clearRetainingCapacity();
        }
    }
};
