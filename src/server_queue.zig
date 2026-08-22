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
