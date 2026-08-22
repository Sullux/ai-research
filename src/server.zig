const std = @import("std");
pub const protocol = @import("protocol.zig");
pub const model = @import("model.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const ring_buffer = @import("ring_buffer.zig");
pub const memory = @import("memory.zig");
pub const storage = @import("storage.zig");
pub const hippocampus = @import("hippocampus.zig");
pub const quiescence = @import("quiescence.zig");
pub const gpu = @import("gpu.zig");
pub const server_queue = @import("server_queue.zig");

pub const ServerConfig = struct {
    thinking_budget: usize = 512, temperature: f32 = 1.0, top_p: f32 = 0.95,
    quiescence_threshold: f32 = 0.001, max_tokens: usize = 64,
};

pub const Server = struct {
    allocator: std.mem.Allocator, m: *const model.Model, tok: *const tokenizer.Tokenizer,
    ring: *ring_buffer.DynamicRingBuffer, scratch: *model.ForwardScratch,
    thread_pool: *std.Thread.Pool, gpu_opt: ?*gpu.model_gpu.GpuModelContext,
    archive: ?*memory.DiffArchive, store: ?*storage.PersistentDiffStore,
    hippo: hippocampus.Hippocampus, q_tracker: quiescence.QuiescenceTracker,
    config: ServerConfig = .{}, queue: server_queue.MessageQueue,
    clock: usize = 0, is_aborted: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    in_thinking_channel: bool = false,

    pub fn init(allocator: std.mem.Allocator, m: *const model.Model, tok: *const tokenizer.Tokenizer, ring: *ring_buffer.DynamicRingBuffer, scratch: *model.ForwardScratch, tp: *std.Thread.Pool, gpu_opt: ?*gpu.model_gpu.GpuModelContext, archive: ?*memory.DiffArchive, store: ?*storage.PersistentDiffStore, q_enabled: bool, q_thresh: f32) !Server {
        const hippo = try hippocampus.Hippocampus.init(allocator, m.config.hidden_size, 128, 6000);
        return .{
            .allocator = allocator, .m = m, .tok = tok, .ring = ring, .scratch = scratch,
            .thread_pool = tp, .gpu_opt = gpu_opt, .archive = archive, .store = store,
            .hippo = hippo, .q_tracker = quiescence.QuiescenceTracker.init(.{ .enabled = q_enabled, .threshold = q_thresh }, m.config.num_hidden_layers),
            .queue = server_queue.MessageQueue.init(allocator),
        };
    }

    pub fn deinit(self: *Server) void {
        self.queue.deinit();
        self.hippo.deinit();
    }

    pub fn run(self: *Server, reader: anytype, writer: anytype) !void {
        const ReaderCtx = struct {
            srv: *Server, r: @TypeOf(reader),
            fn loop(ctx: @This()) void {
                while (true) {
                    const hdr = protocol.readHeader(ctx.r) catch break;
                    if (hdr.opcode == protocol.OP_ABORT) {
                        ctx.srv.is_aborted.store(true, .seq_cst);
                        ctx.srv.hippo.markInterrupted();
                        continue;
                    }
                    var p: []u8 = &.{};
                    if (hdr.payload_len > 0) {
                        p = ctx.srv.allocator.alloc(u8, hdr.payload_len) catch break;
                        ctx.r.readNoEof(p) catch { ctx.srv.allocator.free(p); break; };
                    }
                    ctx.srv.queue.push(.{ .hdr = hdr, .payload = p });
                    if (hdr.opcode == protocol.OP_SHUTDOWN) break;
                }
                ctx.srv.queue.close();
            }
        };
        const reader_thread = try std.Thread.spawn(.{}, ReaderCtx.loop, .{ReaderCtx{ .srv = self, .r = reader }});
        defer reader_thread.join();

        while (self.queue.pop()) |msg| {
            defer if (msg.payload.len > 0) self.allocator.free(msg.payload);
            if (!try self.dispatch(msg.hdr, msg.payload, writer)) break;
        }
    }

    fn dispatch(self: *Server, hdr: protocol.Header, p: []const u8, writer: anytype) !bool {
        switch (hdr.opcode) {
            protocol.OP_STREAM_INPUT => try self.handleStreamInput(hdr.msg_id, p, writer),
            protocol.OP_MEM_QUERY => try self.handleMemQuery(hdr.msg_id, p, writer),
            protocol.OP_SET_CONFIG => self.handleSetConfig(p),
            protocol.OP_MEM_COMMIT => { if (self.archive) |a| _ = self.hippo.commit(a, self.ring, self.store); },
            protocol.OP_PING => try protocol.writePong(writer, hdr.msg_id),
            protocol.OP_SHUTDOWN => return false,
            else => try protocol.writeError(writer, hdr.msg_id, "Unsupported opcode"),
        }
        return true;
    }

    fn handleSetConfig(self: *Server, p: []const u8) void {
        if (p.len < 16) return;
        self.config.thinking_budget = std.mem.readInt(u32, p[0..4], .little);
        self.config.temperature = @bitCast(std.mem.readInt(u32, p[4..8], .little));
        self.config.top_p = @bitCast(std.mem.readInt(u32, p[8..12], .little));
        self.config.quiescence_threshold = @bitCast(std.mem.readInt(u32, p[12..16], .little));
        if (p.len >= 20 and std.mem.readInt(u32, p[16..20], .little) > 0) {
            self.config.max_tokens = std.mem.readInt(u32, p[16..20], .little);
        }
    }

    fn handleMemQuery(self: *Server, msg_id: u16, payload: []const u8, writer: anytype) !void {
        if (self.archive == null or payload.len < 24) {
            try protocol.writeMemResponse(writer, msg_id, 0, 0x02, 0, 0, &.{});
            return;
        }
        const top_k: usize = payload[1];
        const query_tokens = self.tok.encode(self.allocator, payload[24..], false) catch {
            try protocol.writeMemResponse(writer, msg_id, 0, 0x01, 0, 0, &.{});
            return;
        };
        defer self.allocator.free(query_tokens);
        const query_vec = try self.allocator.alloc(f32, self.m.config.hidden_size);
        defer self.allocator.free(query_vec);

        if (!model.memory_inject.computeKeywordQueryVector(self.m, query_tokens, query_vec)) {
            try protocol.writeMemResponse(writer, msg_id, 0, 0x01, 0, 0, &.{});
            return;
        }
        var indices: [16]usize = undefined;
        const count = self.archive.?.scan(query_vec, @intCast(self.clock), &indices, @min(top_k, 16));
        var timestamps: [16]u64 = undefined;
        for (0..count) |i| timestamps[i] = self.archive.?.metas[indices[i]].timestamp;
        try protocol.writeMemResponse(writer, msg_id, @intCast(count), 0x00, 0, 0, timestamps[0..count]);
    }

    fn handleStreamInput(self: *Server, msg_id: u16, payload: []const u8, writer: anytype) !void {
        if (payload.len < 8) return;
        self.is_aborted.store(false, .seq_cst);
        const tokens = try self.parseTokens(payload);
        defer self.allocator.free(tokens);

        var cur: u32 = 0;
        for (tokens, 0..) |t, i| {
            if (self.is_aborted.load(.monotonic)) break;
            cur = self.m.forwardToken(self.ring, self.scratch, t, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, i == tokens.len - 1);
            self.clock += 1;
        }
        if (self.is_aborted.load(.monotonic)) {
            try protocol.writeTurnComplete(writer, msg_id, 0, 0, 0.0, protocol.STOP_ABORTED);
            return;
        }
        try self.decodeResponse(msg_id, cur, writer);
    }

    fn parseTokens(self: *Server, p: []const u8) ![]u32 {
        if (p[0] == protocol.MODE_TEXT) return self.tok.encode(self.allocator, p[8..], false);
        const count = std.mem.readInt(u16, p[2..4], .little);
        const slice: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, p[8 .. 8 + count * 4]));
        const copy = try self.allocator.alloc(u32, slice.len);
        @memcpy(copy, slice);
        return copy;
    }

    fn decodeResponse(self: *Server, msg_id: u16, first_token: u32, writer: anytype) !void {
        var cur = first_token;
        const start = std.time.milliTimestamp();
        var gen_count: u32 = 0;
        var reason: u8 = protocol.STOP_END_OF_TURN;

        for (0..self.config.max_tokens) |_| {
            if (self.is_aborted.load(.monotonic)) { reason = protocol.STOP_ABORTED; break; }
            if (cur == self.tok.eos_token_id or cur == 106) break;

            const str = self.tok.decode(cur);
            if (std.mem.indexOf(u8, str, "<|channel>thought") != null) self.in_thinking_channel = true;
            if (std.mem.indexOf(u8, str, "<channel|>") != null) self.in_thinking_channel = false;

            const opcode = if (self.in_thinking_channel) protocol.OP_STREAM_THOUGHT else protocol.OP_STREAM_CONTENT;
            try protocol.writeToken(writer, msg_id, opcode, cur, @intCast(self.clock), 0xFFFFFFFFFFFF, protocol.TOKEN_TYPE_TEXT, str);
            gen_count += 1;

            cur = self.m.forwardToken(self.ring, self.scratch, cur, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, true);
            self.clock += 1;
        }

        const elapsed: u32 = @intCast(@max(1, std.time.milliTimestamp() - start));
        const tok_sec = (@as(f32, @floatFromInt(gen_count)) / @as(f32, @floatFromInt(elapsed))) * 1000.0;
        try protocol.writeTurnComplete(writer, msg_id, gen_count, elapsed, tok_sec, reason);
        if (self.archive) |a| _ = self.hippo.commit(a, self.ring, self.store);
    }
};
