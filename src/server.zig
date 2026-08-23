const std = @import("std");
const protocol = @import("protocol.zig");
const model = @import("model.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const memory = @import("memory.zig");
const storage = @import("storage.zig");
const quiescence = @import("quiescence.zig");
const hippocampus = @import("hippocampus.zig");
const server_queue = @import("server_queue.zig");
const gpu = @import("gpu.zig");
const sampler = @import("sampler.zig");

pub const Server = struct {
    allocator: std.mem.Allocator,
    m: *const model.Model,
    ring: *ring_buffer.DynamicRingBuffer,
    tok: *const tokenizer.Tokenizer,
    archive: ?*memory.DiffArchive,
    store: ?*storage.PersistentDiffStore,
    scratch: *model.ForwardScratch,
    thread_pool: ?*std.Thread.Pool,
    config: model.ModelConfig,
    max_tokens: usize,
    top_p: f32 = 0.95,
    temp: f32 = 0.7,
    sampler: sampler.Sampler,
    q_tracker: quiescence.QuiescenceTracker,
    hippo: ?hippocampus.Hippocampus = null,
    clock: usize = 0,
    is_aborted: std.atomic.Value(bool),
    in_thinking_channel: bool = false,
    gpu_opt: ?*gpu.model_gpu.GpuModelContext,

    pub fn init(allocator: std.mem.Allocator, m: *const model.Model, ring: *ring_buffer.DynamicRingBuffer, tok: *const tokenizer.Tokenizer, archive: ?*memory.DiffArchive, store: ?*storage.PersistentDiffStore, scratch: *model.ForwardScratch, tp: ?*std.Thread.Pool, config: model.ModelConfig, max_tokens: usize, q_thresh: f32, gpu_opt: ?*gpu.model_gpu.GpuModelContext) !Server {
        return Server{
            .allocator = allocator, .m = m, .ring = ring, .tok = tok, .archive = archive, .store = store, .scratch = scratch, .thread_pool = tp, .config = config, .max_tokens = max_tokens, .sampler = sampler.Sampler.init(1337, 0.7, 0.95, 1.05), .q_tracker = quiescence.QuiescenceTracker.init(.{ .enabled = true, .threshold = q_thresh }, config.num_hidden_layers), .hippo = null, .is_aborted = std.atomic.Value(bool).init(false), .gpu_opt = gpu_opt,
        };
    }

    pub fn deinit(self: *Server) void { _ = self; }

    pub fn run(self: *Server, reader: anytype, writer: anytype) !void {
        var queue = server_queue.MessageQueue.init(self.allocator);
        defer queue.deinit();
        const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
        try protocol.writeStatus(writer, 0, protocol.STATUS_IDLE, 0.0, 0, 0, 0, 0, is_gpu);

        const ReaderThread = struct {
            fn run_reader(q: *server_queue.MessageQueue, r: @TypeOf(reader), aborted: *std.atomic.Value(bool)) void {
                while (true) {
                    const hdr = protocol.readHeader(r) catch break;
                    if (hdr.opcode == protocol.OP_ABORT) { aborted.store(true, .seq_cst); continue; }
                    var p: []u8 = &.{};
                    if (hdr.payload_len > 0) {
                        p = q.allocator.alloc(u8, hdr.payload_len) catch break;
                        r.readNoEof(p) catch { q.allocator.free(p); break; };
                    }
                    q.push(.{ .hdr = hdr, .payload = p });
                    if (hdr.opcode == protocol.OP_SHUTDOWN) { q.close(); break; }
                }
            }
        };

        var thread = try std.Thread.spawn(.{}, ReaderThread.run_reader, .{ &queue, reader, &self.is_aborted });
        defer thread.join();

        while (true) {
            const frame = queue.pop() orelse break;
            const hdr, const p = .{ frame.hdr, frame.payload };
            defer if (p.len > 0) self.allocator.free(p);
            switch (hdr.opcode) {
                protocol.OP_STREAM_INPUT => try self.handleStreamInput(hdr.msg_id, p, writer),
                protocol.OP_SET_CONFIG => self.handleSetConfig(p),
                protocol.OP_MEM_QUERY => try self.handleMemQuery(hdr.msg_id, p, writer),
                protocol.OP_PING => try protocol.writePong(writer, hdr.msg_id),
                protocol.OP_SHUTDOWN => break,
                else => {},
            }
        }
    }

    fn handleSetConfig(self: *Server, p: []const u8) void {
        if (p.len < 20) return;
        self.temp = @bitCast(std.mem.readInt(u32, p[4..8], .little));
        self.top_p = @bitCast(std.mem.readInt(u32, p[8..12], .little));
        self.sampler.temp = self.temp;
        self.sampler.top_p = self.top_p;
        self.q_tracker.config.threshold = @bitCast(std.mem.readInt(u32, p[12..16], .little));
        self.max_tokens = std.mem.readInt(u32, p[16..20], .little);
    }

    fn handleMemQuery(self: *Server, msg_id: u16, p: []const u8, writer: anytype) !void {
        if (self.archive == null or p.len < 4) { try protocol.writeMemResponse(writer, msg_id, 0, 0x01, 0, 0, &.{}); return; }
        const top_k = std.mem.readInt(u16, p[0..2], .little);
        const query_tokens = try self.tok.encode(self.allocator, p[4..], false);
        defer self.allocator.free(query_tokens);
        const query_vec = try self.allocator.alloc(f32, self.config.hidden_size);
        defer self.allocator.free(query_vec);
        if (!model.memory_inject.computeKeywordQueryVector(self.m, query_tokens, query_vec)) {
            try protocol.writeMemResponse(writer, msg_id, 0, 0x01, 0, 0, &.{}); return;
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

        const total_prefill: u32 = @intCast(tokens.len);
        const prefill_start = std.time.milliTimestamp();
        const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
        const active_slots: u16 = @intCast(self.ring.getActiveSlots(0, self.clock, self.scratch.active_slots));
        const diff_count: u16 = if (self.archive) |a| @intCast(a.count) else 0;
        try protocol.writeStatus(writer, msg_id, protocol.STATUS_ENCODING, 0.0, active_slots, diff_count, 0, total_prefill, is_gpu);

        var cur: u32 = 0;
        for (tokens, 0..) |t, i| {
            if (self.is_aborted.load(.monotonic)) break;
            cur = self.m.forwardToken(self.ring, self.scratch, t, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, i == tokens.len - 1);
            self.clock += 1;
            if ((i + 1) % 16 == 0 or i == tokens.len - 1) {
                const elapsed = @max(1, std.time.milliTimestamp() - prefill_start);
                const rate = (@as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(elapsed))) * 1000.0;
                try protocol.writeStatus(writer, msg_id, protocol.STATUS_ENCODING, rate, active_slots, diff_count, @intCast(i + 1), total_prefill, is_gpu);
            }
        }
        if (self.is_aborted.load(.monotonic)) {
            try protocol.writeTurnComplete(writer, msg_id, 0, 0, 0.0, protocol.STOP_ABORTED);
            try protocol.writeStatus(writer, msg_id, protocol.STATUS_IDLE, 0.0, active_slots, diff_count, 0, 0, is_gpu);
            return;
        }
        if (tokens.len > 0) {
            cur = self.sampler.sample(self.scratch.logits);
            self.sampler.recordToken(cur);
        }
        try self.decodeResponse(msg_id, cur, writer, active_slots, diff_count, is_gpu);
    }

    fn parseTokens(self: *Server, p: []const u8) ![]u32 {
        const add_bos = (self.clock == 0);
        if (p[0] == protocol.MODE_TEXT) return self.tok.encode(self.allocator, p[8..], add_bos);
        const count = std.mem.readInt(u16, p[2..4], .little);
        const slice: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, p[8 .. 8 + count * 4]));
        const copy = try self.allocator.alloc(u32, slice.len);
        @memcpy(copy, slice);
        return copy;
    }

    fn decodeResponse(self: *Server, msg_id: u16, first_token: u32, writer: anytype, active_slots: u16, diff_count: u16, is_gpu: u8) !void {
        var cur = first_token;
        const start = std.time.milliTimestamp();
        var gen_count: u32 = 0;
        var reason: u8 = protocol.STOP_END_OF_TURN;

        for (0..self.max_tokens) |_| {
            if (self.is_aborted.load(.monotonic)) { reason = protocol.STOP_ABORTED; break; }
            if (cur == self.tok.eos_token_id or cur == 106) break; // <turn|>
            if (cur == 100) { self.in_thinking_channel = true; cur = self.advanceToken(cur); continue; }
            if (cur == 101) { self.in_thinking_channel = false; cur = self.advanceToken(cur); continue; }
            if (cur == 105 or cur == 98) { cur = self.advanceToken(cur); continue; }

            const str = self.tok.decode(cur);
            if (self.in_thinking_channel and std.mem.eql(u8, str, "thought")) { cur = self.advanceToken(cur); continue; }
            const opcode = if (self.in_thinking_channel) protocol.OP_STREAM_THOUGHT else protocol.OP_STREAM_CONTENT;
            try protocol.writeToken(writer, msg_id, opcode, cur, @intCast(self.clock), 0xFFFFFFFFFFFF, protocol.TOKEN_TYPE_TEXT, str);
            gen_count += 1;

            if (gen_count % 8 == 0) {
                const el = @max(1, std.time.milliTimestamp() - start);
                const rate = (@as(f32, @floatFromInt(gen_count)) / @as(f32, @floatFromInt(el))) * 1000.0;
                try protocol.writeStatus(writer, msg_id, protocol.STATUS_GENERATING, rate, active_slots, diff_count, gen_count, @intCast(self.max_tokens), is_gpu);
            }
            cur = self.advanceToken(cur);
        }

        const elapsed: u32 = @intCast(@max(1, std.time.milliTimestamp() - start));
        const tok_sec = (@as(f32, @floatFromInt(gen_count)) / @as(f32, @floatFromInt(elapsed))) * 1000.0;
        try protocol.writeTurnComplete(writer, msg_id, gen_count, elapsed, tok_sec, reason);
        try protocol.writeStatus(writer, msg_id, protocol.STATUS_IDLE, tok_sec, active_slots, diff_count, gen_count, gen_count, is_gpu);
        if (self.archive != null and self.hippo != null) _ = self.hippo.?.commit(self.archive.?, self.ring, self.store);
    }

    fn advanceToken(self: *Server, cur: u32) u32 {
        _ = self.m.forwardToken(self.ring, self.scratch, cur, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, true);
        self.clock += 1;
        const next = self.sampler.sample(self.scratch.logits);
        self.sampler.recordToken(next);
        return next;
    }
};
