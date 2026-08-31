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

const PrefillProgress = struct {
    w: *server_queue.AsyncWriter, msg_id: u16, slots: u16, diff_count: u16, total_tok: u32, is_gpu: u8, start_time: i64,
    fn cb(layer_idx: usize, total_layers: usize, ctx_ptr: ?*anyopaque) void {
        const ptr: *@This() = @ptrCast(@alignCast(ctx_ptr.?));
        const el = @max(1, std.time.milliTimestamp() - ptr.start_time);
        const tok_prog: u32 = @intCast((layer_idx * ptr.total_tok) / total_layers);
        const tok_sec = (@as(f32, @floatFromInt(tok_prog)) / @as(f32, @floatFromInt(el))) * 1000.0;
        protocol.writeStatus(ptr.w, ptr.msg_id, protocol.STATUS_ENCODING, tok_sec, ptr.slots, ptr.diff_count, tok_prog, ptr.total_tok, ptr.is_gpu) catch {};
        ptr.w.flush();
    }
};

pub const Server = struct {
    allocator: std.mem.Allocator, m: *const model.Model, ring: *ring_buffer.DynamicRingBuffer, tok: *const tokenizer.Tokenizer, archive: ?*memory.DiffArchive, store: ?*storage.PersistentDiffStore, scratch: *model.ForwardScratch, thread_pool: ?*std.Thread.Pool, config: model.ModelConfig, max_tokens: usize, thinking_budget: usize = 512, top_p: f32 = 0.95, temp: f32 = 0.7, repeat_last_n: usize = 64, sampler: sampler.Sampler, q_tracker: quiescence.QuiescenceTracker, hippo: ?hippocampus.Hippocampus = null, clock: usize = 0, is_aborted: std.atomic.Value(bool), in_thinking_channel: bool = false, gpu_opt: ?*gpu.model_gpu.GpuModelContext,

    pub fn init(allocator: std.mem.Allocator, m: *const model.Model, ring: *ring_buffer.DynamicRingBuffer, tok: *const tokenizer.Tokenizer, archive: ?*memory.DiffArchive, store: ?*storage.PersistentDiffStore, scratch: *model.ForwardScratch, tp: ?*std.Thread.Pool, config: model.ModelConfig, max_tokens: usize, q_thresh: f32, gpu_opt: ?*gpu.model_gpu.GpuModelContext) !Server {
        @memset(scratch.x, 0.0); @memset(scratch.logits, 0.0); @memset(ring.k, 0.0); @memset(ring.v, 0.0);
        if (gpu_opt) |g| {
            @memset(g.buf_logits.asSlice(f32), 0.0); @memset(g.buf_x.asSlice(f32), 0.0);
            if (g.batch_prefill_ctx) |bp| { @memset(bp.buf_x.asSlice(f32), 0.0); @memset(bp.buf_normed_x.asSlice(f32), 0.0); }
        }
        var hippo_inst: ?hippocampus.Hippocampus = null;
        if (archive != null or store != null) {
            const kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
            hippo_inst = try hippocampus.Hippocampus.init(allocator, config.hidden_size, 64, 6000, config.num_hidden_layers, kv_dim);
        }
        return .{ .allocator = allocator, .m = m, .ring = ring, .tok = tok, .archive = archive, .store = store, .scratch = scratch, .thread_pool = tp, .config = config, .max_tokens = max_tokens, .thinking_budget = 512, .temp = 0.7, .top_p = 0.95, .repeat_last_n = 64, .sampler = sampler.Sampler.init(@intCast(@max(1, std.time.nanoTimestamp())), 0.7, 0.95), .q_tracker = quiescence.QuiescenceTracker.init(.{ .enabled = q_thresh > 0.0, .threshold = q_thresh }, config.num_hidden_layers), .hippo = hippo_inst, .is_aborted = std.atomic.Value(bool).init(false), .gpu_opt = gpu_opt };
    }
    pub fn deinit(self: *Server) void {
        if (self.hippo) |*h| h.deinit();
    }
    inline fn slots(self: *Server) u16 { return @intCast(self.ring.getActiveSlots(0, self.clock, self.scratch.active_slots)); }

    pub fn run(self: *Server, reader: anytype, writer: anytype) !void {
        var in_queue = server_queue.MessageQueue.init(self.allocator); defer in_queue.deinit();
        var out_queue = try server_queue.OutboundQueue.init(self.allocator); defer out_queue.deinit();

        const WriterThread = struct {
            fn run(q: *server_queue.OutboundQueue, w: @TypeOf(writer)) void {
                var chunk_buf: [16384]u8 = undefined;
                while (true) {
                    const n = q.readChunk(&chunk_buf);
                    if (n == 0) break;
                    w.writeAll(chunk_buf[0..n]) catch break;
                }
            }
        };
        var w_thread = try std.Thread.spawn(.{}, WriterThread.run, .{ &out_queue, writer });
        defer { out_queue.close(); w_thread.join(); }

        var async_writer = server_queue.AsyncWriter.init(&out_queue, self.allocator);
        defer async_writer.deinit();

        try protocol.writeStatus(&async_writer, 0, protocol.STATUS_IDLE, 0.0, 0, 0, 0, 0, if (self.gpu_opt != null) 1 else 0);
        async_writer.flush();

        const ReaderThread = struct {
            fn run(q: *server_queue.MessageQueue, r: @TypeOf(reader), aborted: *std.atomic.Value(bool)) void {
                defer q.close();
                while (true) {
                    const hdr = protocol.readHeader(r) catch break;
                    if (hdr.opcode == protocol.OP_ABORT) { aborted.store(true, .seq_cst); continue; }
                    var p: []u8 = &.{};
                    if (hdr.payload_len > 0) {
                        p = q.allocator.alloc(u8, hdr.payload_len) catch break;
                        r.readNoEof(p) catch { q.allocator.free(p); break; };
                    }
                    q.push(.{ .hdr = hdr, .payload = p });
                    if (hdr.opcode == protocol.OP_SHUTDOWN) break;
                }
            }
        };
        var r_thread = try std.Thread.spawn(.{}, ReaderThread.run, .{ &in_queue, reader, &self.is_aborted });
        defer r_thread.join();

        while (true) {
            const frame = in_queue.pop() orelse break;
            const hdr, const p = .{ frame.hdr, frame.payload }; defer if (p.len > 0) self.allocator.free(p);
            switch (hdr.opcode) {
                protocol.OP_STREAM_INPUT => try self.handleStreamInput(hdr.msg_id, p, &async_writer),
                protocol.OP_SET_CONFIG => self.handleSetConfig(p),
                protocol.OP_MEM_QUERY => try self.handleMemQuery(hdr.msg_id, p, &async_writer),
                protocol.OP_PING => { try protocol.writePong(&async_writer, hdr.msg_id); async_writer.flush(); },
                protocol.OP_SHUTDOWN => break,
                else => {},
            }
        }
    }

    fn handleSetConfig(self: *Server, p: []const u8) void {
        if (p.len < 20) return;
        self.thinking_budget = std.mem.readInt(u32, p[0..4], .little);
        self.temp = @bitCast(std.mem.readInt(u32, p[4..8], .little));
        self.top_p = @bitCast(std.mem.readInt(u32, p[8..12], .little));
        self.sampler.temp = self.temp;
        self.sampler.top_p = self.top_p;
        self.q_tracker.config.threshold = @bitCast(std.mem.readInt(u32, p[12..16], .little));
        self.max_tokens = std.mem.readInt(u32, p[16..20], .little);
        if (p.len >= 28) {
            self.sampler.min_p = @bitCast(std.mem.readInt(u32, p[20..24], .little));
            self.sampler.repeat_penalty = @bitCast(std.mem.readInt(u32, p[24..28], .little));
        }
        if (p.len >= 32) {
            self.repeat_last_n = std.mem.readInt(u32, p[28..32], .little);
        }
        if (p.len >= 40) {
            self.sampler.frequency_penalty = @bitCast(std.mem.readInt(u32, p[32..36], .little));
            self.sampler.presence_penalty = @bitCast(std.mem.readInt(u32, p[36..40], .little));
        }
    }

    fn handleMemQuery(self: *Server, msg_id: u16, p: []const u8, writer: anytype) !void {
        if (self.archive == null or p.len < 4) {
            try protocol.writeMemResponse(writer, msg_id, 0, 0x01, 0, 0, &.{}); writer.flush(); return;
        }
        const top_k = std.mem.readInt(u16, p[0..2], .little);
        const q_tokens = try self.tok.encode(self.allocator, p[4..], false); defer self.allocator.free(q_tokens);
        const q_vec = try self.allocator.alloc(f32, self.config.hidden_size); defer self.allocator.free(q_vec);
        if (!model.memory_inject.computeKeywordQueryVector(self.m, q_tokens, q_vec)) {
            try protocol.writeMemResponse(writer, msg_id, 0, 0x01, 0, 0, &.{}); writer.flush(); return;
        }
        var indices: [16]usize = undefined;
        const count = self.archive.?.scan(q_vec, @intCast(self.clock), &indices, @min(top_k, 16));
        var timestamps: [16]u64 = undefined;
        for (0..count) |i| timestamps[i] = self.archive.?.metas[indices[i]].timestamp;
        try protocol.writeMemResponse(writer, msg_id, @intCast(count), 0x00, 0, 0, timestamps[0..count]);
        writer.flush();
    }

    fn handleStreamInput(self: *Server, msg_id: u16, payload: []const u8, writer: anytype) !void {
        if (payload.len < 8) return;
        self.is_aborted.store(false, .seq_cst);
        const tokens = try self.parseTokens(payload); defer self.allocator.free(tokens);
        if (self.clock == 0 and tokens.len > 0) {
            var sys_len: usize = 0;
            for (tokens, 0..) |t, i| { if (t == 106) { sys_len = i + 1; break; } }
            self.ring.setNumAnchors(if (sys_len > 0) sys_len else @min(tokens.len, 512));
        }
        const total_prefill: u32 = @intCast(tokens.len);
        const prefill_start = std.time.milliTimestamp();
        const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
        const diff_count: u16 = if (self.archive) |a| @intCast(a.count) else 0;
        try protocol.writeStatus(writer, msg_id, protocol.STATUS_ENCODING, 0.0, self.slots(), diff_count, 0, total_prefill, is_gpu);
        writer.flush();

        var cur: u32 = 0;
        if (tokens.len > 1 and self.gpu_opt != null and self.gpu_opt.?.batch_prefill_ctx != null) {
            const gmc, const bp = .{ self.gpu_opt.?, self.gpu_opt.?.batch_prefill_ctx.? };
            var off: usize = 0;
            while (off < tokens.len) {
                if (self.is_aborted.load(.monotonic)) break;
                const chunk = tokens[off..@min(tokens.len, off + bp.max_tokens)];
                const is_last = (off + chunk.len == tokens.len);
                const prev_count = self.ring.getActiveSlots(0, self.clock, self.scratch.active_slots);
                var c_slots = try self.allocator.alloc(u32, prev_count + chunk.len); defer self.allocator.free(c_slots);
                for (self.scratch.active_slots[0..prev_count], 0..) |s, j| c_slots[j] = @intCast(s);
                for (chunk, 0..) |_, i| {
                    const c = self.clock + i; c_slots[prev_count + i] = @intCast(self.ring.getSlotIndex(c));
                    for (0..self.config.num_hidden_layers) |l| _ = self.ring.activateSlot(l, c);
                }
                const l_dst = if (is_last) self.scratch.logits else self.scratch.logits[0..0];
                var p_prog = PrefillProgress{ .w = writer, .msg_id = msg_id, .slots = self.slots(), .diff_count = diff_count, .total_tok = total_prefill, .is_gpu = is_gpu, .start_time = prefill_start };
                try gpu.batch_dispatch.gpuDispatchPrefillBatch(bp, gmc, &self.config, self.m.layers, chunk, self.m.embed_tokens, c_slots, self.clock, prev_count, l_dst, PrefillProgress.cb, &p_prog);
                if (is_last) {
                    const ids = gmc.buf_topk_ids.asSlice(u32)[0..64];
                    const vals = gmc.buf_topk_vals.asSlice(f32)[0..64];
                    for (&self.scratch.topk_candidates, ids, vals) |*dst, id, v| {
                        dst.* = .{ .id = id, .val = v };
                    }
                }
                self.clock += chunk.len; off += chunk.len;
            }
        } else {
            for (tokens, 0..) |t, i| {
                if (self.is_aborted.load(.monotonic)) break;
                cur = self.m.forwardToken(self.ring, self.scratch, t, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, i == tokens.len - 1);
                self.clock += 1;
                if ((i + 1) % 16 == 0 or i == tokens.len - 1) {
                    const el = @max(1, std.time.milliTimestamp() - prefill_start);
                    try protocol.writeStatus(writer, msg_id, protocol.STATUS_ENCODING, (@as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(el))) * 1000.0, self.slots(), diff_count, @intCast(i + 1), total_prefill, is_gpu);
                    writer.flush();
                }
            }
        }
        if (self.is_aborted.load(.monotonic)) {
            try protocol.writeTurnComplete(writer, msg_id, 0, 0, 0.0, protocol.STOP_ABORTED);
            try protocol.writeStatus(writer, msg_id, protocol.STATUS_IDLE, 0.0, self.slots(), diff_count, 0, 0, is_gpu);
            writer.flush(); return;
        }
        if (tokens.len > 0) {
            cur = if (self.gpu_opt != null) self.sampler.sampleTopK(&self.scratch.topk_candidates, null) else self.sampler.sample(self.scratch.logits, null);
            var is_thk = false;
            for (tokens) |t| { if (t == 100) is_thk = true else if (t == 101) is_thk = false; }
            self.in_thinking_channel = is_thk;
        }
        try self.decodeResponse(msg_id, cur, writer, diff_count, is_gpu);
    }

    fn parseTokens(self: *Server, p: []const u8) ![]u32 {
        if (p[0] == protocol.MODE_TEXT) {
            const raw_text = p[8..];
            return self.tok.encode(self.allocator, raw_text, self.clock == 0);
        }
        const count = std.mem.readInt(u16, p[2..4], .little);
        const slice: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, p[8 .. 8 + count * 4]));
        const copy = try self.allocator.alloc(u32, slice.len); @memcpy(copy, slice); return copy;
    }

    fn decodeResponse(self: *Server, msg_id: u16, first_token: u32, writer: anytype, diff_count: u16, is_gpu: u8) !void {
        var cur = first_token;
        var start: i64 = 0;
        var thinking_count: u32 = 0;
        var response_count: u32 = 0;
        var reason: u8 = protocol.STOP_END_OF_TURN;
        const max_recent: usize = @max(64, self.max_tokens + self.thinking_budget);
        const recent_buf = try self.allocator.alloc(u32, max_recent);
        defer self.allocator.free(recent_buf);
        var recent_count: usize = 0;
        self.sampler.suppress_thinking = false;

        while (true) {
            if (self.is_aborted.load(.monotonic)) {
                reason = protocol.STOP_ABORTED;
                if (self.hippo) |*h| h.markInterrupted();
                break;
            }
            if (cur == self.tok.eos_token_id or cur == 106) {
                _ = self.m.forwardToken(self.ring, self.scratch, cur, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, false);
                self.clock += 1; break;
            }

            if (start == 0) start = std.time.milliTimestamp();

            if (recent_count < max_recent) {
                recent_buf[recent_count] = cur;
                recent_count += 1;
            } else {
                std.mem.copyForwards(u32, recent_buf[0 .. max_recent - 1], recent_buf[1..max_recent]);
                recent_buf[max_recent - 1] = cur;
            }

            const w_start = if (recent_count > self.repeat_last_n) recent_count - self.repeat_last_n else 0;
            const window_tokens = recent_buf[w_start..recent_count];

            if (cur == 100) {
                if (self.sampler.suppress_thinking) {
                    cur = self.advanceToken(101, window_tokens);
                    continue;
                }
                self.in_thinking_channel = true;
                cur = self.advanceToken(cur, window_tokens);
                continue;
            }
            if (cur == 101) { self.in_thinking_channel = false; cur = self.advanceToken(cur, window_tokens); continue; }
            if (cur == 105 or cur == 98) { cur = self.advanceToken(cur, window_tokens); continue; }

            if (self.in_thinking_channel) {
                if (thinking_count >= self.thinking_budget) {
                    self.in_thinking_channel = false;
                    self.sampler.suppress_thinking = true;
                    cur = self.advanceToken(101, window_tokens);
                    continue;
                }
                thinking_count += 1;
            } else {
                if (response_count >= self.max_tokens) {
                    reason = protocol.STOP_MAX_TOKENS;
                    break;
                }
                response_count += 1;
            }

            const str = self.tok.decode(cur);
            const opcode = if (self.in_thinking_channel) protocol.OP_STREAM_THOUGHT else protocol.OP_STREAM_CONTENT;
            try protocol.writeToken(writer, msg_id, opcode, cur, @intCast(self.clock), 0xFFFFFFFFFFFF, protocol.TOKEN_TYPE_TEXT, str);
            writer.flush();

            const total_gen = thinking_count + response_count;
            if (total_gen % 8 == 0) {
                const el = @max(1, std.time.milliTimestamp() - start);
                const dividend = if (total_gen > 1) total_gen - 1 else total_gen;
                const total_budget: u32 = @intCast(self.max_tokens + self.thinking_budget);
                try protocol.writeStatus(writer, msg_id, protocol.STATUS_GENERATING, (@as(f32, @floatFromInt(dividend)) / @as(f32, @floatFromInt(el))) * 1000.0, self.slots(), diff_count, total_gen, total_budget, is_gpu);
                writer.flush();
            }
            cur = self.advanceToken(cur, window_tokens);
        }
        const total_gen = thinking_count + response_count;
        const now = std.time.milliTimestamp();
        const elapsed: u32 = @intCast(@max(1, now - (if (start > 0) start else now)));
        const dividend = if (total_gen > 1) total_gen - 1 else total_gen;
        const tok_sec = (@as(f32, @floatFromInt(dividend)) / @as(f32, @floatFromInt(elapsed))) * 1000.0;
        try protocol.writeTurnComplete(writer, msg_id, total_gen, elapsed, tok_sec, reason);
        try protocol.writeStatus(writer, msg_id, protocol.STATUS_IDLE, tok_sec, self.slots(), diff_count, total_gen, total_gen, is_gpu);
        writer.flush();
        if (self.hippo) |*h| {
            const start_clock = if (self.clock >= h.count) self.clock - h.count else 0;
            _ = h.commit(self.archive, self.ring, self.store, start_clock);
        }
    }

    fn advanceToken(self: *Server, cur: u32, recent_tokens: []const u32) u32 {
        const has_penalties = (self.sampler.repeat_penalty != 1.0 or self.sampler.frequency_penalty != 0.0 or self.sampler.presence_penalty != 0.0);
        const needs_logits = (self.sampler.temp > 0.0 or has_penalties or self.gpu_opt == null);
        const next_tok = self.m.forwardToken(self.ring, self.scratch, cur, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, needs_logits);
        self.clock += 1;
        if (self.hippo) |*h| {
            const x_vec = if (self.gpu_opt) |g| g.buf_x.asSlice(f32)[0..self.config.hidden_size] else self.scratch.x[0..self.config.hidden_size];
            const now_ms = std.time.milliTimestamp();
            const slot_idx = self.ring.getSlotIndex(self.clock - 1);
            h.stage(x_vec, @intCast(@max(0, now_ms)), 1.0, @intCast(self.config.num_hidden_layers - 1), cur, slot_idx, now_ms);
            if (h.shouldFlush(now_ms, false)) {
                const start_clock = if (self.clock >= h.count) self.clock - h.count else 0;
                _ = h.commit(self.archive, self.ring, self.store, start_clock);
            }
        }
        if (!needs_logits) return next_tok;
        if (self.gpu_opt != null) {
            return self.sampler.sampleTopK(&self.scratch.topk_candidates, recent_tokens);
        }
        return self.sampler.sample(self.scratch.logits, recent_tokens);
    }
};
