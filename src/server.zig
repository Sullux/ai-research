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
    w: *server_queue.AsyncWriter, msg_id: u16, slots: u16, diff_count: u16, base_tok: u32, chunk_tok: u32, total_tok: u32, is_gpu: u8, flags: u16, start_time: i64,
    fn cb(layer_idx: usize, total_layers: usize, ctx_ptr: ?*anyopaque) void {
        const ptr: *@This() = @ptrCast(@alignCast(ctx_ptr.?));
        const el = @max(1, std.time.milliTimestamp() - ptr.start_time);
        const chunk_prog: u32 = @intCast((layer_idx * ptr.chunk_tok) / total_layers);
        const tok_prog: u32 = @min(ptr.total_tok, ptr.base_tok + chunk_prog);
        const tok_sec = (@as(f32, @floatFromInt(tok_prog)) / @as(f32, @floatFromInt(el))) * 1000.0;
        protocol.writeStatus(ptr.w, ptr.msg_id, protocol.STATUS_ENCODING, tok_sec, ptr.slots, ptr.diff_count, tok_prog, ptr.total_tok, ptr.is_gpu, ptr.flags) catch {};
        ptr.w.flush();
    }
};

const SyntaxTracker = struct {
    in_code_fence: bool = false,
    in_inline_code: bool = false,
    in_double_quote: bool = false,
    paren_depth: u32 = 0,
    brace_depth: u32 = 0,
    bracket_depth: u32 = 0,

    pub fn reset(self: *SyntaxTracker) void {
        self.in_code_fence = false;
        self.in_inline_code = false;
        self.in_double_quote = false;
        self.paren_depth = 0;
        self.brace_depth = 0;
        self.bracket_depth = 0;
    }

    pub fn ingestChunk(self: *SyntaxTracker, chunk: []const u8) void {
        var i: usize = 0;
        while (i < chunk.len) {
            const ch = chunk[i];
            if (i + 3 <= chunk.len and std.mem.eql(u8, chunk[i .. i + 3], "```")) {
                self.in_code_fence = !self.in_code_fence;
                i += 3;
                continue;
            }
            if (ch == '`' and !self.in_code_fence) {
                self.in_inline_code = !self.in_inline_code;
                i += 1;
                continue;
            }
            if (self.in_code_fence or self.in_inline_code) {
                i += 1;
                continue;
            }

            if (ch == '"') {
                self.in_double_quote = !self.in_double_quote;
            } else if (ch == '(') {
                self.paren_depth += 1;
            } else if (ch == ')' and self.paren_depth > 0) {
                self.paren_depth -= 1;
            } else if (ch == '{') {
                self.brace_depth += 1;
            } else if (ch == '}' and self.brace_depth > 0) {
                self.brace_depth -= 1;
            } else if (ch == '[') {
                self.bracket_depth += 1;
            } else if (ch == ']' and self.bracket_depth > 0) {
                self.bracket_depth -= 1;
            }
            i += 1;
        }
    }

    pub fn isAtRest(self: *const SyntaxTracker) bool {
        return !self.in_code_fence and
            !self.in_inline_code and
            !self.in_double_quote and
            self.paren_depth == 0 and
            self.brace_depth == 0 and
            self.bracket_depth == 0;
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
    inline fn statusFlags(self: *Server) u16 {
        var flags: u16 = 0;
        if (self.ring.isWorkingSetSaturated(0.85, 0.35)) {
            flags |= protocol.STATUS_FLAG_SATURATED;
        }
        return flags;
    }

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

        try protocol.writeStatus(&async_writer, 0, protocol.STATUS_IDLE, 0.0, 0, 0, 0, 0, if (self.gpu_opt != null) 1 else 0, 0);
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
                protocol.OP_SET_SYSTEM => try self.handleSetSystem(hdr.msg_id, p, &async_writer),
                protocol.OP_SET_CONFIG => self.handleSetConfig(p),
                protocol.OP_MEM_QUERY => try self.handleMemQuery(hdr.msg_id, p, &async_writer),
                protocol.OP_MEM_COMMIT => self.handleMemCommit(),
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
        _ = model.memory_inject.primeSubconsciousMemory(self.archive.?, self.ring, self.scratch, q_vec, @intCast(self.clock), self.gpu_opt);
        var timestamps: [16]u64 = undefined;
        for (0..count) |i| timestamps[i] = self.archive.?.metas[indices[i]].timestamp;
        try protocol.writeMemResponse(writer, msg_id, @intCast(count), 0x00, 0, 0, timestamps[0..count]);
        writer.flush();
    }

    fn prefillTokens(self: *Server, msg_id: u16, tokens: []const u32, writer: anytype, is_system_only: bool) !u32 {
        const total_prefill: u32 = @intCast(tokens.len);
        const prefill_start = std.time.milliTimestamp();
        const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
        const diff_count: u16 = if (self.archive) |a| @intCast(a.count) else 0;

        if (self.archive) |a| {
            if (model.memory_inject.computeKeywordQueryVector(self.m, tokens, self.scratch.normed_x)) {
                _ = model.memory_inject.primeSubconsciousMemory(a, self.ring, self.scratch, self.scratch.normed_x, @intCast(self.clock), self.gpu_opt);
            }
        }

        try protocol.writeStatus(writer, msg_id, protocol.STATUS_ENCODING, 0.0, self.slots(), diff_count, 0, total_prefill, is_gpu, self.statusFlags());
        writer.flush();

        var cur: u32 = 0;
        if (tokens.len > 1 and self.gpu_opt != null and self.gpu_opt.?.batch_prefill_ctx != null) {
            const gmc, const bp = .{ self.gpu_opt.?, self.gpu_opt.?.batch_prefill_ctx.? };
            var off: usize = 0;
            while (off < tokens.len) {
                if (self.is_aborted.load(.monotonic)) break;
                const chunk = tokens[off..@min(tokens.len, off + bp.max_tokens)];
                const is_last = (off + chunk.len == tokens.len);
                const prev_count = self.ring.getPrefillPrevSlots(0, self.clock, chunk.len, self.scratch.active_slots);
                var c_slots = try self.allocator.alloc(u32, prev_count + chunk.len); defer self.allocator.free(c_slots);
                for (self.scratch.active_slots[0..prev_count], 0..) |s, j| c_slots[j] = @intCast(s);
                for (chunk, 0..) |_, i| {
                    const c = self.clock + i; c_slots[prev_count + i] = @intCast(self.ring.getSlotIndex(c));
                    for (0..self.config.num_hidden_layers) |l| _ = self.ring.activateSlot(l, c);
                }
                const l_dst = if (is_last and !is_system_only) self.scratch.logits else self.scratch.logits[0..0];
                var p_prog = PrefillProgress{ .w = writer, .msg_id = msg_id, .slots = self.slots(), .diff_count = diff_count, .base_tok = @intCast(off), .chunk_tok = @intCast(chunk.len), .total_tok = total_prefill, .is_gpu = is_gpu, .flags = self.statusFlags(), .start_time = prefill_start };
                try gpu.batch_dispatch.gpuDispatchPrefillBatch(bp, gmc, &self.config, self.m.layers, chunk, self.m.embed_tokens, c_slots, self.clock, prev_count, l_dst, PrefillProgress.cb, &p_prog);
                if (is_last and !is_system_only) {
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
                cur = self.m.forwardToken(self.ring, self.scratch, t, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, !is_system_only and (i == tokens.len - 1));
                self.clock += 1;
                if ((i + 1) % 16 == 0 or i == tokens.len - 1) {
                    const el = @max(1, std.time.milliTimestamp() - prefill_start);
                    try protocol.writeStatus(writer, msg_id, protocol.STATUS_ENCODING, (@as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(el))) * 1000.0, self.slots(), diff_count, @intCast(i + 1), total_prefill, is_gpu, self.statusFlags());
                    writer.flush();
                }
            }
        }

        if (self.is_aborted.load(.monotonic)) {
            try protocol.writeTurnComplete(writer, msg_id, 0, 0, 0.0, protocol.STOP_ABORTED);
            try protocol.writeStatus(writer, msg_id, protocol.STATUS_IDLE, 0.0, self.slots(), diff_count, 0, 0, is_gpu, self.statusFlags());
            writer.flush();
            return 0;
        }

        if (!is_system_only and tokens.len > 0) {
            cur = if (self.gpu_opt != null) self.sampler.sampleTopK(&self.scratch.topk_candidates, null) else self.sampler.sample(self.scratch.logits, null);
            // If the prompt ended inside a thinking channel (<|channel>thought\n), preserve in_thinking_channel
            var in_thought = false;
            var t_idx: usize = tokens.len;
            while (t_idx > 0) {
                t_idx -= 1;
                const tok_id = tokens[t_idx];
                if (tok_id == 101 or tok_id == 106) {
                    break;
                }
                if (tok_id == 100) {
                    in_thought = true;
                    break;
                }
            }
            self.in_thinking_channel = in_thought;
        }
        return cur;
    }

    fn handleSetSystem(self: *Server, msg_id: u16, payload: []const u8, writer: anytype) !void {
        if (payload.len == 0) return;
        self.is_aborted.store(false, .seq_cst);
        self.ring.markTurnBoundary(self.clock);

        // Parse JSON payload or raw text
        var formatted_system: []const u8 = payload;
        var parsed_json: ?std.json.Parsed(std.json.Value) = null;
        defer if (parsed_json) |*p| p.deinit();

        parsed_json = std.json.parseFromSlice(std.json.Value, self.allocator, payload, .{}) catch null;
        var dyn_buf = std.ArrayList(u8).init(self.allocator);
        defer dyn_buf.deinit();

        if (parsed_json) |p| {
            if (p.value == .object) {
                const root = p.value.object;
                try dyn_buf.appendSlice("<|turn>system\n<|think|>\n");
                if (root.get("instructions")) |inst| {
                    if (inst == .string) {
                        try dyn_buf.appendSlice(inst.string);
                        try dyn_buf.appendSlice("\n");
                    }
                }
                if (root.get("tools")) |tools_val| {
                    if (tools_val == .array) {
                        for (tools_val.array.items) |tool_item| {
                            if (tool_item != .object) continue;
                            const t_obj = tool_item.object;
                            const t_name = if (t_obj.get("name")) |n| (if (n == .string) n.string else "") else "";
                            const t_desc = if (t_obj.get("description")) |d| (if (d == .string) d.string else "") else "";
                            try dyn_buf.appendSlice("<|tool>declaration:");
                            try dyn_buf.appendSlice(t_name);
                            try dyn_buf.appendSlice("{description:<|\"|>");
                            try dyn_buf.appendSlice(t_desc);
                            try dyn_buf.appendSlice("<|\"|>");
                            if (t_obj.get("parameters")) |param_val| {
                                if (param_val == .object) {
                                    const p_obj = param_val.object;
                                    try dyn_buf.appendSlice(",parameters:{");
                                    if (p_obj.get("properties")) |props_val| {
                                        if (props_val == .object) {
                                            try dyn_buf.appendSlice("properties:{");
                                            var f_first = false;
                                            // Collect keys and sort them to match Jinja dictsort
                                            var key_list = std.ArrayList([]const u8).init(self.allocator);
                                            defer key_list.deinit();
                                            var it = props_val.object.iterator();
                                            while (it.next()) |entry| try key_list.append(entry.key_ptr.*);
                                            std.mem.sort([]const u8, key_list.items, {}, struct {
                                                fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                                                    return std.mem.order(u8, a, b) == .lt;
                                                }
                                            }.lessThan);

                                            for (key_list.items) |k| {
                                                const v = props_val.object.get(k).?;
                                                if (f_first) try dyn_buf.appendSlice(",");
                                                f_first = true;
                                                try dyn_buf.appendSlice(k);
                                                try dyn_buf.appendSlice(":{");
                                                if (v == .object) {
                                                    const sub = v.object;
                                                    if (sub.get("description")) |sd| {
                                                        if (sd == .string) {
                                                            try dyn_buf.appendSlice("description:<|\"|>");
                                                            try dyn_buf.appendSlice(sd.string);
                                                            try dyn_buf.appendSlice("<|\"|>,");
                                                        }
                                                    }
                                                    const st = if (sub.get("type")) |st_val| (if (st_val == .string) st_val.string else "STRING") else "STRING";
                                                    if (std.ascii.eqlIgnoreCase(st, "array")) {
                                                        try dyn_buf.appendSlice("items:{type:<|\"|>STRING<|\"|>},");
                                                    }
                                                    try dyn_buf.appendSlice("type:<|\"|>");
                                                    var st_up: [32]u8 = undefined;
                                                    const up_len = @min(st.len, st_up.len);
                                                    for (0..up_len) |ui| st_up[ui] = std.ascii.toUpper(st[ui]);
                                                    try dyn_buf.appendSlice(st_up[0..up_len]);
                                                    try dyn_buf.appendSlice("<|\"|>}");
                                                } else {
                                                    try dyn_buf.appendSlice("type:<|\"|>STRING<|\"|>}");
                                                }
                                            }
                                            try dyn_buf.appendSlice("},");
                                        }
                                    }
                                    if (p_obj.get("required")) |req_val| {
                                        if (req_val == .array and req_val.array.items.len > 0) {
                                            try dyn_buf.appendSlice("required:[");
                                            for (req_val.array.items, 0..) |ri, r_idx| {
                                                if (r_idx > 0) try dyn_buf.appendSlice(",");
                                                try dyn_buf.appendSlice("<|\"|>");
                                                if (ri == .string) try dyn_buf.appendSlice(ri.string);
                                                try dyn_buf.appendSlice("<|\"|>");
                                            }
                                            try dyn_buf.appendSlice("],");
                                        }
                                    }
                                    try dyn_buf.appendSlice("type:<|\"|>OBJECT<|\"|>}");
                                }
                            }
                            try dyn_buf.appendSlice("}<tool|>");
                        }
                    }
                }
                try dyn_buf.appendSlice("<turn|>\n");
                formatted_system = dyn_buf.items;
            }
        }

        const tokens = try self.tok.encode(self.allocator, formatted_system, self.clock == 0);
        defer self.allocator.free(tokens);

        if (self.clock == 0 and tokens.len > 0) {
            self.ring.setNumAnchors(tokens.len);
        }

        _ = try self.prefillTokens(msg_id, tokens, writer, true);

        const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
        const diff_count: u16 = if (self.archive) |a| @intCast(a.count) else 0;
        try protocol.writeStatus(writer, msg_id, protocol.STATUS_IDLE, 0.0, self.slots(), diff_count, 0, 0, is_gpu, self.statusFlags());
        writer.flush();
    }

    fn handleStreamInput(self: *Server, msg_id: u16, payload: []const u8, writer: anytype) !void {
        if (payload.len < 8) return;
        self.is_aborted.store(false, .seq_cst);
        self.ring.markTurnBoundary(self.clock);
        const tokens = try self.parseTokens(payload); defer self.allocator.free(tokens);
        if (self.clock == 0 and tokens.len > 0) {
            var sys_len: usize = 0;
            for (tokens, 0..) |t, i| { if (t == 106) { sys_len = i + 1; break; } }
            self.ring.setNumAnchors(if (sys_len > 0) sys_len else @min(tokens.len, 512));
        }

        const cur = try self.prefillTokens(msg_id, tokens, writer, false);
        if (self.is_aborted.load(.monotonic)) return;

        const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
        const diff_count: u16 = if (self.archive) |a| @intCast(a.count) else 0;
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
        var syntax = SyntaxTracker{};

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
                thinking_count = 0;
                // Absorb channel identifier (e.g. "thought\n")
                var chan_tok = self.advanceToken(cur, window_tokens);
                while (chan_tok != 101 and chan_tok != self.tok.eos_token_id) {
                    if (chan_tok == 107 or chan_tok == 108) {
                        // Reached end of channel header line; advance to first reasoning token
                        cur = self.advanceToken(chan_tok, window_tokens);
                        break;
                    }
                    chan_tok = self.advanceToken(chan_tok, window_tokens);
                }
                if (chan_tok == 101 or chan_tok == self.tok.eos_token_id) {
                    cur = chan_tok;
                }
                continue;
            }
            if (cur == 101) { self.in_thinking_channel = false; self.sampler.suppress_critique = false; cur = self.advanceToken(cur, window_tokens); continue; }
            if (cur == 48) {
                // Token 48: <|tool_call>
                // Collect tool invocation until token 49 (<tool_call|>) or turn end
                var call_buf: [4096]u8 = undefined;
                var call_len: usize = 0;
                var next_tok = self.advanceToken(cur, window_tokens);
                while (next_tok != 49 and next_tok != 106 and next_tok != self.tok.eos_token_id) {
                    const dec_str = self.tok.decode(next_tok);
                    if (call_len + dec_str.len < call_buf.len) {
                        @memcpy(call_buf[call_len .. call_len + dec_str.len], dec_str);
                        call_len += dec_str.len;
                    }
                    next_tok = self.advanceToken(next_tok, window_tokens);
                }
                const raw_call = call_buf[0..call_len];
                // Parse "call:tool_name{args...}"
                var tool_name: []const u8 = "";
                var args_json: []const u8 = "{}";
                if (std.mem.indexOf(u8, raw_call, ":")) |c_idx| {
                    const rest = raw_call[c_idx + 1 ..];
                    if (std.mem.indexOf(u8, rest, "{")) |b_idx| {
                        tool_name = std.mem.trim(u8, rest[0..b_idx], " \t\r\n");
                        args_json = std.mem.trim(u8, rest[b_idx..], " \t\r\n");
                    } else {
                        tool_name = std.mem.trim(u8, rest, " \t\r\n");
                    }
                }
                try protocol.writeToolCall(writer, msg_id, 1, tool_name, args_json);
                writer.flush();
                reason = protocol.STOP_TOOL_CALL;
                if (next_tok == 49) {
                    _ = self.m.forwardToken(self.ring, self.scratch, 49, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, false);
                    self.clock += 1;
                }
                break;
            }
            if (cur == 105 or cur == 98) { cur = self.advanceToken(cur, window_tokens); continue; }

            if (self.in_thinking_channel) {
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

            syntax.ingestChunk(str);

            // Continuous Streaming Transduction: Elastic Syntactic Unit Gating
            // Evaluates natural resting boundaries in both response generation and thinking channels.
            // Only yields when syntactically at rest (no unclosed code fences, quotes, parens, brackets, or braces).
            if (syntax.isAtRest()) {
                const is_para_break = (cur == 108 or (str.len > 0 and std.mem.endsWith(u8, str, "\n\n")));
                const is_sentence_newline = (cur == 107 and recent_count >= 2 and (
                    recent_buf[recent_count - 2] == 108 or
                    recent_buf[recent_count - 2] == 235270 or // '.'
                    recent_buf[recent_count - 2] == 235327 or // '?'
                    recent_buf[recent_count - 2] == 235272    // '!'
                ));

                if (self.in_thinking_channel) {
                    // In thinking mode, yield at paragraph breaks or complete bullet/numbered thought steps (>= 24 tokens)
                    const should_yield_thinking = (thinking_count >= 24 and is_para_break) or
                        (thinking_count >= 36 and is_sentence_newline) or
                        (thinking_count >= 64 and (is_para_break or is_sentence_newline));

                    if (should_yield_thinking) {
                        reason = protocol.STOP_ELASTIC_YIELD;
                        _ = self.m.forwardToken(self.ring, self.scratch, cur, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, false);
                        self.clock += 1;
                        break;
                    }
                } else if (response_count >= 10) {
                    // Evaluate cognitive confidence via Top-1 vs Top-2 logit margin
                    const top1_val = self.scratch.topk_candidates[0].val;
                    const top2_val = self.scratch.topk_candidates[1].val;
                    const logit_margin = top1_val - top2_val;
                    const high_confidence = logit_margin >= 1.5;

                    // Elastic yield conditions for visible response generation:
                    // 1. Definite boundary: paragraph break (\n\n) after >= 16 tokens
                    // 2. High-confidence clause/sentence boundary: sentence ending + newline with strong logit certainty after >= 12 tokens
                    // 3. Maximum elastic micro-burst: substantive sentence boundary reached after >= 32 tokens
                    const should_yield = (response_count >= 16 and is_para_break) or
                        (response_count >= 12 and is_sentence_newline and high_confidence) or
                        (response_count >= 32 and is_sentence_newline) or
                        (response_count >= 48 and (is_para_break or is_sentence_newline));

                    if (should_yield) {
                        reason = protocol.STOP_ELASTIC_YIELD;
                        _ = self.m.forwardToken(self.ring, self.scratch, cur, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, false);
                        self.clock += 1;
                        break;
                    }
                }
            }

            if (self.in_thinking_channel) {
                self.sampler.suppress_critique = (cur == 107 or cur == 108 or (str.len > 0 and str[str.len - 1] == '\n'));
                const grace_ceiling = self.thinking_budget + 32;
                const is_boundary = (str.len > 0 and (str[str.len - 1] == '\n' or str[str.len - 1] == '.' or str[str.len - 1] == '!' or str[str.len - 1] == '?' or str[str.len - 1] == ':'));

                if ((thinking_count >= self.thinking_budget and is_boundary) or thinking_count >= grace_ceiling) {
                    const bridge_text = "\n\nIdentified steps complete. To execute across multiple stages, invoke `plan()`. Otherwise, deliver the final answer.\n";
                    const bridge_tokens = try self.tok.encode(self.allocator, bridge_text, false);
                    defer self.allocator.free(bridge_tokens);
                    for (bridge_tokens) |bt| {
                        _ = self.m.forwardToken(self.ring, self.scratch, bt, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, false);
                        self.clock += 1;
                        const b_str = self.tok.decode(bt);
                        try protocol.writeToken(writer, msg_id, protocol.OP_STREAM_THOUGHT, bt, @intCast(self.clock), 0xFFFFFFFFFFFF, protocol.TOKEN_TYPE_TEXT, b_str);
                    }
                    writer.flush();
                    self.in_thinking_channel = false;
                    self.sampler.suppress_critique = false;
                    thinking_count = 0;
                    cur = self.advanceToken(101, window_tokens);
                    continue;
                }
            } else {
                self.sampler.suppress_critique = false;
            }

            const total_gen = thinking_count + response_count;
            if (total_gen % 8 == 0) {
                const el = @max(1, std.time.milliTimestamp() - start);
                const dividend = if (total_gen > 1) total_gen - 1 else total_gen;
                const total_budget: u32 = @intCast(self.max_tokens + self.thinking_budget);
                try protocol.writeStatus(writer, msg_id, protocol.STATUS_GENERATING, (@as(f32, @floatFromInt(dividend)) / @as(f32, @floatFromInt(el))) * 1000.0, self.slots(), diff_count, total_gen, total_budget, is_gpu, self.statusFlags());
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
        try protocol.writeStatus(writer, msg_id, protocol.STATUS_IDLE, tok_sec, self.slots(), diff_count, total_gen, total_gen, is_gpu, self.statusFlags());
        writer.flush();
        if (self.hippo) |*h| {
            const start_clock = if (self.clock >= h.count) self.clock - h.count else 0;
            _ = h.commit(self.archive, self.ring, self.store, start_clock);
        }
    }

    fn handleMemCommit(self: *Server) void {
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
