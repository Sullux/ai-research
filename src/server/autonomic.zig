const std = @import("std");
const protocol = @import("../protocol.zig");
const template_state = @import("template_state.zig");
const Server = @import("../server.zig").Server;

pub const AutonomicProbeResult = struct {
    winning_idx: usize,
    winning_token: u32,
    confidence: f32,
    entropy: f32,
    cost_ms: f32,
    decoded_buf: [256]u8 = undefined,
    decoded_len: usize = 0,

    pub fn decoded(self: *const AutonomicProbeResult) []const u8 {
        return self.decoded_buf[0..self.decoded_len];
    }
};

pub fn probeAutonomic(
    self: *Server,
    msg_id: u16,
    prompt: []const u8,
    candidates: []const u32,
    max_decode_tokens: usize,
    writer: anytype,
) anyerror!AutonomicProbeResult {
    const t_start = std.time.milliTimestamp();

    // Canonical template framing based on outer template state
    const framed_prompt = try template_state.formatProbeFrame(self.allocator, self.template_state, prompt);
    defer self.allocator.free(framed_prompt);

    const tokens = try self.tok.encode(self.allocator, framed_prompt, false);
    defer self.allocator.free(tokens);

    if (tokens.len == 0) {
        return .{
            .winning_idx = 0,
            .winning_token = if (candidates.len > 0) candidates[0] else 0,
            .confidence = 0.0,
            .entropy = 0.0,
            .cost_ms = 0.0,
        };
    }

    const saved_clock = self.clock;
    const saved_state = self.template_state;
    defer {
        self.ring.rollbackClock(saved_clock);
        self.clock = saved_clock;
        self.template_state = saved_state;
    }

    const last_tok: u32 = tokens[tokens.len - 1];
    if (tokens.len > 1) {
        _ = try self.prefillTokens(msg_id, tokens[0 .. tokens.len - 1], writer, true);
    }
    _ = self.m.forwardToken(self.ring, self.scratch, last_tok, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, true);
    self.clock += 1;

    var result = AutonomicProbeResult{
        .winning_idx = 0,
        .winning_token = 0,
        .confidence = 0.0,
        .entropy = 0.0,
        .cost_ms = 0.0,
    };

    if (max_decode_tokens <= 1 and candidates.len > 0) {
        const eval_res = if (self.gpu_opt != null)
            self.sampler.evalConstrainedTopK(&self.scratch.topk_candidates, candidates)
        else
            self.sampler.evalConstrained(self.scratch.logits, candidates);

        result.winning_idx = eval_res.winning_idx;
        result.winning_token = eval_res.winning_token;
        result.confidence = eval_res.confidence;
        result.entropy = eval_res.entropy;
    } else {
        var decoded_tokens: [16]u32 = undefined;
        var decoded_count: usize = 0;
        const max_count = @min(max_decode_tokens, decoded_tokens.len);

        var next_tok = if (self.gpu_opt != null)
            self.sampler.sampleTopK(&self.scratch.topk_candidates, null)
        else
            self.sampler.sample(self.scratch.logits, null);

        while (decoded_count < max_count) {
            if (next_tok == 106 or next_tok == 100 or next_tok == 101 or next_tok == 107 or next_tok == 108) break;
            decoded_tokens[decoded_count] = next_tok;
            decoded_count += 1;

            _ = self.m.forwardToken(self.ring, self.scratch, next_tok, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, true);
            self.clock += 1;

            next_tok = if (self.gpu_opt != null)
                self.sampler.sampleTopK(&self.scratch.topk_candidates, null)
            else
                self.sampler.sample(self.scratch.logits, null);
        }

        var title_len: usize = 0;
        for (decoded_tokens[0..decoded_count]) |dt| {
            const piece = self.tok.decode(dt);
            if (title_len + piece.len > result.decoded_buf.len) break;
            @memcpy(result.decoded_buf[title_len .. title_len + piece.len], piece);
            title_len += piece.len;
        }

        if (title_len > 0) {
            const full_text = result.decoded_buf[0..title_len];
            const trimmed = std.mem.trim(u8, full_text, " \t\r\n\"'");
            var clean_len = trimmed.len;
            if (std.mem.indexOf(u8, trimmed, "\n")) |nl| {
                clean_len = nl;
            }
            const clean_slice = std.mem.trim(u8, trimmed[0..clean_len], " \t\r\n\"'");
            @memcpy(result.decoded_buf[0..clean_slice.len], clean_slice);
            result.decoded_len = clean_slice.len;
        }
        result.confidence = 1.0;
    }

    const t_now = std.time.milliTimestamp();
    result.cost_ms = @floatFromInt(@max(0, t_now - t_start));
    return result;
}

pub fn probeThinkingGate(self: *Server, msg_id: u16, writer: anytype) anyerror!bool {
    var cand_toks: [2]u32 = undefined;
    const cand_strs = [_][]const u8{ "0", "1" };
    for (cand_strs, 0..) |cs, i| {
        const enc = try self.tok.encode(self.allocator, cs, false);
        defer self.allocator.free(enc);
        if (enc.len > 0) cand_toks[i] = enc[0] else return true;
    }

    const probe_prompt = "\n[Reflex Gate] For the preceding request, step-by-step reasoning is:\n0: Unnecessary (simple, factual, or conversational greeting)\n1: Essential (complex, logical, analytical, or multi-step)\nDecision: ";
    const probe_res = try self.probeAutonomic(msg_id, probe_prompt, &cand_toks, 1, writer);
    if (probe_res.winning_idx == 0 and probe_res.confidence >= 0.70) {
        return false;
    }
    return true;
}

pub fn handleProbeAutonomic(self: *Server, msg_id: u16, p: []const u8, writer: anytype) !void {
    if (p.len < 10) return;
    const max_decode_tokens = std.mem.readInt(u16, p[0..2][0..2], .little);
    const candidate_count = std.mem.readInt(u16, p[2..4][0..2], .little);
    const prompt_len = std.mem.readInt(u16, p[8..10][0..2], .little);
    if (p.len < 10 + prompt_len) return;
    const prompt_str = p[10 .. 10 + prompt_len];

    var offset: usize = 10 + prompt_len;
    var cand_toks: [32]u32 = undefined;
    var cand_count: usize = 0;

    for (0..candidate_count) |_| {
        if (offset + 2 > p.len) break;
        const c_len = std.mem.readInt(u16, p[offset .. offset + 2][0..2], .little);
        offset += 2;
        if (offset + c_len > p.len) break;
        const c_str = p[offset .. offset + c_len];
        offset += c_len;
        if (cand_count < cand_toks.len) {
            const encoded = try self.tok.encode(self.allocator, c_str, false);
            defer self.allocator.free(encoded);
            if (encoded.len > 0) {
                cand_toks[cand_count] = encoded[0];
                cand_count += 1;
            }
        }
    }

    const res = try self.probeAutonomic(msg_id, prompt_str, cand_toks[0..cand_count], max_decode_tokens, writer);
    try protocol.writeAutonomicResult(writer, msg_id, @intCast(res.winning_idx), res.confidence, res.entropy, res.cost_ms, res.decoded());
    writer.flush();
}

pub fn handleTaskTriage(self: *Server, msg_id: u16, p: []const u8, writer: anytype) !void {
    if (p.len < 4) {
        try protocol.writeEventRouted(writer, msg_id, 0, 0, 1);
        writer.flush();
        return;
    }
    const event_id = std.mem.readInt(u16, p[0..2][0..2], .little);
    const num_tasks = std.mem.readInt(u16, p[2..4][0..2], .little);
    var offset: usize = 4;

    var task_ids: [16]u16 = undefined;
    var task_count: usize = 0;

    var prompt_buf: [2048]u8 = undefined;
    var stream = std.io.fixedBufferStream(&prompt_buf);
    const p_writer = stream.writer();

    try p_writer.writeAll("Task Routing\nActive tasks:\n");

    for (0..num_tasks) |_| {
        if (offset + 4 > p.len) break;
        const tid = std.mem.readInt(u16, p[offset .. offset + 2][0..2], .little);
        const title_len = std.mem.readInt(u16, p[offset + 2 .. offset + 4][0..2], .little);
        offset += 4;
        if (offset + title_len > p.len) break;
        const title = p[offset .. offset + title_len];
        offset += title_len;
        if (task_count < 15) {
            task_ids[task_count] = tid;
            task_count += 1;
            try std.fmt.format(p_writer, "{d}: {s}\n", .{ task_count, title });
        }
    }

    try p_writer.writeAll("0: New independent task (only if completely unrelated to active tasks above)\n\nNote: Follow-ups, revisions, negative constraints (\"no X\", \"use Y instead\"), corrections, and steering belong to the active task being steered.\n");

    if (offset + 2 <= p.len) {
        const ev_len = std.mem.readInt(u16, p[offset .. offset + 2][0..2], .little);
        offset += 2;
        if (offset + ev_len <= p.len) {
            const ev_text = p[offset .. offset + ev_len];
            try std.fmt.format(p_writer, "Event: {s}\nTarget index: ", .{ ev_text });
        }
    }

    if (task_count == 0) {
        try protocol.writeEventRouted(writer, msg_id, event_id, 0, 1);
        writer.flush();
        return;
    }

    var cand_toks: [16]u32 = undefined;
    var cand_count: usize = 0;
    for (1..task_count + 1) |i| {
        var digit_buf: [4]u8 = undefined;
        const digit_str = try std.fmt.bufPrint(&digit_buf, "{d}", .{i});
        const encoded = try self.tok.encode(self.allocator, digit_str, false);
        defer self.allocator.free(encoded);
        if (encoded.len > 0) {
            cand_toks[cand_count] = encoded[0];
            cand_count += 1;
        }
    }
    {
        const encoded = try self.tok.encode(self.allocator, "0", false);
        defer self.allocator.free(encoded);
        if (encoded.len > 0) {
            cand_toks[cand_count] = encoded[0];
            cand_count += 1;
        }
    }

    const probe_res = try self.probeAutonomic(msg_id, stream.getWritten(), cand_toks[0..cand_count], 1, writer);
    var chosen_idx: usize = 0;
    if (probe_res.winning_idx < task_count) {
        chosen_idx = probe_res.winning_idx + 1;
    } else {
        chosen_idx = 0;
    }

    if (chosen_idx == 0) {
        try protocol.writeEventRouted(writer, msg_id, event_id, 0, 1);
    } else {
        const target_task_id = task_ids[chosen_idx - 1];
        try protocol.writeEventRouted(writer, msg_id, event_id, target_task_id, 0);
    }
    writer.flush();
}

pub fn handleTaskTitle(self: *Server, msg_id: u16, p: []const u8, writer: anytype) !void {
    if (p.len < 6) return;
    const task_id = std.mem.readInt(u32, p[0..4][0..4], .little);
    const prompt_len = std.mem.readInt(u16, p[4..6][0..2], .little);
    if (p.len < 6 + prompt_len) return;
    const prompt_text = p[6 .. 6 + prompt_len];

    var prompt_buf: [2048]u8 = undefined;
    var stream = std.io.fixedBufferStream(&prompt_buf);
    const p_writer = stream.writer();

    try p_writer.writeAll("Provide a concise 2 to 5 word title for this user request:\n\"");
    const max_sample = @min(prompt_text.len, 256);
    try p_writer.writeAll(prompt_text[0..max_sample]);
    try p_writer.writeAll("\"\nOutput only the title, nothing else.");

    const probe_res = try self.probeAutonomic(msg_id, stream.getWritten(), &.{}, 14, writer);
    var title_str = probe_res.decoded();
    if (title_str.len == 0) {
        const fb_len = @min(prompt_text.len, 50);
        title_str = prompt_text[0..fb_len];
    }

    try protocol.writeTaskTitleResult(writer, msg_id, task_id, title_str);
    writer.flush();
}

pub fn handleBacklogTriage(self: *Server, msg_id: u16, p: []const u8, writer: anytype) !void {
    if (p.len < 4) {
        try protocol.writeError(writer, msg_id, "Payload too short for backlog triage");
        writer.flush();
        return;
    }
    const event_id = std.mem.readInt(u16, p[0..2][0..2], .little);
    const title_len = std.mem.readInt(u16, p[2..4][0..2], .little);
    if (4 + title_len > p.len) {
        try protocol.writeError(writer, msg_id, "Malformed backlog triage title");
        writer.flush();
        return;
    }
    const title = p[4 .. 4 + title_len];

    var prompt_buf: [2048]u8 = undefined;
    var stream = std.io.fixedBufferStream(&prompt_buf);
    const p_writer = stream.writer();

    try std.fmt.format(p_writer, "Backlog Triage\nTask: {s}\n0: Satisfied or obsolete (ACK)\n1: Execute now (Resume)\n2: Defer for later (Snooze)\nDecision: ", .{ title });

    const cand_strs = [_][]const u8{ "0", "1", "2" };
    var cand_toks: [3]u32 = undefined;
    var cand_count: usize = 0;
    for (cand_strs) |cs| {
        const encoded = try self.tok.encode(self.allocator, cs, false);
        defer self.allocator.free(encoded);
        if (encoded.len > 0) {
            cand_toks[cand_count] = encoded[0];
            cand_count += 1;
        }
    }

    const probe_res = try self.probeAutonomic(msg_id, stream.getWritten(), cand_toks[0..cand_count], 1, writer);
    const chosen_action: u8 = @intCast(probe_res.winning_idx);
    try protocol.writeBacklogRouted(writer, msg_id, event_id, chosen_action);
    writer.flush();
}
