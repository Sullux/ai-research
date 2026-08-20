const std = @import("std");
pub const safetensors = @import("safetensors.zig");
pub const tensor = @import("tensor.zig");
pub const kernels = @import("kernels.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const model = @import("model.zig");
pub const ring_buffer = @import("ring_buffer.zig");
pub const diff = @import("diff.zig");
pub const memory = @import("memory.zig");
pub const storage = @import("storage.zig");
pub const quiescence = @import("quiescence.zig");
pub const vq = @import("vq.zig");
pub const gpu = @import("gpu.zig");
pub const quant = @import("quant.zig");

const MEMORY_CAPACITY: usize = 8192;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var thread_pool: std.Thread.Pool = undefined;
    try thread_pool.init(.{ .allocator = allocator });
    defer thread_pool.deinit();

    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var model_dir: []const u8 = "../gemma-4-E2B";
    var max_tokens: usize = 128;
    var num_anchors: usize = 32;
    var window_size: usize = 512;
    var num_recall: usize = 96;
    var memory_enabled = true;
    var quiescence_enabled = false;
    var gpu_enabled = false;
    var quant_mode: quant.QuantMode = .none;
    var storage_path: ?[]const u8 = null;
    var prompt_buf = std.ArrayList(u8).init(allocator);
    defer prompt_buf.deinit();

    var arg_idx: usize = 1;
    while (arg_idx < args.len) : (arg_idx += 1) {
        const arg = args[arg_idx];
        if ((std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) and arg_idx + 1 < args.len) { model_dir = args[arg_idx + 1]; arg_idx += 1;
        } else if ((std.mem.eql(u8, arg, "--max-tokens") or std.mem.eql(u8, arg, "-n")) and arg_idx + 1 < args.len) { max_tokens = std.fmt.parseInt(usize, args[arg_idx + 1], 10) catch 128; arg_idx += 1;
        } else if (std.mem.eql(u8, arg, "--anchors") and arg_idx + 1 < args.len) { num_anchors = std.fmt.parseInt(usize, args[arg_idx + 1], 10) catch 32; arg_idx += 1;
        } else if (std.mem.eql(u8, arg, "--window") and arg_idx + 1 < args.len) { window_size = std.fmt.parseInt(usize, args[arg_idx + 1], 10) catch 512; arg_idx += 1;
        } else if (std.mem.eql(u8, arg, "--recall") and arg_idx + 1 < args.len) { num_recall = std.fmt.parseInt(usize, args[arg_idx + 1], 10) catch 96; arg_idx += 1;
        } else if (std.mem.eql(u8, arg, "--storage") and arg_idx + 1 < args.len) { storage_path = args[arg_idx + 1]; arg_idx += 1;
        } else if (std.mem.eql(u8, arg, "--quiescence")) { quiescence_enabled = true;
        } else if (std.mem.eql(u8, arg, "--gpu")) { gpu_enabled = true;
        } else if (std.mem.eql(u8, arg, "--q8")) { quant_mode = .q8; gpu_enabled = true;
        } else if (std.mem.eql(u8, arg, "--q4")) { quant_mode = .q4; gpu_enabled = true;
        } else if (std.mem.eql(u8, arg, "--quant") and arg_idx + 1 < args.len) { quant_mode = quant.QuantMode.fromString(args[arg_idx + 1]); gpu_enabled = true; arg_idx += 1;
        } else if (std.mem.eql(u8, arg, "--no-memory")) { memory_enabled = false;
        } else {
            if (prompt_buf.items.len > 0) try prompt_buf.append(' ');
            try prompt_buf.appendSlice(arg);
        }
    }

    num_recall = @min(num_recall, model.types.MAX_RECALL_SLOTS);
    try stdout.print("Loading model from: {s}...\n", .{model_dir});

    var config_path_buf: [512]u8 = undefined;
    const config = try model.ModelConfig.loadFromJson(allocator, try std.fmt.bufPrint(&config_path_buf, "{s}/config.json", .{model_dir}));
    var tok_path_buf: [512]u8 = undefined;
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{model_dir}));
    defer tok.deinit();
    var st = try safetensors.SafeTensors.openDir(allocator, model_dir);
    defer st.deinit();
    var m = try model.Model.loadFromSafeTensors(allocator, &st, config);
    defer m.deinit();

    var gpu_ctx: ?gpu.context.GpuContext = if (gpu_enabled) gpu.context.GpuContext.init(allocator) catch null else null;
    defer if (gpu_ctx) |*gc| gc.deinit();
    var gpu_model_ctx: ?gpu.model_gpu.GpuModelContext = if (gpu_ctx) |*gc| gpu.model_gpu.GpuModelContext.init(allocator, gc, &m, config, quant_mode) catch |err| {
        std.debug.print("GPU init error: {any}\n", .{err});
        return err;
    } else null;
    defer if (gpu_model_ctx) |*gmc| gmc.deinit();
    const gpu_ptr: ?*gpu.model_gpu.GpuModelContext = if (gpu_model_ctx) |*gmc| gmc else null;
    if (gpu_ctx) |gc| {
        const mode_tag = switch (quant_mode) { .none => "BF16", .q8 => "Q8_0", .q4 => "Q4_0" };
        try stdout.print("GPU: {s} (UMA Compute, {s})\n", .{ gc.device_name[0..(std.mem.indexOfScalar(u8, &gc.device_name, 0) orelse gc.device_name.len)], mode_tag });
    }

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, num_anchors, window_size, num_recall);
    defer ring.deinit();

    var archive: ?memory.DiffArchive = null;
    var store: ?storage.PersistentDiffStore = null;
    defer if (store) |*s| s.close();
    if (memory_enabled) {
        var arch = try memory.DiffArchive.init(allocator, config.hidden_size, MEMORY_CAPACITY, .{});
        if (storage_path) |sp| {
            var s = try storage.PersistentDiffStore.open(sp, MEMORY_CAPACITY, config.hidden_size);
            _ = s.loadIntoArchive(&arch);
            store = s;
        }
        archive = arch;
    }
    defer if (archive) |*a| {
        if (store) |*s| s.saveFromArchive(a);
        a.deinit();
    };

    var q_tracker = quiescence.QuiescenceTracker.init(.{ .enabled = quiescence_enabled }, config.num_hidden_layers);
    const q_ptr: ?*quiescence.QuiescenceTracker = if (quiescence_enabled) &q_tracker else null;
    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);
    const memory_ptr: ?*memory.DiffArchive = if (archive) |*a| a else null;

    if (prompt_buf.items.len > 0) {
        var clock: usize = 0;
        try runInference(&m, &tok, &ring, &scratch, prompt_buf.items, max_tokens, &thread_pool, stdout, allocator, memory_ptr, q_ptr, gpu_ptr, &clock, true);
        try stdout.print("\n", .{});
        return;
    }

    try stdout.print("\n=== Gemma 4 Dynamic Streaming REPL ({s}) ===\nAnchors: {d}, Window: {d}, Recall: {d}, Total Slots: {d}\n\n", .{ model_dir, num_anchors, window_size, num_recall, ring.total_slots });

    var global_clock: usize = 0;
    var line_buf: [4096]u8 = undefined;
    while (true) {
        try stdout.print(">>> ", .{});
        const maybe_line = try stdin.readUntilDelimiterOrEof(&line_buf, '\n');
        const line = maybe_line orelse break;
        const trimmed = std.mem.trim(u8, line, " \r\t");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) break;

        try runInference(&m, &tok, &ring, &scratch, trimmed, max_tokens, &thread_pool, stdout, allocator, memory_ptr, q_ptr, gpu_ptr, &global_clock, false);
        try stdout.print("\n\n", .{});
    }
}

fn printToken(stdout: anytype, token_str: []const u8) !void {
    var i: usize = 0;
    while (i < token_str.len) {
        if (i + 2 < token_str.len and token_str[i] == 0xE2 and token_str[i + 1] == 0x96 and token_str[i + 2] == 0x81) {
            try stdout.print(" ", .{});
            i += 3;
        } else {
            try stdout.print("{c}", .{token_str[i]});
            i += 1;
        }
    }
}

fn runInference(
    m: *const model.Model, tok: *const tokenizer.Tokenizer, ring: *ring_buffer.DynamicRingBuffer,
    scratch: *model.ForwardScratch, prompt: []const u8, max_tokens: usize,
    thread_pool: *std.Thread.Pool, stdout: anytype, allocator: std.mem.Allocator,
    memory_opt: ?*memory.DiffArchive, quiescence_opt: ?*quiescence.QuiescenceTracker,
    gpu_opt: ?*gpu.model_gpu.GpuModelContext,
    clock_ptr: *usize, reset_ring: bool,
) !void {
    const prompt_tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(prompt_tokens);
    if (reset_ring) ring.reset();

    for (prompt_tokens) |t| {
        m.forwardToken(ring, scratch, t, clock_ptr.*, thread_pool, memory_opt, quiescence_opt, gpu_opt);
        clock_ptr.* += 1;
    }
    var current_token = kernels.sampleArgmax(scratch.logits);

    const gen_start = std.time.milliTimestamp();
    var gen_count: usize = 0;
    for (0..max_tokens) |_| {
        const token_str = tok.decode(current_token);
        try printToken(stdout, token_str);
        gen_count += 1;
        if (current_token == tok.eos_token_id) break;
        m.forwardToken(ring, scratch, current_token, clock_ptr.*, thread_pool, memory_opt, quiescence_opt, gpu_opt);
        current_token = kernels.sampleArgmax(scratch.logits);
        clock_ptr.* += 1;
    }
    const elapsed_ms = std.time.milliTimestamp() - gen_start;
    if (gen_count > 0 and elapsed_ms > 0) {
        const tps = (@as(f64, @floatFromInt(gen_count)) / @as(f64, @floatFromInt(elapsed_ms))) * 1000.0;
        try stdout.print("\n[{d} tokens, {d:.1} tok/s]", .{ gen_count, tps });
    }
}

test {
    _ = @import("safetensors.zig"); _ = @import("tensor.zig"); _ = @import("kernels.zig");
    _ = @import("tokenizer.zig"); _ = @import("ring_buffer.zig"); _ = @import("diff.zig");
    _ = @import("memory.zig"); _ = @import("storage.zig"); _ = @import("quiescence.zig");
    _ = @import("vq.zig"); _ = @import("gpu.zig"); _ = @import("quant.zig");
}
