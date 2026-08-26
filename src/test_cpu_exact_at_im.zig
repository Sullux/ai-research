const std = @import("std");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const sampler = @import("sampler.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const model_dir = "../gemma-4-12B-it-qat-q4_0-unquantized";
    var config_path_buf: [512]u8 = undefined;
    const config = try model.ModelConfig.loadFromJson(allocator, try std.fmt.bufPrint(&config_path_buf, "{s}/config.json", .{model_dir}));
    var tok_path_buf: [512]u8 = undefined;
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{model_dir}));
    defer tok.deinit();
    var st = try safetensors.SafeTensors.openDir(allocator, model_dir);
    defer st.deinit();
    var m = try model.Model.loadFromSafeTensors(allocator, &st, config);
    defer m.deinit();

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, 384, 512, 96);
    defer ring.deinit();

    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    var tp: std.Thread.Pool = undefined;
    try tp.init(.{ .allocator = allocator, .n_jobs = 16 });
    defer tp.deinit();

    var samp = sampler.Sampler.init(42, 0.0, 1.0);

    const kernel_file = try std.fs.cwd().openFile("tui/PROMPT_KERNEL.md", .{});
    defer kernel_file.close();
    const kernel_size = try kernel_file.getEndPos();
    const kernel_buf = try allocator.alloc(u8, kernel_size);
    defer allocator.free(kernel_buf);
    _ = try kernel_file.readAll(kernel_buf);

    var prompt_buf = std.ArrayList(u8).init(allocator);
    defer prompt_buf.deinit();
    try prompt_buf.appendSlice("<|turn>system\n<|think|>\n");
    try prompt_buf.appendSlice(std.mem.trim(u8, kernel_buf, " \t\r\n"));
    try prompt_buf.appendSlice("\n<turn|>\n<|turn>user\nHow are you today?<turn|>\n<|turn>model\n");

    const tok1 = try tok.encode(allocator, prompt_buf.items, true);
    defer allocator.free(tok1);

    std.debug.print("CPU FP32 Prompt token length: {}\n", .{tok1.len});

    var clock: usize = 0;
    var cur: u32 = 0;
    for (tok1, 0..) |t, i| {
        cur = m.forwardToken(&ring, &scratch, t, clock, &tp, null, null, null, i == tok1.len - 1);
        clock += 1;
    }

    std.debug.print("Prefill completed. Sampling step 0: id={} str='{s}'\n", .{ cur, tok.decode(cur) });
}
