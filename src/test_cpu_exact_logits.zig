const std = @import("std");
const model = @import("model/types.zig");
const loader = @import("model/loader.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const quiescence = @import("quiescence.zig");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const model_dir = "../gemma-4-12B-it-qat-q4_0-unquantized";
    const config = try model.ModelConfig.loadFromJson(alloc, model_dir ++ "/config.json");
    var tok = try tokenizer.Tokenizer.loadFromJson(alloc, model_dir ++ "/tokenizer.json");
    defer tok.deinit();

    var st = try safetensors.SafeTensors.openDir(alloc, model_dir);
    defer st.deinit();

    var m = try loader.Model.loadFromSafeTensors(alloc, &st, config);
    defer m.deinit();

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(alloc, config.num_hidden_layers, max_kv_dim, 32, 2048, 96);
    defer ring.deinit();

    var q_tracker = quiescence.QuiescenceTracker.init(.{ .threshold = 0.0 }, config.num_hidden_layers);
    var scratch = try model.ForwardScratch.init(alloc, config);
    defer scratch.deinit(alloc);

    var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = alloc });
    defer pool.deinit();

    // Exact prompt up to "I'" on CPU BF16 unquantized
    const full_prompt =
        "<|turn>system\n<|think|>\n" ++
        "You are a helpful, respectful and honest AI assistant operating in an AMD Ryzen computing runtime." ++
        "\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n" ++
        "<|channel>thought\n" ++
        "The user is asking \"How are you doing today?\". This is a standard social greeting. I should respond politely and helpfully, acknowledging my nature as an AI.\n\n" ++
        "Plan:\n1. Acknow or respond to the greeting.\n2. State that I'm doing well and ready to assist.\n3. Ask how I can help the user.<channel|>" ++
        "I'm doing well, thank you for asking! I'";

    const tokens = try tok.encode(alloc, full_prompt, true);
    defer alloc.free(tokens);

    for (tokens, 0..) |t, i| {
        _ = m.forwardToken(&ring, &scratch, t, i, &pool, null, &q_tracker, null, i + 1 == tokens.len);
    }

    std.debug.print("CPU BF16 Logits after \"I'\":\n", .{});
    const id_m: u32 = 236757; // 'm'
    const id_s: u32 = 236751; // 's'
    const id_ll: u32 = 236829; // 'll'
    const id_ve: u32 = 236814; // 've'
    const id_d: u32 = 236754; // 'd'

    std.debug.print("  'm'  (id {d}): {d:.4}\n", .{ id_m, scratch.logits[id_m] });
    std.debug.print("  's'  (id {d}): {d:.4}\n", .{ id_s, scratch.logits[id_s] });
    std.debug.print("  'll' (id {d}): {d:.4}\n", .{ id_ll, scratch.logits[id_ll] });
    std.debug.print("  've' (id {d}): {d:.4}\n", .{ id_ve, scratch.logits[id_ve] });
    std.debug.print("  'd'  (id {d}): {d:.4}\n", .{ id_d, scratch.logits[id_d] });

    // Top 5 candidate logits
    var top_cand: [5]struct { id: u32, val: f32 } = undefined;
    for (&top_cand) |*c| c.* = .{ .id = 0, .val = -1e9 };
    for (scratch.logits, 0..) |v, id| {
        if (id == 0 or id == 258882 or id == 258883) continue;
        for (0..5) |k| {
            if (v > top_cand[k].val) {
                var r: usize = 4;
                while (r > k) : (r -= 1) top_cand[r] = top_cand[r - 1];
                top_cand[k] = .{ .id = @intCast(id), .val = v };
                break;
            }
        }
    }
    std.debug.print("Top 5 candidates:\n", .{});
    for (top_cand) |c| {
        std.debug.print("  id={d:<8} val={d:<8.4} '{s}'\n", .{ c.id, c.val, tok.decode(c.id) });
    }
}
