const std = @import("std");
const model = @import("model/types.zig");
const loader = @import("model/loader.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const quiescence = @import("quiescence.zig");
const tokenizer = @import("tokenizer.zig");
const gpu = @import("gpu.zig");

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
    var ring = try ring_buffer.DynamicRingBuffer.init(alloc, config.num_hidden_layers, max_kv_dim, 32, 512, 96);
    defer ring.deinit();

    var q_tracker = quiescence.QuiescenceTracker.init(.{ .threshold = 0.0 }, config.num_hidden_layers);
    var scratch = try model.ForwardScratch.init(alloc, config);
    defer scratch.deinit(alloc);

    var gctx = try gpu.context.GpuContext.init(alloc);
    defer gctx.deinit();

    var gpu_ctx = try gpu.model_gpu.GpuModelContext.init(alloc, &gctx, &m, config, .q4, 0.0);
    defer gpu_ctx.deinit();

    var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = alloc });
    defer pool.deinit();

    const prompt = "<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n";
    const tokens = try tok.encode(alloc, prompt, true);
    defer alloc.free(tokens);

    var cur: u32 = 0;
    for (tokens, 0..) |t, i| {
        cur = m.forwardToken(&ring, &scratch, t, i, &pool, null, &q_tracker, &gpu_ctx, i + 1 == tokens.len);
    }
    std.debug.print("First generated token after prompt: id={d} ({s})\n", .{ cur, tok.decode(cur) });

    var step: usize = tokens.len;
    while (step < tokens.len + 90) : (step += 1) {
        const next_tok = m.forwardToken(&ring, &scratch, cur, step, &pool, null, &q_tracker, &gpu_ctx, true);
        const s = tok.decode(cur);
        std.debug.print("{s}", .{s});
        if (std.mem.eql(u8, s, "'")) {
            std.debug.print("\n[Step {d} after \"'\"]: Top 5 candidate logits:\n", .{step});
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
            for (top_cand) |c| {
                std.debug.print("  id={d:<8} val={d:<8.4} '{s}'\n", .{ c.id, c.val, tok.decode(c.id) });
            }
        }
        cur = next_tok;
        if (cur == tok.eos_token_id or cur == 106) break;
    }
    std.debug.print("\n", .{});
}
