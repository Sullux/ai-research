const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const batch_prefill = @import("gpu/batch_prefill.zig");
const types = @import("gpu/types.zig");
const types_dispatch = @import("gpu/types_dispatch.zig");

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

    var gpu_ctx = try context.GpuContext.init(allocator);
    defer gpu_ctx.deinit();

    var gpu_model = try model_gpu.GpuModelContext.init(allocator, &gpu_ctx, &m, config, .q4, 0.0);
    defer gpu_model.deinit();

    const bp = gpu_model.batch_prefill_ctx.?;
    const N: u32 = 405;
    const H: u32 = @intCast(config.hidden_size);
    const inter: u32 = @intCast(config.intermediate_size);
    const q_dim: u32 = @intCast(m.layers[0].q_dim);
    const kv_dim: u32 = @intCast(m.layers[0].kv_dim);

    const d = bp.layers[0];

    const n_tiles = (N + 63) / 64;
    const q_tiles = (q_dim + 15) / 16;
    const inter_tiles = (inter + 15) / 16;
    const h_tiles = (H + 15) / 16;
    _ = h_tiles;

    // Time Q projection GEMM alone (405 tokens)
    const pc_q = [4]u32{ N, q_dim, H, 0 };
    const begin_info = types_dispatch.VkCommandBufferBeginInfo{};

    var total_gemm_time: i64 = 0;
    const iters = 20;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_gemm_q4.record(bp.cmd_buf, d.q_proj, std.mem.sliceAsBytes(&pc_q), q_tiles, n_tiles, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);

        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.microTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.microTimestamp();
        total_gemm_time += (t1 - t0);
    }
    std.debug.print("Single Q_proj GEMM (405 tok x 3840 dim): {d:.2} ms\n", .{@as(f64, @floatFromInt(total_gemm_time)) / @as(f64, @floatFromInt(iters * 1000))});

    // Time Fused MLP GEMM alone (405 tokens x 15360 dim)
    const pc_mlp = [4]u32{ N, inter, H, 0 };
    var total_mlp_time: i64 = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_fused_mlp_q4.record(bp.cmd_buf, d.gate_up_proj, std.mem.sliceAsBytes(&pc_mlp), inter_tiles, n_tiles, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);

        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.microTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.microTimestamp();
        total_mlp_time += (t1 - t0);
    }
    std.debug.print("Single Fused MLP (405 tok x 15360 dim): {d:.2} ms\n", .{@as(f64, @floatFromInt(total_mlp_time)) / @as(f64, @floatFromInt(iters * 1000))});

    // Time Causal Attention alone (405 tokens)
    const pc_attn = extern struct {
        head_dim: u32, kv_dim: u32, gqa_ratio: u32, inv_sqrt_dim: f32,
        num_q_heads: u32, N: u32, num_prev_slots: u32, pad: u32,
    }{
        .head_dim = 256, .kv_dim = kv_dim, .gqa_ratio = 8,
        .inv_sqrt_dim = 1.0, .num_q_heads = 16, .N = N,
        .num_prev_slots = 0, .pad = 0,
    };
    var total_attn_time: i64 = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_causal_attn.record(bp.cmd_buf, d.causal_attn, std.mem.asBytes(&pc_attn), 16, N, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);

        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.microTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.microTimestamp();
        total_attn_time += (t1 - t0);
    }
    std.debug.print("Single Causal Attention (405 tok): {d:.2} ms\n", .{@as(f64, @floatFromInt(total_attn_time)) / @as(f64, @floatFromInt(iters * 1000))});
}
