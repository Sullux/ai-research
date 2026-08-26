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
    const num_q_heads: u32 = @intCast(config.num_attention_heads);
    const num_kv_heads: u32 = @intCast(config.num_key_value_heads);
    const head_dim: u32 = @intCast(config.head_dim);
    const gqa_ratio = num_q_heads / num_kv_heads;
    const inv_sqrt_dim = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const d = bp.layers[0];
    const n_tiles: u32 = (N + 15) / 16;
    const q_tiles: u32 = (q_dim + 3) / 4;
    const kv_tiles: u32 = (kv_dim + 3) / 4;
    const h_tiles: u32 = @intCast((H + 3) / 4);
    const inter_tiles: u32 = @intCast((inter + 3) / 4);

    const pc_q = [4]u32{ N, q_dim, H, 0 };
    const pc_kv = [4]u32{ N, kv_dim, H, 0 };
    const pc_o = [4]u32{ N, H, q_dim, 0 };
    const pc_down = [4]u32{ N, H, inter, 0 };
    const pc_fused = [4]u32{ N, inter, H, 0 };
    const pc_norm = [4]u32{ N, H, 0, 0 };
    const pc_attn = [8]u32{ head_dim, kv_dim, gqa_ratio, @as(u32, @bitCast(inv_sqrt_dim)), num_q_heads, N, 0, 0 };

    const begin_info = types_dispatch.VkCommandBufferBeginInfo{};
    const iters = 20;

    std.debug.print("\n=== Per-Kernel Breakdown (405 tokens) ===\n", .{});

    // 1. RMSNorm
    var total: i128 = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_rmsnorm.record(bp.cmd_buf, d.input_norm, std.mem.sliceAsBytes(&pc_norm), (H + 255) / 256, N, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.nanoTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.nanoTimestamp();
        total += (t1 - t0);
    }
    std.debug.print("1. Input RMSNorm        : {d:6.2} ms\n", .{@as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1e6});

    // 2. Q Proj
    total = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_gemm_q4.record(bp.cmd_buf, d.q_proj, std.mem.sliceAsBytes(&pc_q), q_tiles, n_tiles, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.nanoTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.nanoTimestamp();
        total += (t1 - t0);
    }
    std.debug.print("2. Q Proj GEMM          : {d:6.2} ms\n", .{@as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1e6});

    // 3. K Proj
    total = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_gemm_q4.record(bp.cmd_buf, d.k_proj, std.mem.sliceAsBytes(&pc_kv), kv_tiles, n_tiles, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.nanoTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.nanoTimestamp();
        total += (t1 - t0);
    }
    std.debug.print("3. K Proj GEMM          : {d:6.2} ms\n", .{@as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1e6});

    // 4. V Proj
    total = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_gemm_q4.record(bp.cmd_buf, d.v_proj, std.mem.sliceAsBytes(&pc_kv), kv_tiles, n_tiles, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.nanoTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.nanoTimestamp();
        total += (t1 - t0);
    }
    std.debug.print("4. V Proj GEMM          : {d:6.2} ms\n", .{@as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1e6});

    // 5. QKV RoPE
    total = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_qkv_rope.record(bp.cmd_buf, d.qkv_rope, std.mem.sliceAsBytes(&[3]u32{ N, 0, 0 }), (N + 15) / 16, 1, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.nanoTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.nanoTimestamp();
        total += (t1 - t0);
    }
    std.debug.print("5. QKV RoPE             : {d:6.2} ms\n", .{@as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1e6});

    // 6. Causal Attention
    total = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_causal_attn.record(bp.cmd_buf, d.causal_attn, std.mem.sliceAsBytes(&pc_attn), num_q_heads, N, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.nanoTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.nanoTimestamp();
        total += (t1 - t0);
    }
    std.debug.print("6. Causal Attention     : {d:6.2} ms\n", .{@as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1e6});

    // 7. O Proj
    total = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_gemm_q4.record(bp.cmd_buf, d.o_proj, std.mem.sliceAsBytes(&pc_o), h_tiles, n_tiles, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.nanoTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.nanoTimestamp();
        total += (t1 - t0);
    }
    std.debug.print("7. O Proj GEMM          : {d:6.2} ms\n", .{@as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1e6});

    // 8. Post-Attn AddNorm
    total = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_add_norm.record(bp.cmd_buf, d.post_attn_norm, std.mem.sliceAsBytes(&pc_norm), (H + 255) / 256, N, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.nanoTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.nanoTimestamp();
        total += (t1 - t0);
    }
    std.debug.print("8. Post-Attn AddNorm    : {d:6.2} ms\n", .{@as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1e6});

    // 9. Fused MLP
    total = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_fused_mlp_q4.record(bp.cmd_buf, d.gate_up_proj, std.mem.sliceAsBytes(&pc_fused), inter_tiles, n_tiles, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.nanoTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.nanoTimestamp();
        total += (t1 - t0);
    }
    std.debug.print("9. Fused MLP Gate/Up    : {d:6.2} ms\n", .{@as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1e6});

    // 10. Down Proj
    total = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_gemm_q4.record(bp.cmd_buf, d.down_proj, std.mem.sliceAsBytes(&pc_down), h_tiles, n_tiles, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.nanoTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.nanoTimestamp();
        total += (t1 - t0);
    }
    std.debug.print("10. Down Proj GEMM      : {d:6.2} ms\n", .{@as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1e6});

    // 11. Post-MLP AddNorm
    total = 0;
    for (0..iters) |_| {
        _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
        bp.pipe_add_norm.record(bp.cmd_buf, d.post_ffn_norm, std.mem.sliceAsBytes(&pc_norm), (H + 255) / 256, N, 1);
        _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
        const t0 = std.time.nanoTimestamp();
        _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
        _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
        _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
        const t1 = std.time.nanoTimestamp();
        total += (t1 - t0);
    }
    std.debug.print("11. Post-MLP AddNorm    : {d:6.2} ms\n", .{@as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1e6});
}
