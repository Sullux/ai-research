const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const ring_buffer = @import("ring_buffer.zig");
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

    const H = config.hidden_size;
    const inter = config.intermediate_size;
    const eps = config.rms_norm_eps;
    const q_dim = m.layers[0].q_dim;
    const kv_dim = m.layers[0].kv_dim;
    const head_dim: u32 = @intCast(m.layers[0].head_dim);
    const rot_dim: u32 = @intCast(m.layers[0].rotary_dim);
    const theta = config.rope_theta;
    const gqa_ratio: u32 = @intCast(config.num_attention_heads / m.layers[0].num_kv_heads);
    const l_gpu = gpu_model.layers[0];

    // Measure each individual kernel for 100 iterations
    const ITERS: usize = 100;
    const cmd = gpu_model.engine.cmd_buf;

    std.debug.print("\n=== Per-Kernel Decode Breakdown (1 token, 1 layer) ===\n", .{});

    // 1. RMSNorm
    {
        const t0 = std.time.nanoTimestamp();
        for (0..ITERS) |_| {
            const bi = types_dispatch.VkCommandBufferBeginInfo{};
            _ = gpu_ctx.api.vkBeginCommandBuffer(cmd, &bi);
            gpu_model.engine.recordRmsNorm(cmd, l_gpu.desc.input_norm, H, eps);
            _ = gpu_ctx.api.vkEndCommandBuffer(cmd);
            try gpu_model.engine.submitPreRecorded(cmd);
        }
        const t1 = std.time.nanoTimestamp();
        std.debug.print("1. RMSNorm         : {d:.3} ms\n", .{@as(f64, @floatFromInt(t1 - t0)) / (1e6 * @as(f64, @floatFromInt(ITERS)))});
    }

    // 2. Q Proj GEMV
    {
        const t0 = std.time.nanoTimestamp();
        for (0..ITERS) |_| {
            const bi = types_dispatch.VkCommandBufferBeginInfo{};
            _ = gpu_ctx.api.vkBeginCommandBuffer(cmd, &bi);
            gpu_model.engine.recordGemv(cmd, l_gpu.desc.q_proj, q_dim, H);
            _ = gpu_ctx.api.vkEndCommandBuffer(cmd);
            try gpu_model.engine.submitPreRecorded(cmd);
        }
        const t1 = std.time.nanoTimestamp();
        std.debug.print("2. Q Proj GEMV     : {d:.3} ms\n", .{@as(f64, @floatFromInt(t1 - t0)) / (1e6 * @as(f64, @floatFromInt(ITERS)))});
    }

    // 3. K/V Proj GEMV
    {
        const t0 = std.time.nanoTimestamp();
        for (0..ITERS) |_| {
            const bi = types_dispatch.VkCommandBufferBeginInfo{};
            _ = gpu_ctx.api.vkBeginCommandBuffer(cmd, &bi);
            gpu_model.engine.recordGemv(cmd, l_gpu.desc.k_proj, kv_dim, H);
            _ = gpu_ctx.api.vkEndCommandBuffer(cmd);
            try gpu_model.engine.submitPreRecorded(cmd);
        }
        const t1 = std.time.nanoTimestamp();
        std.debug.print("3. K Proj GEMV     : {d:.3} ms\n", .{@as(f64, @floatFromInt(t1 - t0)) / (1e6 * @as(f64, @floatFromInt(ITERS)))});
    }

    // 4. QKV RoPE
    {
        const t0 = std.time.nanoTimestamp();
        for (0..ITERS) |_| {
            const bi = types_dispatch.VkCommandBufferBeginInfo{};
            _ = gpu_ctx.api.vkBeginCommandBuffer(cmd, &bi);
            gpu_model.engine.recordQkvRope(cmd, l_gpu.desc.qkv_rope, config.num_attention_heads, m.layers[0].num_kv_heads, head_dim, rot_dim, m.layers[0].k_eq_v, theta, eps);
            _ = gpu_ctx.api.vkEndCommandBuffer(cmd);
            try gpu_model.engine.submitPreRecorded(cmd);
        }
        const t1 = std.time.nanoTimestamp();
        std.debug.print("4. QKV RoPE        : {d:.3} ms\n", .{@as(f64, @floatFromInt(t1 - t0)) / (1e6 * @as(f64, @floatFromInt(ITERS)))});
    }

    // 5. Decode Attention
    {
        const t0 = std.time.nanoTimestamp();
        for (0..ITERS) |_| {
            const bi = types_dispatch.VkCommandBufferBeginInfo{};
            _ = gpu_ctx.api.vkBeginCommandBuffer(cmd, &bi);
            gpu_model.engine.recordDecodeAttn(cmd, l_gpu.desc.attn, head_dim, kv_dim, gqa_ratio, 1.0, config.num_attention_heads);
            _ = gpu_ctx.api.vkEndCommandBuffer(cmd);
            try gpu_model.engine.submitPreRecorded(cmd);
        }
        const t1 = std.time.nanoTimestamp();
        std.debug.print("5. Decode Attention: {d:.3} ms\n", .{@as(f64, @floatFromInt(t1 - t0)) / (1e6 * @as(f64, @floatFromInt(ITERS)))});
    }

    // 6. O Proj GEMV
    {
        const t0 = std.time.nanoTimestamp();
        for (0..ITERS) |_| {
            const bi = types_dispatch.VkCommandBufferBeginInfo{};
            _ = gpu_ctx.api.vkBeginCommandBuffer(cmd, &bi);
            gpu_model.engine.recordGemv(cmd, l_gpu.desc.o_proj, H, q_dim);
            _ = gpu_ctx.api.vkEndCommandBuffer(cmd);
            try gpu_model.engine.submitPreRecorded(cmd);
        }
        const t1 = std.time.nanoTimestamp();
        std.debug.print("6. O Proj GEMV     : {d:.3} ms\n", .{@as(f64, @floatFromInt(t1 - t0)) / (1e6 * @as(f64, @floatFromInt(ITERS)))});
    }

    // 7. Gate/Up Fused MLP
    {
        const t0 = std.time.nanoTimestamp();
        for (0..ITERS) |_| {
            const bi = types_dispatch.VkCommandBufferBeginInfo{};
            _ = gpu_ctx.api.vkBeginCommandBuffer(cmd, &bi);
            gpu_model.engine.recordGateUpSwiGlu(cmd, l_gpu.desc.gate_up_swiglu, inter, H);
            _ = gpu_ctx.api.vkEndCommandBuffer(cmd);
            try gpu_model.engine.submitPreRecorded(cmd);
        }
        const t1 = std.time.nanoTimestamp();
        std.debug.print("7. Gate/Up MLP     : {d:.3} ms\n", .{@as(f64, @floatFromInt(t1 - t0)) / (1e6 * @as(f64, @floatFromInt(ITERS)))});
    }

    // 8. Down Proj GEMV
    {
        const t0 = std.time.nanoTimestamp();
        for (0..ITERS) |_| {
            const bi = types_dispatch.VkCommandBufferBeginInfo{};
            _ = gpu_ctx.api.vkBeginCommandBuffer(cmd, &bi);
            gpu_model.engine.recordGemvMlp(cmd, l_gpu.desc.down_proj, H, inter);
            _ = gpu_ctx.api.vkEndCommandBuffer(cmd);
            try gpu_model.engine.submitPreRecorded(cmd);
        }
        const t1 = std.time.nanoTimestamp();
        std.debug.print("8. Down Proj GEMV  : {d:.3} ms\n", .{@as(f64, @floatFromInt(t1 - t0)) / (1e6 * @as(f64, @floatFromInt(ITERS)))});
    }
}
