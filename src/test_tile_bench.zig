const std = @import("std");
const context = @import("gpu/context.zig");
const model_gpu = @import("gpu/model_gpu.zig");
const model = @import("model.zig");
const safetensors = @import("safetensors.zig");
const pipeline = @import("gpu/pipeline.zig");
const descriptors = @import("gpu/descriptors.zig");
const buffer = @import("gpu/buffer.zig");
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
    const d = bp.layers[0];

    std.debug.print("Benchmarking current Fused MLP for N={}...\n", .{N});
    const pc_mlp = [4]u32{ N, inter, H, 0 };
    const begin_info = types_dispatch.VkCommandBufferBeginInfo{};

    _ = bp.ctx.api.vkBeginCommandBuffer(bp.cmd_buf, &begin_info);
    bp.pipe_fused_mlp_q4.record(bp.cmd_buf, d.gate_up_proj, std.mem.sliceAsBytes(&pc_mlp), (inter + 3) / 4, (N + 3) / 4, 1);
    _ = bp.ctx.api.vkEndCommandBuffer(bp.cmd_buf);

    const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&bp.cmd_buf)[0..1].ptr };
    const t0 = std.time.microTimestamp();
    _ = bp.ctx.api.vkQueueSubmit(bp.ctx.queue, 1, (&submit_info)[0..1].ptr, bp.fence);
    _ = bp.ctx.api.vkWaitForFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr, 1, std.math.maxInt(u64));
    _ = bp.ctx.api.vkResetFences(bp.ctx.device, 1, (&bp.fence)[0..1].ptr);
    const t1 = std.time.microTimestamp();

    std.debug.print("Fused MLP execution time: {d:.2} ms\n", .{@as(f64, @floatFromInt(t1 - t0)) / 1000.0});
}
