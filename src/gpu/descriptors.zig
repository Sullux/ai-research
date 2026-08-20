const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");

pub const LayerDescriptorSets = struct {
    q_proj: types.VkDescriptorSet,
    k_proj: types.VkDescriptorSet,
    v_proj: types.VkDescriptorSet,
    o_proj: types.VkDescriptorSet,
    gate_proj: types.VkDescriptorSet,
    up_proj: types.VkDescriptorSet,
    swiglu: types.VkDescriptorSet,
    down_proj: types.VkDescriptorSet,
    gate_up_swiglu: types.VkDescriptorSet,
    pre_ffn_norm: types.VkDescriptorSet,
    post_attn_norm: types.VkDescriptorSet,
};

pub const DescriptorManager = struct {
    ctx: *const context.GpuContext,
    pool: types.VkDescriptorPool,

    pub fn init(ctx: *const context.GpuContext, max_sets: u32) !DescriptorManager {
        const pool_size = [_]types_dispatch.VkDescriptorPoolSize{
            .{ .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = max_sets * 4 },
        };
        const pool_info = types_dispatch.VkDescriptorPoolCreateInfo{
            .maxSets = max_sets,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        var pool: types.VkDescriptorPool = null;
        if (ctx.api.vkCreateDescriptorPool(ctx.device, &pool_info, null, &pool) != .SUCCESS) return error.VkDescPoolCreationFailed;
        return .{ .ctx = ctx, .pool = pool };
    }

    pub fn deinit(self: *DescriptorManager) void {
        self.ctx.api.vkDestroyDescriptorPool(self.ctx.device, self.pool, null);
    }

    pub fn allocateSet(self: *const DescriptorManager, layout: types.VkDescriptorSetLayout) !types.VkDescriptorSet {
        const alloc_info = types_dispatch.VkDescriptorSetAllocateInfo{
            .descriptorPool = self.pool,
            .descriptorSetCount = 1,
            .pSetLayouts = (&layout)[0..1].ptr,
        };
        var set: types.VkDescriptorSet = null;
        if (self.ctx.api.vkAllocateDescriptorSets(self.ctx.device, &alloc_info, (&set)[0..1].ptr) != .SUCCESS) return error.VkDescSetAllocFailed;
        return set;
    }

    pub fn bindBuffers(self: *const DescriptorManager, set: types.VkDescriptorSet, bufs: []const *const buffer.GpuBuffer) void {
        var writes: [8]types_dispatch.VkWriteDescriptorSet = undefined;
        var infos: [8]types_dispatch.VkDescriptorBufferInfo = undefined;
        const count = @min(bufs.len, 8);

        for (0..count) |i| {
            infos[i] = .{
                .buffer = bufs[i].buffer,
                .offset = 0,
                .range = bufs[i].size,
            };
            writes[i] = .{
                .dstSet = set,
                .dstBinding = @intCast(i),
                .descriptorCount = 1,
                .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = (&infos[i])[0..1].ptr,
            };
        }
        self.ctx.api.vkUpdateDescriptorSets(self.ctx.device, @intCast(count), &writes, 0, null);
    }
};
