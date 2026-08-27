const std = @import("std");
pub const kernels = @import("../kernels.zig");
pub const diff = @import("../diff.zig");
pub const memory = @import("../memory.zig");
pub const types = @import("types.zig");
pub const loader = @import("loader.zig");
pub const ring_buffer = @import("../ring_buffer.zig");
pub const gpu = @import("../gpu.zig");

const Model = loader.Model;
const ForwardScratch = types.ForwardScratch;
const DynamicRingBuffer = ring_buffer.DynamicRingBuffer;

pub fn syncRecallToGpu(ring: *const DynamicRingBuffer, gpu_ctx: *gpu.model_gpu.GpuModelContext) void {
    const total_slots = ring.total_slots;
    for (0..gpu_ctx.layers.len) |l| {
        const recall_count = ring.recallSlots(l);
        if (recall_count == 0) continue;
        const r_start = ring.recallStart(l);
        const gpu_k = gpu_ctx.layers[l].buf_k_cache.asSlice(f32);
        const gpu_v = gpu_ctx.layers[l].buf_v_cache.asSlice(f32);
        for (0..recall_count) |r| {
            const slot = r_start + r;
            const r_slot = l * total_slots + slot;
            const r_off = r_slot * ring.max_kv_dim;
            const g_off = slot * ring.max_kv_dim;
            @memcpy(gpu_k[g_off .. g_off + ring.max_kv_dim], ring.k[r_off .. r_off + ring.max_kv_dim]);
            @memcpy(gpu_v[g_off .. g_off + ring.max_kv_dim], ring.v[r_off .. r_off + ring.max_kv_dim]);
        }
    }
}

pub fn primeSubconsciousMemory(mem: *memory.DiffArchive, ring: *DynamicRingBuffer, scratch: *ForwardScratch, query_vec: []const f32, now_ts: u64, gpu_opt: ?*gpu.model_gpu.GpuModelContext) usize {
    const selected = mem.primeTier3(query_vec, now_ts, ring, scratch.recall_indices);
    if (selected > 0 and gpu_opt != null) {
        syncRecallToGpu(ring, gpu_opt.?);
    }
    return selected;
}

pub fn integrateMemory(self: *const Model, mem: *memory.DiffArchive, ring: *DynamicRingBuffer, scratch: *ForwardScratch, clock: usize, H: usize, tp: ?*std.Thread.Pool) void {
    _ = self;
    _ = tp;
    _ = H;
    const now_ts: u64 = @intCast(clock);
    _ = primeSubconsciousMemory(mem, ring, scratch, scratch.normed_x, now_ts, null);
}

pub fn computeKeywordQueryVector(self: *const Model, token_ids: []const u32, out_vector: []f32) bool {
    if (token_ids.len == 0 or out_vector.len != self.config.hidden_size) return false;
    @memset(out_vector, 0);

    var valid_tokens: usize = 0;
    const H = self.config.hidden_size;

    for (token_ids) |tok| {
        if (tok >= self.config.vocab_size) continue;
        const src = self.embed_tokens[tok * H .. (tok + 1) * H];
        for (out_vector, src) |*dst, s| {
            dst.* += s.toF32();
        }
        valid_tokens += 1;
    }

    if (valid_tokens == 0) return false;

    var sum_sq: f32 = 0.0;
    for (out_vector) |v| sum_sq += v * v;
    const inv = if (sum_sq > 1e-12) 1.0 / @sqrt(sum_sq) else 0.0;
    for (out_vector) |*v| v.* *= inv;
    return true;
}

pub fn searchExplicitMemory(mem: *memory.DiffArchive, query: []const f32, now: u64, out_indices: []usize, top_k: usize) usize {
    return mem.scan(query, now, out_indices, top_k);
}
