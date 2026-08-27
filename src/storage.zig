const std = @import("std");
pub const memory = @import("memory.zig");

pub const FILE_MAGIC = [4]u8{ 'E', 'M', 'E', 'M' };
pub const FILE_VERSION: u32 = 1;

pub const FileFlags = struct {
    pub const TENSOR_FP16: u32 = 1 << 0;
    pub const TENSOR_Q8: u32 = 1 << 1;
    pub const COMPRESSED: u32 = 1 << 2;
};

pub const EpisodeFlags = struct {
    pub const IS_INTERRUPTED: u16 = 1 << 0;
    pub const HAS_TOOL_CALL: u16 = 1 << 1;
    pub const PINNED: u16 = 1 << 2;
    pub const SLAB_PRUNED: u16 = 1 << 3;
};

pub const FileHeader = extern struct {
    magic: [4]u8,
    version: u32,
    file_flags: u32,
    num_layers: u32,
    hidden_size: u32,
    kv_dim: u32,
    max_episodes: u32,
    reserved_pad: u32 = 0,
    total_episodes: u64,
    write_head: u64,
    model_id_hash: u64,
    reserved: [8]u8,
};

pub const EpisodeHeader = extern struct {
    episode_id: u64,
    parent_episode_id: u64,
    created_timestamp: u64,
    last_accessed: u64,
    start_clock: u64,
    token_count: u32,
    access_count: u32,
    child_count: u32,
    salience_norm: f32,
    continuation_token: u32,
    flags: u16,
    summary_len: u16,
};

comptime {
    std.debug.assert(@sizeOf(FileHeader) == 64);
    std.debug.assert(@sizeOf(EpisodeHeader) == 64);
}

pub const PersistentDiffStore = struct {
    file: std.fs.File,
    mapped: []align(std.heap.page_size_min) u8,
    max_episodes: usize,
    num_layers: usize,
    hidden_size: usize,
    kv_dim: usize,
    max_tokens_per_episode: usize,
    entry_size: usize,
    file_size: usize,

    pub fn open(
        path: []const u8,
        max_episodes: usize,
        num_layers: usize,
        hidden_size: usize,
        kv_dim: usize,
        max_tokens_per_episode: usize,
    ) !PersistentDiffStore {
        const centroid_bytes = hidden_size * @sizeOf(f16);
        const max_summary_bytes = 256;
        const max_kv_bytes = num_layers * 2 * max_tokens_per_episode * kv_dim * @sizeOf(f16);
        const raw_entry_size = @sizeOf(EpisodeHeader) + centroid_bytes + max_summary_bytes + max_kv_bytes;
        const entry_size = std.mem.alignForward(usize, raw_entry_size, 64);
        const required_size = @sizeOf(FileHeader) + max_episodes * entry_size;

        var file = std.fs.cwd().openFile(path, .{ .mode = .read_write }) catch |err| switch (err) {
            error.FileNotFound => try std.fs.cwd().createFile(path, .{ .read = true, .truncate = false }),
            else => return err,
        };
        errdefer file.close();

        const current_size = try file.getEndPos();
        if (current_size < required_size) try file.setEndPos(required_size);

        const mapped = try std.posix.mmap(
            null,
            required_size,
            std.posix.PROT.READ | std.posix.PROT.WRITE,
            .{ .TYPE = .SHARED },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(mapped);

        var store = PersistentDiffStore{
            .file = file,
            .mapped = mapped,
            .max_episodes = max_episodes,
            .num_layers = num_layers,
            .hidden_size = hidden_size,
            .kv_dim = kv_dim,
            .max_tokens_per_episode = max_tokens_per_episode,
            .entry_size = entry_size,
            .file_size = required_size,
        };

        const header = store.getHeader();
        if (!std.mem.eql(u8, &header.magic, &FILE_MAGIC) or
            header.version != FILE_VERSION or
            header.hidden_size != hidden_size or
            header.num_layers != num_layers)
        {
            store.initFreshHeader();
        }
        return store;
    }

    pub fn close(self: *PersistentDiffStore) void {
        std.posix.munmap(self.mapped);
        self.file.close();
    }

    pub fn getHeader(self: *const PersistentDiffStore) *FileHeader {
        return @ptrCast(@alignCast(self.mapped.ptr));
    }

    fn initFreshHeader(self: *PersistentDiffStore) void {
        const header = self.getHeader();
        header.magic = FILE_MAGIC;
        header.version = FILE_VERSION;
        header.file_flags = FileFlags.TENSOR_FP16;
        header.num_layers = @intCast(self.num_layers);
        header.hidden_size = @intCast(self.hidden_size);
        header.kv_dim = @intCast(self.kv_dim);
        header.max_episodes = @intCast(self.max_episodes);
        header.total_episodes = 0;
        header.write_head = 0;
        header.model_id_hash = 0;
        @memset(&header.reserved, 0);
    }

    fn entryOffset(self: *const PersistentDiffStore, slot: usize) usize {
        return @sizeOf(FileHeader) + (slot % self.max_episodes) * self.entry_size;
    }

    pub fn appendEpisode(
        self: *PersistentDiffStore,
        meta: EpisodeHeader,
        centroid: []const f32,
        summary: []const u8,
        kv_data_f32: ?[]const f32,
    ) void {
        const header = self.getHeader();
        const slot: usize = @intCast(header.write_head % header.max_episodes);
        const base = self.entryOffset(slot);

        const ep_hdr: *EpisodeHeader = @ptrCast(@alignCast(self.mapped.ptr + base));
        ep_hdr.* = meta;
        ep_hdr.summary_len = @min(@as(u16, @intCast(summary.len)), 256);

        var offset = base + @sizeOf(EpisodeHeader);
        const vec_ptr: [*]f16 = @ptrCast(@alignCast(self.mapped.ptr + offset));
        for (0..@min(centroid.len, self.hidden_size)) |i| {
            vec_ptr[i] = @floatCast(centroid[i]);
        }
        offset += self.hidden_size * @sizeOf(f16);

        if (ep_hdr.summary_len > 0) {
            @memcpy(self.mapped[offset .. offset + ep_hdr.summary_len], summary[0..ep_hdr.summary_len]);
        }
        offset += 256;

        if (kv_data_f32) |kv| {
            const num_elems = @min(kv.len, self.num_layers * 2 * meta.token_count * self.kv_dim);
            const kv_ptr: [*]f16 = @ptrCast(@alignCast(self.mapped.ptr + offset));
            for (0..num_elems) |i| kv_ptr[i] = @floatCast(kv[i]);
        }

        header.write_head +%= 1;
        header.total_episodes +%= 1;
    }

    pub fn getEpisodeHeader(self: *const PersistentDiffStore, slot: usize) *EpisodeHeader {
        const base = self.entryOffset(slot);
        return @ptrCast(@alignCast(self.mapped.ptr + base));
    }

    pub fn getEpisodeCentroid(self: *const PersistentDiffStore, slot: usize, out_vec: []f32) void {
        const base = self.entryOffset(slot);
        const offset = base + @sizeOf(EpisodeHeader);
        const vec_ptr: [*]const f16 = @ptrCast(@alignCast(self.mapped.ptr + offset));
        for (0..@min(out_vec.len, self.hidden_size)) |i| {
            out_vec[i] = @floatCast(vec_ptr[i]);
        }
    }

    pub fn getEpisodeKVSlab(self: *const PersistentDiffStore, slot: usize, out_kv: []f32) usize {
        const ep_hdr = self.getEpisodeHeader(slot);
        if (ep_hdr.flags & EpisodeFlags.SLAB_PRUNED != 0 or ep_hdr.token_count == 0) return 0;
        const base = self.entryOffset(slot);
        const offset = base + @sizeOf(EpisodeHeader) + self.hidden_size * @sizeOf(f16) + 256;
        const total_elems = self.num_layers * 2 * ep_hdr.token_count * self.kv_dim;
        const to_copy = @min(out_kv.len, total_elems);
        const kv_ptr: [*]const f16 = @ptrCast(@alignCast(self.mapped.ptr + offset));
        for (0..to_copy) |i| out_kv[i] = @floatCast(kv_ptr[i]);
        return to_copy;
    }

    pub fn loadIntoArchive(self: *const PersistentDiffStore, archive: *memory.DiffArchive) usize {
        const header = self.getHeader();
        const total: usize = @intCast(header.total_episodes);
        if (total == 0) return 0;

        const available = @min(total, self.max_episodes);
        const to_load = @min(available, archive.capacity);
        const start_idx = if (total > to_load) total - to_load else 0;

        archive.reset();
        const temp_vec = archive.allocator.alloc(f32, self.hidden_size) catch return 0;
        defer archive.allocator.free(temp_vec);

        for (start_idx..total) |i| {
            const slot = i % self.max_episodes;
            const ep = self.getEpisodeHeader(slot);
            self.getEpisodeCentroid(slot, temp_vec);
            archive.appendWithMeta(
                ep.created_timestamp,
                ep.salience_norm,
                0,
                ep.flags & EpisodeFlags.IS_INTERRUPTED != 0,
                ep.continuation_token,
                temp_vec,
            );
        }
        return archive.count;
    }

    pub fn saveFromArchive(self: *PersistentDiffStore, archive: *const memory.DiffArchive) void {
        if (archive.count == 0) return;
        const start = if (archive.count == archive.capacity) archive.write_head else 0;
        for (0..archive.count) |i| {
            const slot = (start + i) % archive.capacity;
            const meta = archive.metas[slot];
            const vec = archive.vectors[slot * self.hidden_size .. (slot + 1) * self.hidden_size];
            const ep_meta = EpisodeHeader{
                .episode_id = @intCast(self.getHeader().total_episodes + 1),
                .parent_episode_id = 0,
                .created_timestamp = meta.timestamp,
                .last_accessed = meta.last_accessed,
                .start_clock = 0,
                .token_count = 1,
                .access_count = meta.access_count,
                .child_count = 0,
                .salience_norm = meta.salience_norm,
                .continuation_token = meta.token_id,
                .flags = if (meta.is_interrupted) EpisodeFlags.IS_INTERRUPTED else 0,
                .summary_len = 0,
            };
            self.appendEpisode(ep_meta, vec, "", null);
        }
    }
};

test "persistent episodic store binary layout and reopen" {
    const test_path = "/tmp/test_episodic_store.mem";
    std.fs.cwd().deleteFile(test_path) catch {};
    defer std.fs.cwd().deleteFile(test_path) catch {};

    {
        var store = try PersistentDiffStore.open(test_path, 4, 2, 4, 2, 2);
        defer store.close();
        const ep1 = EpisodeHeader{
            .episode_id = 1,
            .parent_episode_id = 0,
            .created_timestamp = 1000,
            .last_accessed = 1000,
            .start_clock = 0,
            .token_count = 2,
            .access_count = 0,
            .child_count = 1,
            .salience_norm = 0.75,
            .continuation_token = 42,
            .flags = 0,
            .summary_len = 5,
        };
        const centroid1 = [_]f32{ 1.0, -0.5, 0.25, -0.125 };
        const summary1 = "hello";
        const kv1 = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6 };
        store.appendEpisode(ep1, &centroid1, summary1, &kv1);
        try std.testing.expectEqual(@as(u64, 1), store.getHeader().total_episodes);
    }
    {
        var store = try PersistentDiffStore.open(test_path, 4, 2, 4, 2, 2);
        defer store.close();
        try std.testing.expectEqual(@as(u64, 1), store.getHeader().total_episodes);
        const ep = store.getEpisodeHeader(0);
        try std.testing.expectEqual(@as(u64, 1), ep.episode_id);
        try std.testing.expectEqual(@as(u64, 1000), ep.created_timestamp);
        try std.testing.expectEqual(@as(u32, 2), ep.token_count);
        try std.testing.expectEqual(@as(u32, 1), ep.child_count);

        var c_out: [4]f32 = undefined;
        store.getEpisodeCentroid(0, &c_out);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), c_out[0], 0.01);
        try std.testing.expectApproxEqAbs(@as(f32, -0.5), c_out[1], 0.01);

        var kv_out: [16]f32 = undefined;
        const copied = store.getEpisodeKVSlab(0, &kv_out);
        try std.testing.expectEqual(@as(usize, 16), copied);
        try std.testing.expectApproxEqAbs(@as(f32, 0.1), kv_out[0], 0.01);
    }
}

test "persistent store round-trip with DiffArchive" {
    const test_path = "/tmp/test_archive_roundtrip.mem";
    std.fs.cwd().deleteFile(test_path) catch {};
    defer std.fs.cwd().deleteFile(test_path) catch {};

    var arch1 = try memory.DiffArchive.init(std.testing.allocator, 4, 4, .{});
    defer arch1.deinit();
    arch1.append(1000, 0.8, 0, &[_]f32{ 1.0, 0.0, 0.0, 0.0 });
    arch1.append(2000, 0.9, 0, &[_]f32{ 0.0, 1.0, 0.0, 0.0 });

    var store = try PersistentDiffStore.open(test_path, 8, 1, 4, 2, 2);
    defer store.close();
    store.saveFromArchive(&arch1);

    var arch2 = try memory.DiffArchive.init(std.testing.allocator, 4, 4, .{});
    defer arch2.deinit();
    const loaded = store.loadIntoArchive(&arch2);

    try std.testing.expectEqual(@as(usize, 2), loaded);
    try std.testing.expectEqual(@as(u64, 1000), arch2.metas[0].timestamp);
    try std.testing.expectEqual(@as(u64, 2000), arch2.metas[1].timestamp);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), arch2.metas[0].salience_norm, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), arch2.metas[1].salience_norm, 0.01);
}
