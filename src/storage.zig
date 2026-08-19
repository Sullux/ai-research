const std = @import("std");
pub const memory = @import("memory.zig");

pub const FILE_MAGIC = [4]u8{ 'S', 'D', 'I', 'F' };
pub const FILE_VERSION: u32 = 1;

pub const FileHeader = extern struct {
    magic: [4]u8,
    version: u32,
    entry_size: u32,
    capacity: u32,
    count: u64,
    write_head: u64,
    dim: u32,
    reserved: [28]u8,
};

pub const DiffHeader = extern struct {
    timestamp: u64,
    last_accessed: u64,
    salience_norm: f32,
    access_count: u32,
    layer_id: u32,
    dim: u32,
};

pub const PersistentDiffStore = struct {
    file: std.fs.File,
    mapped: []align(std.heap.page_size_min) u8,
    capacity: usize,
    dim: usize,
    entry_size: usize,
    file_size: usize,

    pub fn open(path: []const u8, capacity: usize, dim: usize) !PersistentDiffStore {
        const raw_entry_size = @sizeOf(DiffHeader) + dim * @sizeOf(f32);
        const entry_size = std.mem.alignForward(usize, raw_entry_size, 8);
        const required_size = @sizeOf(FileHeader) + capacity * entry_size;

        var file = std.fs.cwd().openFile(path, .{ .mode = .read_write }) catch |err| switch (err) {
            error.FileNotFound => try std.fs.cwd().createFile(path, .{ .read = true, .truncate = false }),
            else => return err,
        };
        errdefer file.close();

        const current_size = try file.getEndPos();
        if (current_size < required_size) try file.setEndPos(required_size);

        const mapped = try std.posix.mmap(null, required_size, std.posix.PROT.READ | std.posix.PROT.WRITE, .{ .TYPE = .SHARED }, file.handle, 0);
        errdefer std.posix.munmap(mapped);

        var store = PersistentDiffStore{
            .file = file,
            .mapped = mapped,
            .capacity = capacity,
            .dim = dim,
            .entry_size = entry_size,
            .file_size = required_size,
        };

        const header = store.getHeader();
        if (!std.mem.eql(u8, &header.magic, &FILE_MAGIC) or header.version != FILE_VERSION or header.dim != dim) {
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
        header.entry_size = @intCast(self.entry_size);
        header.capacity = @intCast(self.capacity);
        header.count = 0;
        header.write_head = 0;
        header.dim = @intCast(self.dim);
        @memset(&header.reserved, 0);
    }

    pub fn append(self: *PersistentDiffStore, timestamp: u64, salience_norm: f32, layer_id: u32, vector: []const f32) void {
        std.debug.assert(vector.len == self.dim);
        const header = self.getHeader();
        const slot: usize = @intCast(header.write_head % header.capacity);
        const offset = @sizeOf(FileHeader) + slot * self.entry_size;

        const diff_hdr: *DiffHeader = @ptrCast(@alignCast(self.mapped.ptr + offset));
        diff_hdr.* = .{
            .timestamp = timestamp,
            .last_accessed = timestamp,
            .salience_norm = salience_norm,
            .access_count = 0,
            .layer_id = layer_id,
            .dim = @intCast(self.dim),
        };
        const data_ptr: [*]f32 = @ptrCast(@alignCast(self.mapped.ptr + offset + @sizeOf(DiffHeader)));
        @memcpy(data_ptr[0..self.dim], vector);

        header.write_head +%= 1;
        header.count +%= 1;
    }

    pub fn getEntry(self: *const PersistentDiffStore, slot: usize) struct { header: *DiffHeader, vector: []const f32 } {
        const offset = @sizeOf(FileHeader) + (slot % self.capacity) * self.entry_size;
        const diff_hdr: *DiffHeader = @ptrCast(@alignCast(self.mapped.ptr + offset));
        const data_ptr: [*]const f32 = @ptrCast(@alignCast(self.mapped.ptr + offset + @sizeOf(DiffHeader)));
        return .{ .header = diff_hdr, .vector = data_ptr[0..self.dim] };
    }

    /// Populate in-memory archive from disk.
    pub fn loadIntoArchive(self: *const PersistentDiffStore, archive: *memory.DiffArchive) usize {
        const header = self.getHeader();
        const total: usize = @intCast(header.count);
        if (total == 0) return 0;

        const available = @min(total, self.capacity);
        const to_load = @min(available, archive.capacity);
        const start_idx = if (total > to_load) total - to_load else 0;

        archive.reset();
        for (start_idx..total) |i| {
            const entry = self.getEntry(i % self.capacity);
            archive.append(entry.header.timestamp, entry.header.salience_norm, @intCast(entry.header.layer_id), entry.vector);
        }
        return archive.count;
    }

    /// Persist all active items from the in-memory archive to disk.
    pub fn saveFromArchive(self: *PersistentDiffStore, archive: *const memory.DiffArchive) void {
        if (archive.count == 0) return;
        const start = if (archive.count == archive.capacity) archive.write_head else 0;
        for (0..archive.count) |i| {
            const slot = (start + i) % archive.capacity;
            const meta = archive.metas[slot];
            const vec = archive.vectors[slot * self.dim .. (slot + 1) * self.dim];
            self.append(meta.timestamp, meta.salience_norm, meta.layer_id, vec);
        }
    }
};

test "persistent diff store binary layout and reopen" {
    const test_path = "/tmp/test_diff_store.sdif";
    std.fs.cwd().deleteFile(test_path) catch {};
    defer std.fs.cwd().deleteFile(test_path) catch {};

    {
        var store = try PersistentDiffStore.open(test_path, 4, 3);
        defer store.close();
        store.append(100, 0.75, 0, &[_]f32{ 1.0, 2.0, 3.0 });
        store.append(101, 0.85, 1, &[_]f32{ 4.0, 5.0, 6.0 });
        try std.testing.expectEqual(@as(u64, 2), store.getHeader().count);
    }
    {
        var store = try PersistentDiffStore.open(test_path, 4, 3);
        defer store.close();
        try std.testing.expectEqual(@as(u64, 2), store.getHeader().count);
        const e0 = store.getEntry(0);
        try std.testing.expectEqual(@as(u64, 100), e0.header.timestamp);
        try std.testing.expectEqual(@as(f32, 0.75), e0.header.salience_norm);
        try std.testing.expectEqual(@as(f32, 1.0), e0.vector[0]);
    }
}

test "persistent store round-trip with DiffArchive" {
    const test_path = "/tmp/test_archive_roundtrip.sdif";
    std.fs.cwd().deleteFile(test_path) catch {};
    defer std.fs.cwd().deleteFile(test_path) catch {};

    var arch1 = try memory.DiffArchive.init(std.testing.allocator, 2, 4, .{});
    defer arch1.deinit();
    arch1.append(10, 0.5, 0, &[_]f32{ 1.0, 0.0 });
    arch1.append(20, 0.9, 0, &[_]f32{ 0.0, 1.0 });

    var store = try PersistentDiffStore.open(test_path, 8, 2);
    defer store.close();
    store.saveFromArchive(&arch1);

    var arch2 = try memory.DiffArchive.init(std.testing.allocator, 2, 4, .{});
    defer arch2.deinit();
    const loaded = store.loadIntoArchive(&arch2);

    try std.testing.expectEqual(@as(usize, 2), loaded);
    try std.testing.expectEqual(@as(u64, 10), arch2.metas[0].timestamp);
    try std.testing.expectEqual(@as(u64, 20), arch2.metas[1].timestamp);
}
