const std = @import("std");

pub const DType = enum {
    BF16,
    F16,
    F32,
    I32,
    I16,
    I8,
    U8,
    Unknown,

    pub fn fromString(str: []const u8) DType {
        if (std.mem.eql(u8, str, "BF16")) return .BF16;
        if (std.mem.eql(u8, str, "F16")) return .F16;
        if (std.mem.eql(u8, str, "F32")) return .F32;
        if (std.mem.eql(u8, str, "I32")) return .I32;
        if (std.mem.eql(u8, str, "I16")) return .I16;
        if (std.mem.eql(u8, str, "I8")) return .I8;
        if (std.mem.eql(u8, str, "U8")) return .U8;
        return .Unknown;
    }

    pub fn byteSize(self: DType) usize {
        return switch (self) {
            .BF16, .F16, .I16 => 2,
            .F32, .I32 => 4,
            .I8, .U8 => 1,
            .Unknown => 1,
        };
    }
};

pub const TensorView = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    data: []const u8,

    pub fn asSlice(self: TensorView, comptime T: type) []const T {
        const ptr: [*]const T = @ptrCast(@alignCast(self.data.ptr));
        const len = self.data.len / @sizeOf(T);
        return ptr[0..len];
    }
};

pub const SafeTensors = struct {
    allocator: std.mem.Allocator,
    mapped_data: []align(std.heap.page_size_min) const u8,
    tensors: std.StringHashMap(TensorView),
    file_handle: std.fs.File,

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !SafeTensors {
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        errdefer file.close();

        const file_size = try file.getEndPos();
        if (file_size < 8) return error.InvalidSafeTensorsFile;

        const mapped = try std.posix.mmap(
            null,
            file_size,
            std.posix.PROT.READ,
            .{ .TYPE = .SHARED },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(mapped);

        var st = SafeTensors{
            .allocator = allocator,
            .mapped_data = mapped,
            .tensors = std.StringHashMap(TensorView).init(allocator),
            .file_handle = file,
        };
        errdefer {
            var it = st.tensors.iterator();
            while (it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.shape);
            }
            st.tensors.deinit();
            std.posix.munmap(mapped);
        }

        try st.parseHeader();
        return st;
    }

    pub fn deinit(self: *SafeTensors) void {
        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.shape);
        }
        self.tensors.deinit();
        std.posix.munmap(self.mapped_data);
        self.file_handle.close();
    }

    fn parseHeader(self: *SafeTensors) !void {
        if (self.mapped_data.len < 8) return error.InvalidHeader;
        const header_len = std.mem.readInt(u64, self.mapped_data[0..8], .little);
        const data_start = 8 + header_len;
        if (self.mapped_data.len < data_start) return error.InvalidHeaderLength;

        const header_json = self.mapped_data[8..data_start];
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, header_json, .{});
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.InvalidJsonObject;

        var obj_it = root.object.iterator();
        while (obj_it.next()) |entry| {
            const raw_name = entry.key_ptr.*;
            if (std.mem.eql(u8, raw_name, "__metadata__")) continue;

            const t_obj = entry.value_ptr.*;
            if (t_obj != .object) continue;

            const dtype_val = t_obj.object.get("dtype") orelse continue;
            const shape_val = t_obj.object.get("shape") orelse continue;
            const offsets_val = t_obj.object.get("data_offsets") orelse continue;

            if (dtype_val != .string or shape_val != .array or offsets_val != .array) continue;
            if (offsets_val.array.items.len != 2) continue;

            const offset_begin: usize = @intCast(offsets_val.array.items[0].integer);
            const offset_end: usize = @intCast(offsets_val.array.items[1].integer);

            const byte_start = data_start + offset_begin;
            const byte_end = data_start + offset_end;

            if (byte_end > self.mapped_data.len) return error.TensorOffsetOutOfBounds;

            const raw_slice = self.mapped_data[byte_start..byte_end];

            var shape_list = try self.allocator.alloc(usize, shape_val.array.items.len);
            for (shape_val.array.items, 0..) |s, i| {
                shape_list[i] = @intCast(s.integer);
            }

            const owned_name = try self.allocator.dupe(u8, raw_name);

            const view = TensorView{
                .name = owned_name,
                .dtype = DType.fromString(dtype_val.string),
                .shape = shape_list,
                .data = raw_slice,
            };

            try self.tensors.put(owned_name, view);
        }
    }

    pub fn get(self: *const SafeTensors, name: []const u8) ?TensorView {
        return self.tensors.get(name);
    }
};
