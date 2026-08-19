const std = @import("std");

/// Weights for the dual-score associative salience function:
///   Salience(Mi) = alpha*cos(q, Mi) + beta*e^(-lambda*dt) + gamma*||Di||
pub const SalienceConfig = struct {
    alpha: f32 = 1.0,           // associative cosine resonance weight
    beta: f32 = 0.5,            // temporal recency weight
    gamma: f32 = 1.0,           // stored delta magnitude weight
    lambda: f32 = 1.0 / 2048.0, // recency decay per token clock
};

pub const MemoryMeta = struct {
    timestamp: u64,
    last_accessed: u64,
    access_count: u32,
    salience_norm: f32,
    layer_id: u8,
};

/// Bounded, circular, structure-of-arrays archive of salient hidden-state
/// snapshots. Vectors are stored unit-normalized so the associative resonance
/// term is a plain dot product. The NVMe persistence layer (storage.zig) will
/// later serialize this archive to disk.
pub const DiffArchive = struct {
    allocator: std.mem.Allocator,
    dim: usize,
    capacity: usize,
    count: usize,
    write_head: usize,
    config: SalienceConfig,

    vectors: []f32,        // capacity * dim, row-major, unit-normalized
    metas: []MemoryMeta,
    scan_scores: []f32,
    scan_indices: []usize,

    pub fn init(allocator: std.mem.Allocator, dim: usize, capacity: usize, config: SalienceConfig) !DiffArchive {
        const vectors = try allocator.alloc(f32, capacity * dim);
        errdefer allocator.free(vectors);
        const metas = try allocator.alloc(MemoryMeta, capacity);
        errdefer allocator.free(metas);
        const scores = try allocator.alloc(f32, capacity);
        errdefer allocator.free(scores);
        const indices = try allocator.alloc(usize, capacity);
        errdefer allocator.free(indices);

        @memset(vectors, 0);

        return .{
            .allocator = allocator,
            .dim = dim,
            .capacity = capacity,
            .count = 0,
            .write_head = 0,
            .config = config,
            .vectors = vectors,
            .metas = metas,
            .scan_scores = scores,
            .scan_indices = indices,
        };
    }

    pub fn deinit(self: *DiffArchive) void {
        self.allocator.free(self.vectors);
        self.allocator.free(self.metas);
        self.allocator.free(self.scan_scores);
        self.allocator.free(self.scan_indices);
    }

    pub fn reset(self: *DiffArchive) void {
        self.count = 0;
        self.write_head = 0;
    }

    /// Append a hidden-state snapshot, unit-normalized in place. When full, the
    /// oldest entry is overwritten (circular ring).
    pub fn append(self: *DiffArchive, timestamp: u64, salience_norm: f32, layer_id: u8, vector: []const f32) void {
        std.debug.assert(vector.len == self.dim);

        var sum_sq: f32 = 0.0;
        for (vector) |v| sum_sq += v * v;
        const inv = if (sum_sq > 1e-12) 1.0 / @sqrt(sum_sq) else 0.0;

        const idx = self.write_head;
        const row = self.vectors[idx * self.dim .. (idx + 1) * self.dim];
        for (row, vector) |*dst, v| dst.* = v * inv;

        self.metas[idx] = .{
            .timestamp = timestamp,
            .last_accessed = timestamp,
            .access_count = 0,
            .salience_norm = salience_norm,
            .layer_id = layer_id,
        };

        self.write_head = (idx + 1) % self.capacity;
        if (self.count < self.capacity) self.count += 1;
    }

    /// Dual-score top-k associative recall. Returns the number of selected
    /// entries, fills out_indices[0..k] with archive indices, and bumps access
    /// telemetry for the survivors.
    pub fn scan(self: *DiffArchive, query: []const f32, now: u64, out_indices: []usize, top_k: usize) usize {
        const n = self.count;
        if (n == 0) return 0;

        for (0..n) |i| {
            self.scan_scores[i] = self.score(query, now, i);
            self.scan_indices[i] = i;
        }

        const k = @min(top_k, @min(n, out_indices.len));
        for (0..k) |slot| {
            var best = slot;
            for (slot + 1..n) |j| {
                if (self.scan_scores[j] > self.scan_scores[best]) best = j;
            }
            const tmp_score = self.scan_scores[slot];
            self.scan_scores[slot] = self.scan_scores[best];
            self.scan_scores[best] = tmp_score;
            const tmp_idx = self.scan_indices[slot];
            self.scan_indices[slot] = self.scan_indices[best];
            self.scan_indices[best] = tmp_idx;
            out_indices[slot] = self.scan_indices[slot];
        }

        for (out_indices[0..k]) |idx| {
            self.metas[idx].last_accessed = now;
            self.metas[idx].access_count += 1;
        }
        return k;
    }

    fn score(self: *const DiffArchive, query: []const f32, now: u64, i: usize) f32 {
        const meta = self.metas[i];
        const row = self.vectors[i * self.dim .. (i + 1) * self.dim];

        var dot: f32 = 0.0;
        for (query, row) |q, r| dot += q * r;

        const dt = now -| meta.timestamp;
        const recency = @exp(-self.config.lambda * @as(f32, @floatFromInt(dt)));

        return self.config.alpha * dot + self.config.beta * recency + self.config.gamma * meta.salience_norm;
    }
};

test "archive associative recall ranks cosine match above recency" {
    var arch = try DiffArchive.init(std.testing.allocator, 4, 8, .{});
    defer arch.deinit();

    const v0 = [_]f32{ 1, 0, 0, 0 };
    const v1 = [_]f32{ 0, 1, 0, 0 };
    arch.append(0, 0.5, 0, &v0);
    arch.append(1, 0.5, 0, &v1);

    var idx: [4]usize = undefined;
    const query = [_]f32{ 1, 0, 0, 0 };
    const k = arch.scan(&query, 2, &idx, 2);

    try std.testing.expectEqual(@as(usize, 2), k);
    // v0 (cos=1) must outrank v1 (cos=0) despite v1 being more recent.
    try std.testing.expectEqual(@as(usize, 0), idx[0]);
    try std.testing.expectEqual(@as(usize, 1), idx[1]);
}

test "archive wraps circularly and caps count at capacity" {
    var arch = try DiffArchive.init(std.testing.allocator, 2, 3, .{});
    defer arch.deinit();

    const v = [_]f32{ 1, 0 };
    for (0..5) |t| arch.append(@intCast(t), 0.1, 0, &v);

    try std.testing.expectEqual(@as(usize, 3), arch.count);
    try std.testing.expectEqual(@as(usize, 2), arch.write_head);
}
