struct PushConstants {
    M: u32,
    K: u32,
    x_offset: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_reduce: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;      // 0..3 (4 rows per workgroup)
    let lane = lid.x & 31u;            // 0..31 (lane in wave)
    let row = wgid.x * 4u + local_row;

    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 9u;

    var thread_acc: f32 = 0.0;

    if (row < pc.M) {
        var blk = lane;
        while (blk < num_blocks) {
            let blk_word_off = row_word_offset + blk * 9u;
            let s = bitcast<f32>(W[blk_word_off]);

            let w0 = W[blk_word_off + 1u];
            let w1 = W[blk_word_off + 2u];
            let w2 = W[blk_word_off + 3u];
            let w3 = W[blk_word_off + 4u];
            let w4 = W[blk_word_off + 5u];
            let w5 = W[blk_word_off + 6u];
            let w6 = W[blk_word_off + 7u];
            let w7 = W[blk_word_off + 8u];

            let x_base = pc.x_offset + blk * 32u;

            var sum_wx: f32 = 0.0;

            // Word 0 (bytes 0..3)
            var cur = w0;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let sb = f32((i32(cur & 0xFFu) << 24) >> 24);
                sum_wx = sum_wx + sb * X[x_base + j];
                cur = cur >> 8u;
            }
            // Word 1 (bytes 4..7)
            cur = w1;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let sb = f32((i32(cur & 0xFFu) << 24) >> 24);
                sum_wx = sum_wx + sb * X[x_base + 4u + j];
                cur = cur >> 8u;
            }
            // Word 2 (bytes 8..11)
            cur = w2;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let sb = f32((i32(cur & 0xFFu) << 24) >> 24);
                sum_wx = sum_wx + sb * X[x_base + 8u + j];
                cur = cur >> 8u;
            }
            // Word 3 (bytes 12..15)
            cur = w3;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let sb = f32((i32(cur & 0xFFu) << 24) >> 24);
                sum_wx = sum_wx + sb * X[x_base + 12u + j];
                cur = cur >> 8u;
            }
            // Word 4 (bytes 16..19)
            cur = w4;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let sb = f32((i32(cur & 0xFFu) << 24) >> 24);
                sum_wx = sum_wx + sb * X[x_base + 16u + j];
                cur = cur >> 8u;
            }
            // Word 5 (bytes 20..23)
            cur = w5;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let sb = f32((i32(cur & 0xFFu) << 24) >> 24);
                sum_wx = sum_wx + sb * X[x_base + 20u + j];
                cur = cur >> 8u;
            }
            // Word 6 (bytes 24..27)
            cur = w6;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let sb = f32((i32(cur & 0xFFu) << 24) >> 24);
                sum_wx = sum_wx + sb * X[x_base + 24u + j];
                cur = cur >> 8u;
            }
            // Word 7 (bytes 28..31)
            cur = w7;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let sb = f32((i32(cur & 0xFFu) << 24) >> 24);
                sum_wx = sum_wx + sb * X[x_base + 28u + j];
                cur = cur >> 8u;
            }

            thread_acc = thread_acc + s * sum_wx;
            blk = blk + 32u;
        }
    }

    s_reduce[lid.x] = thread_acc;
    workgroupBarrier();

    let base = local_row * 32u;
    if (lane < 16u) { s_reduce[base + lane] = s_reduce[base + lane] + s_reduce[base + lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { s_reduce[base + lane] = s_reduce[base + lane] + s_reduce[base + lane + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { s_reduce[base + lane] = s_reduce[base + lane] + s_reduce[base + lane + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { s_reduce[base + lane] = s_reduce[base + lane] + s_reduce[base + lane + 2u]; }
    workgroupBarrier();
    if (lane < 1u) { s_reduce[base + lane] = s_reduce[base + lane] + s_reduce[base + lane + 1u]; }
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        Y[row] = s_reduce[base];
    }
}
