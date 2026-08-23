struct PushConstants {
    N: u32,
    M: u32,
    K: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wgid.x;
    let t = wgid.y;
    if (row >= pc.M || t >= pc.N) {
        return;
    }
    let lane = lid.x;
    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 9u;
    let x_offset = t * pc.K;
    let lane_word_idx = 1u + (lane >> 2u);
    let byte_shift = (lane & 3u) * 8u;

    var acc: f32 = 0.0;
    var b = 0u;
    while (b < num_blocks) {
        let blk_off0 = row_word_offset + b * 9u;
        let s0 = bitcast<f32>(W[blk_off0]);
        let packed_word0 = W[blk_off0 + lane_word_idx];
        let byte_val0 = (packed_word0 >> byte_shift) & 0xFFu;
        let signed_val0 = f32(bitcast<i32>(byte_val0 << 24u) >> 24);
        let w_val0 = signed_val0 * s0;
        let x_val0 = X[x_offset + b * 32u + lane];
        acc = acc + w_val0 * x_val0;

        let blk_off1 = blk_off0 + 9u;
        let s1 = bitcast<f32>(W[blk_off1]);
        let packed_word1 = W[blk_off1 + lane_word_idx];
        let byte_val1 = (packed_word1 >> byte_shift) & 0xFFu;
        let signed_val1 = f32(bitcast<i32>(byte_val1 << 24u) >> 24);
        let w_val1 = signed_val1 * s1;
        let x_val1 = X[x_offset + (b + 1u) * 32u + lane];
        acc = acc + w_val1 * x_val1;

        b = b + 2u;
    }

    sdata[lane] = acc;
    workgroupBarrier();

    if (lane < 16u) { sdata[lane] = sdata[lane] + sdata[lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { sdata[lane] = sdata[lane] + sdata[lane + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { sdata[lane] = sdata[lane] + sdata[lane + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { sdata[lane] = sdata[lane] + sdata[lane + 2u]; }
    workgroupBarrier();
    if (lane < 1u) { sdata[lane] = sdata[lane] + sdata[lane + 1u]; }
    workgroupBarrier();

    if (lane == 0u) {
        Y[t * pc.M + row] = sdata[0];
    }
}
