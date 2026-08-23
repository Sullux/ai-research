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
    let row_word_offset = row * num_blocks * 5u;
    let x_offset = t * pc.K;
    let lane_word_idx = 1u + (lane >> 3u);
    let nib_shift = (lane & 7u) * 4u;

    var acc: f32 = 0.0;
    var b = 0u;
    while (b < num_blocks) {
        let blk_off0 = row_word_offset + b * 5u;
        let s0 = bitcast<f32>(W[blk_off0]);
        let packed_word0 = W[blk_off0 + lane_word_idx];
        let nibble0 = (packed_word0 >> nib_shift) & 0x0Fu;
        let w_val0 = (f32(nibble0) - 8.0) * s0;
        let x_val0 = X[x_offset + b * 32u + lane];
        acc = acc + w_val0 * x_val0;

        let blk_off1 = blk_off0 + 5u;
        let s1 = bitcast<f32>(W[blk_off1]);
        let packed_word1 = W[blk_off1 + lane_word_idx];
        let nibble1 = (packed_word1 >> nib_shift) & 0x0Fu;
        let w_val1 = (f32(nibble1) - 8.0) * s1;
        let x_val1 = X[x_offset + (b + 1u) * 32u + lane];
        acc = acc + w_val1 * x_val1;

        let blk_off2 = blk_off0 + 10u;
        let s2 = bitcast<f32>(W[blk_off2]);
        let packed_word2 = W[blk_off2 + lane_word_idx];
        let nibble2 = (packed_word2 >> nib_shift) & 0x0Fu;
        let w_val2 = (f32(nibble2) - 8.0) * s2;
        let x_val2 = X[x_offset + (b + 2u) * 32u + lane];
        acc = acc + w_val2 * x_val2;

        let blk_off3 = blk_off0 + 15u;
        let s3 = bitcast<f32>(W[blk_off3]);
        let packed_word3 = W[blk_off3 + lane_word_idx];
        let nibble3 = (packed_word3 >> nib_shift) & 0x0Fu;
        let w_val3 = (f32(nibble3) - 8.0) * s3;
        let x_val3 = X[x_offset + (b + 3u) * 32u + lane];
        acc = acc + w_val3 * x_val3;

        b = b + 4u;
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
