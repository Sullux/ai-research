struct PushConstants {
    M: u32,
    K: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;
    let lane = lid.x & 31u;
    let row = wgid.x * 4u + local_row;

    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 5u;
    let lane_word_idx = 1u + (lane >> 3u);
    let nib_shift = (lane & 7u) * 4u;

    var acc: f32 = 0.0;
    if (row < pc.M) {
        var b = 0u;
        while (b < num_blocks) {
            let bb0 = row_word_offset + b * 5u;
            let scale_f0 = bitcast<f32>(W[bb0]);
            let packed_w0 = W[bb0 + lane_word_idx];
            let nibble0 = (packed_w0 >> nib_shift) & 0x0Fu;
            let weight0 = (f32(nibble0) - 8.0) * scale_f0;
            let x_val0 = X[b * 32u + lane];
            acc = acc + weight0 * x_val0;

            let bb1 = bb0 + 5u;
            let scale_f1 = bitcast<f32>(W[bb1]);
            let packed_w1 = W[bb1 + lane_word_idx];
            let nibble1 = (packed_w1 >> nib_shift) & 0x0Fu;
            let weight1 = (f32(nibble1) - 8.0) * scale_f1;
            let x_val1 = X[(b + 1u) * 32u + lane];
            acc = acc + weight1 * x_val1;

            let bb2 = bb0 + 10u;
            let scale_f2 = bitcast<f32>(W[bb2]);
            let packed_w2 = W[bb2 + lane_word_idx];
            let nibble2 = (packed_w2 >> nib_shift) & 0x0Fu;
            let weight2 = (f32(nibble2) - 8.0) * scale_f2;
            let x_val2 = X[(b + 2u) * 32u + lane];
            acc = acc + weight2 * x_val2;

            let bb3 = bb0 + 15u;
            let scale_f3 = bitcast<f32>(W[bb3]);
            let packed_w3 = W[bb3 + lane_word_idx];
            let nibble3 = (packed_w3 >> nib_shift) & 0x0Fu;
            let weight3 = (f32(nibble3) - 8.0) * scale_f3;
            let x_val3 = X[(b + 3u) * 32u + lane];
            acc = acc + weight3 * x_val3;

            b = b + 4u;
        }
    }

    sdata[lid.x] = acc;
    workgroupBarrier();

    if (lane < 16u) { sdata[lid.x] = sdata[lid.x] + sdata[lid.x + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { sdata[lid.x] = sdata[lid.x] + sdata[lid.x + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { sdata[lid.x] = sdata[lid.x] + sdata[lid.x + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { sdata[lid.x] = sdata[lid.x] + sdata[lid.x + 2u]; }
    workgroupBarrier();
    if (lane < 1u) { sdata[lid.x] = sdata[lid.x] + sdata[lid.x + 1u]; }
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        Y[row] = sdata[lid.x];
    }
}
