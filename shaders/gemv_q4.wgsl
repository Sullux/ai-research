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
            let s0 = bitcast<f32>(W[bb0]);
            let p0 = W[bb0 + lane_word_idx];
            let x0 = X[b * 32u + lane];
            let n0 = (p0 >> nib_shift) & 0x0Fu;
            acc = acc + (f32(n0) - 8.0) * s0 * x0;

            let bb1 = bb0 + 5u;
            let s1 = bitcast<f32>(W[bb1]);
            let p1 = W[bb1 + lane_word_idx];
            let x1 = X[(b + 1u) * 32u + lane];
            let n1 = (p1 >> nib_shift) & 0x0Fu;
            acc = acc + (f32(n1) - 8.0) * s1 * x1;

            let bb2 = bb0 + 10u;
            let s2 = bitcast<f32>(W[bb2]);
            let p2 = W[bb2 + lane_word_idx];
            let x2 = X[(b + 2u) * 32u + lane];
            let n2 = (p2 >> nib_shift) & 0x0Fu;
            acc = acc + (f32(n2) - 8.0) * s2 * x2;

            let bb3 = bb0 + 15u;
            let s3 = bitcast<f32>(W[bb3]);
            let p3 = W[bb3 + lane_word_idx];
            let x3 = X[(b + 3u) * 32u + lane];
            let n3 = (p3 >> nib_shift) & 0x0Fu;
            acc = acc + (f32(n3) - 8.0) * s3 * x3;

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

    if (lane == 0u && row < pc.M) {
        Y[row] = sdata[lid.x] + sdata[lid.x + 1u];
    }
}
