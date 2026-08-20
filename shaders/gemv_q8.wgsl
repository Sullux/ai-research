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
    let row_word_offset = row * num_blocks * 9u;
    let lane_word_idx = 1u + (lane >> 2u);
    let byte_shift = (lane & 3u) * 8u;

    var acc: f32 = 0.0;
    if (row < pc.M) {
        var b = 0u;
        while (b < num_blocks) {
            let bb0 = row_word_offset + b * 9u;
            let s0 = bitcast<f32>(W[bb0]);
            let p0 = W[bb0 + lane_word_idx];
            let raw0 = (p0 >> byte_shift) & 0xFFu;
            let sb0 = (i32(raw0) << 24) >> 24;
            let x0 = X[b * 32u + lane];
            acc = acc + f32(sb0) * s0 * x0;

            let bb1 = bb0 + 9u;
            let s1 = bitcast<f32>(W[bb1]);
            let p1 = W[bb1 + lane_word_idx];
            let raw1 = (p1 >> byte_shift) & 0xFFu;
            let sb1 = (i32(raw1) << 24) >> 24;
            let x1 = X[(b + 1u) * 32u + lane];
            acc = acc + f32(sb1) * s1 * x1;

            let bb2 = bb0 + 18u;
            let s2 = bitcast<f32>(W[bb2]);
            let p2 = W[bb2 + lane_word_idx];
            let raw2 = (p2 >> byte_shift) & 0xFFu;
            let sb2 = (i32(raw2) << 24) >> 24;
            let x2 = X[(b + 2u) * 32u + lane];
            acc = acc + f32(sb2) * s2 * x2;

            let bb3 = bb0 + 27u;
            let s3 = bitcast<f32>(W[bb3]);
            let p3 = W[bb3 + lane_word_idx];
            let raw3 = (p3 >> byte_shift) & 0xFFu;
            let sb3 = (i32(raw3) << 24) >> 24;
            let x3 = X[(b + 3u) * 32u + lane];
            acc = acc + f32(sb3) * s3 * x3;

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
