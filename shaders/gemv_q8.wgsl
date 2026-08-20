struct PushConstants {
    M: u32,
    K: u32,
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
    if (row >= pc.M) {
        return;
    }

    let lane = lid.x;
    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 9u;
    let lane_word_idx = 1u + (lane >> 2u);
    let byte_shift = (lane & 3u) * 8u;

    var acc: f32 = 0.0;
    for (var b = 0u; b < num_blocks; b = b + 1u) {
        let block_base = row_word_offset + b * 9u;
        let scale_f = bitcast<f32>(W[block_base]);
        let packed_w = W[block_base + lane_word_idx];
        let raw_byte = (packed_w >> byte_shift) & 0xFFu;
        let signed_byte = (i32(raw_byte) << 24) >> 24;
        let weight = f32(signed_byte) * scale_f;
        let x_val = X[b * 32u + lane];
        acc = acc + weight * x_val;
    }

    sdata[lane] = acc;
    workgroupBarrier();

    if (lane == 0u) {
        var sum: f32 = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            sum = sum + sdata[i];
        }
        Y[row] = sum;
    }
}
