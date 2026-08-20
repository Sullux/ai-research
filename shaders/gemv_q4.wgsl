struct PushConstants {
    M: u32,
    K: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;
    let lane = lid.x & 31u;
    let row = wgid.x * 2u + local_row;

    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 5u;
    let lane_word_idx = 1u + (lane >> 3u);
    let nib_shift = (lane & 7u) * 4u;

    var acc: f32 = 0.0;
    if (row < pc.M) {
        for (var b = 0u; b < num_blocks; b = b + 1u) {
            let block_base = row_word_offset + b * 5u;
            let scale_f = bitcast<f32>(W[block_base]);
            let packed_w = W[block_base + lane_word_idx];
            let nib = (packed_w >> nib_shift) & 0x0Fu;
            let weight = (f32(nib) - 8.0) * scale_f;
            let x_val = X[b * 32u + lane];
            acc = acc + weight * x_val;
        }
    }

    sdata[lid.x] = acc;
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        let base_idx = local_row * 32u;
        var sum: f32 = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            sum = sum + sdata[base_idx + i];
        }
        Y[row] = sum;
    }
}
