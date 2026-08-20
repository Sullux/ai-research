struct PushConstants {
    M: u32,
    K: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata: array<f32, 32>;

fn bf16_to_f32(u: u32) -> f32 {
    return bitcast<f32>(u << 16u);
}

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
    let k_words = pc.K / 2u;
    let row_offset = row * k_words;

    var acc: f32 = 0.0;
    var idx = lane;
    while (idx < k_words) {
        let packed_w = W[row_offset + idx];
        let w0 = bf16_to_f32(packed_w & 0xFFFFu);
        let w1 = bf16_to_f32(packed_w >> 16u);
        let x0 = X[idx * 2u];
        let x1 = X[idx * 2u + 1u];
        acc = acc + w0 * x0 + w1 * x1;
        idx = idx + 32u;
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
