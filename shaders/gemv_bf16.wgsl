struct PushConstants {
    M: u32,
    K: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata: array<f32, 64>;

fn bf16_to_f32(u: u32) -> f32 {
    return bitcast<f32>(u << 16u);
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;
    let lane = lid.x & 31u;
    let row = wgid.x * 2u + local_row;

    let k_words = pc.K / 2u;
    let row_offset = row * k_words;

    var acc: f32 = 0.0;
    if (row < pc.M) {
        var idx = lane * 4u;
        while (idx < k_words) {
            let p0 = W[row_offset + idx];
            let p1 = W[row_offset + idx + 1u];
            let p2 = W[row_offset + idx + 2u];
            let p3 = W[row_offset + idx + 3u];

            let x_base = idx * 2u;
            acc = acc + bf16_to_f32(p0 & 0xFFFFu) * X[x_base] + bf16_to_f32(p0 >> 16u) * X[x_base + 1u];
            acc = acc + bf16_to_f32(p1 & 0xFFFFu) * X[x_base + 2u] + bf16_to_f32(p1 >> 16u) * X[x_base + 3u];
            acc = acc + bf16_to_f32(p2 & 0xFFFFu) * X[x_base + 4u] + bf16_to_f32(p2 >> 16u) * X[x_base + 5u];
            acc = acc + bf16_to_f32(p3 & 0xFFFFu) * X[x_base + 6u] + bf16_to_f32(p3 >> 16u) * X[x_base + 7u];

            idx = idx + 128u;
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
