struct PushConstants {
    M: u32,
    K: u32,
};

@group(0) @binding(0) var<storage, read> W_gate: array<u32>;
@group(0) @binding(1) var<storage, read> W_up: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata_gate: array<f32, 64>;
var<workgroup> sdata_up: array<f32, 64>;

fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

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

    var gate_acc: f32 = 0.0;
    var up_acc: f32 = 0.0;

    if (row < pc.M) {
        for (var b = 0u; b < num_blocks; b = b + 1u) {
            let block_base = row_word_offset + b * 5u;
            let x_val = X[b * 32u + lane];

            // Gate dequant
            let g_scale = bitcast<f32>(W_gate[block_base]);
            let g_packed = W_gate[block_base + lane_word_idx];
            let g_nib = (g_packed >> nib_shift) & 0x0Fu;
            let g_weight = (f32(g_nib) - 8.0) * g_scale;
            gate_acc = gate_acc + g_weight * x_val;

            // Up dequant
            let u_scale = bitcast<f32>(W_up[block_base]);
            let u_packed = W_up[block_base + lane_word_idx];
            let u_nib = (u_packed >> nib_shift) & 0x0Fu;
            let u_weight = (f32(u_nib) - 8.0) * u_scale;
            up_acc = up_acc + u_weight * x_val;
        }
    }

    sdata_gate[lid.x] = gate_acc;
    sdata_up[lid.x] = up_acc;
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        let base_idx = local_row * 32u;
        var g_sum: f32 = 0.0;
        var u_sum: f32 = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            g_sum = g_sum + sdata_gate[base_idx + i];
            u_sum = u_sum + sdata_up[base_idx + i];
        }
        let act = gelu_tanh(g_sum);
        Y[row] = act * u_sum;
    }
}
