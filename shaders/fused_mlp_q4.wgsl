struct PushConstants {
    M: u32,
    K: u32,
};

@group(0) @binding(0) var<storage, read> W_gate: array<u32>;
@group(0) @binding(1) var<storage, read> W_up: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_reduce_gate: array<f32, 256>;
var<workgroup> s_reduce_up: array<f32, 256>;

fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;      // 0..7 (8 rows per workgroup)
    let lane = lid.x & 31u;            // 0..31 (lane in wave)
    let row = wgid.x * 8u + local_row;

    let blk_in_wave = lane >> 2u;     // 0..7 (which of the 8 blocks)
    let word_in_blk = (lane & 3u) + 1u; // 1..4 (which word in the 32-weight block)
    let vec4_base = (lane & 3u) * 2u;   // 0, 2, 4, 6 (vec4 offset in 32-float block)

    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 5u;

    var gate_acc: f32 = 0.0;
    var up_acc: f32 = 0.0;

    if (row < pc.M) {
        var base_blk = 0u;
        while (base_blk < num_blocks) {
            let cur_blk = base_blk + blk_in_wave;
            let blk_off = row_word_offset + cur_blk * 5u;

            let g_s = unpack2x16float(W_gate[blk_off]).x;
            let gw_packed = W_gate[blk_off + word_in_blk];

            let u_s = unpack2x16float(W_up[blk_off]).x;
            let uw_packed = W_up[blk_off + word_in_blk];

            let cur_k_vec = (cur_blk * 32u >> 2u) + vec4_base;

            // Gate nibbles
            let gn0 = f32(gw_packed & 0xFu) - 8.0;
            let gn1 = f32((gw_packed >> 4u) & 0xFu) - 8.0;
            let gn2 = f32((gw_packed >> 8u) & 0xFu) - 8.0;
            let gn3 = f32((gw_packed >> 12u) & 0xFu) - 8.0;
            let gn4 = f32((gw_packed >> 16u) & 0xFu) - 8.0;
            let gn5 = f32((gw_packed >> 20u) & 0xFu) - 8.0;
            let gn6 = f32((gw_packed >> 24u) & 0xFu) - 8.0;
            let gn7 = f32(gw_packed >> 28u) - 8.0;

            // Up nibbles
            let un0 = f32(uw_packed & 0xFu) - 8.0;
            let un1 = f32((uw_packed >> 4u) & 0xFu) - 8.0;
            let un2 = f32((uw_packed >> 8u) & 0xFu) - 8.0;
            let un3 = f32((uw_packed >> 12u) & 0xFu) - 8.0;
            let un4 = f32((uw_packed >> 16u) & 0xFu) - 8.0;
            let un5 = f32((uw_packed >> 20u) & 0xFu) - 8.0;
            let un6 = f32((uw_packed >> 24u) & 0xFu) - 8.0;
            let un7 = f32(uw_packed >> 28u) - 8.0;

            let v_a = X[cur_k_vec + 0u];
            let v_b = X[cur_k_vec + 1u];

            let sum_gn_x = gn0 * v_a.x + gn1 * v_a.y + gn2 * v_a.z + gn3 * v_a.w +
                           gn4 * v_b.x + gn5 * v_b.y + gn6 * v_b.z + gn7 * v_b.w;
            let sum_un_x = un0 * v_a.x + un1 * v_a.y + un2 * v_a.z + un3 * v_a.w +
                           un4 * v_b.x + un5 * v_b.y + un6 * v_b.z + un7 * v_b.w;

            gate_acc += (g_s * sum_gn_x);
            up_acc   += (u_s * sum_un_x);

            base_blk += 8u;
        }
    }

    s_reduce_gate[lid.x] = gate_acc;
    s_reduce_up[lid.x] = up_acc;
    workgroupBarrier();

    let base = local_row * 32u;
    if (lane < 16u) {
        s_reduce_gate[base + lane] += s_reduce_gate[base + lane + 16u];
        s_reduce_up[base + lane] += s_reduce_up[base + lane + 16u];
    }
    workgroupBarrier();
    if (lane < 8u) {
        s_reduce_gate[base + lane] += s_reduce_gate[base + lane + 8u];
        s_reduce_up[base + lane] += s_reduce_up[base + lane + 8u];
    }
    workgroupBarrier();
    if (lane < 4u) {
        s_reduce_gate[base + lane] += s_reduce_gate[base + lane + 4u];
        s_reduce_up[base + lane] += s_reduce_up[base + lane + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        s_reduce_gate[base + lane] += s_reduce_gate[base + lane + 2u];
        s_reduce_up[base + lane] += s_reduce_up[base + lane + 2u];
    }
    workgroupBarrier();
    if (lane < 1u) {
        s_reduce_gate[base + lane] += s_reduce_gate[base + lane + 1u];
        s_reduce_up[base + lane] += s_reduce_up[base + lane + 1u];
    }
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        Y[row] = gelu_tanh(s_reduce_gate[base]) * s_reduce_up[base];
    }
}
