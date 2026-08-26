struct PushConstants {
    M: u32,
    K: u32,
    x_offset: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_reduce: array<f32, 256>;

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
    let x_vec4_base = pc.x_offset >> 2u;

    var thread_acc: f32 = 0.0;

    if (row < pc.M) {
        var base_blk = 0u;
        while (base_blk < num_blocks) {
            let cur_blk = base_blk + blk_in_wave;
            let blk_off = row_word_offset + cur_blk * 5u;

            let s = unpack2x16float(W[blk_off]).x;
            let w_packed = W[blk_off + word_in_blk];

            let cur_k_vec = (cur_blk * 32u >> 2u) + vec4_base;

            let n0 = f32(w_packed & 0xFu) - 8.0;
            let n1 = f32((w_packed >> 4u) & 0xFu) - 8.0;
            let n2 = f32((w_packed >> 8u) & 0xFu) - 8.0;
            let n3 = f32((w_packed >> 12u) & 0xFu) - 8.0;
            let n4 = f32((w_packed >> 16u) & 0xFu) - 8.0;
            let n5 = f32((w_packed >> 20u) & 0xFu) - 8.0;
            let n6 = f32((w_packed >> 24u) & 0xFu) - 8.0;
            let n7 = f32(w_packed >> 28u) - 8.0;

            let v_a = X[x_vec4_base + cur_k_vec + 0u];
            let v_b = X[x_vec4_base + cur_k_vec + 1u];

            let sum_nx = n0 * v_a.x + n1 * v_a.y + n2 * v_a.z + n3 * v_a.w +
                         n4 * v_b.x + n5 * v_b.y + n6 * v_b.z + n7 * v_b.w;

            thread_acc += (s * sum_nx);

            base_blk += 8u;
        }
    }

    s_reduce[lid.x] = thread_acc;
    workgroupBarrier();

    let base = local_row * 32u;
    if (lane < 16u) { s_reduce[base + lane] += s_reduce[base + lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u)  { s_reduce[base + lane] += s_reduce[base + lane + 8u];  }
    workgroupBarrier();
    if (lane < 4u)  { s_reduce[base + lane] += s_reduce[base + lane + 4u];  }
    workgroupBarrier();
    if (lane < 2u)  { s_reduce[base + lane] += s_reduce[base + lane + 2u];  }
    workgroupBarrier();
    if (lane < 1u)  { s_reduce[base + lane] += s_reduce[base + lane + 1u];  }
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        Y[row] = s_reduce[base];
    }
}
