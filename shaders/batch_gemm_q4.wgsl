struct PushConstants {
    N: u32,
    M: u32,
    K: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

// Shared memory for cooperative X loading: 8 tokens x 16 vec4<f32> (64 floats) = 128 vec4
var<workgroup> s_X: array<vec4<f32>, 128>;
// Shared memory for 8-thread reduction per row
var<workgroup> s_reduce: array<f32, 256>;

fn unpack_dot2(w: u32, v_a: vec4<f32>, v_b: vec4<f32>) -> f32 {
    let n0 = f32(w & 0xFu) - 8.0;
    let n1 = f32((w >> 4u) & 0xFu) - 8.0;
    let n2 = f32((w >> 8u) & 0xFu) - 8.0;
    let n3 = f32((w >> 12u) & 0xFu) - 8.0;
    let n4 = f32((w >> 16u) & 0xFu) - 8.0;
    let n5 = f32((w >> 20u) & 0xFu) - 8.0;
    let n6 = f32((w >> 24u) & 0xFu) - 8.0;
    let n7 = f32(w >> 28u) - 8.0;
    return n0 * v_a.x + n1 * v_a.y + n2 * v_a.z + n3 * v_a.w +
           n4 * v_b.x + n5 * v_b.y + n6 * v_b.z + n7 * v_b.w;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row_base = wgid.x * 32u;
    let t_base = wgid.y * 8u;

    let local_row = lid.x >> 3u;      // 0..31 (32 rows per workgroup)
    let lane8 = lid.x & 7u;            // 0..7  (8 threads per row)
    let row = row_base + local_row;

    let num_blocks = pc.K / 32u;
    let k_vec4 = pc.K >> 2u;
    let valid_row = (row < pc.M);

    let row_word_offset = row * num_blocks * 5u;

    let blk_in_k = lane8 >> 2u;       // 0 or 1
    let word_idx = 1u + (lane8 & 3u); // 1..4

    var acc0: f32 = 0.0; var acc1: f32 = 0.0; var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var acc4: f32 = 0.0; var acc5: f32 = 0.0; var acc6: f32 = 0.0; var acc7: f32 = 0.0;

    var k_base_blk = 0u;
    while (k_base_blk < num_blocks) {
        // 1. Cooperative load of 8 tokens x 16 vec4 from X into s_X
        if (lid.x < 128u) {
            let x_tok = lid.x >> 4u;   // 0..7
            let x_vec = lid.x & 15u;   // 0..15
            let t_idx = t_base + x_tok;
            let k_idx_vec4 = (k_base_blk * 8u) + x_vec;
            if (t_idx < pc.N && k_idx_vec4 < k_vec4) {
                s_X[lid.x] = X[t_idx * k_vec4 + k_idx_vec4];
            } else {
                s_X[lid.x] = vec4<f32>(0.0);
            }
        }
        workgroupBarrier();

        // 2. Compute partial dot products across the 8 tokens
        if (valid_row) {
            let cur_blk = k_base_blk + blk_in_k;
            if (cur_blk < num_blocks) {
                let blk_off = row_word_offset + cur_blk * 5u;
                let s = unpack2x16float(W[blk_off]).x;
                let w_packed = W[blk_off + word_idx];

                let vec_a_idx = lane8 * 2u;
                let vec_b_idx = lane8 * 2u + 1u;

                let va0 = s_X[(0u << 4u) + vec_a_idx]; let vb0 = s_X[(0u << 4u) + vec_b_idx];
                let va1 = s_X[(1u << 4u) + vec_a_idx]; let vb1 = s_X[(1u << 4u) + vec_b_idx];
                let va2 = s_X[(2u << 4u) + vec_a_idx]; let vb2 = s_X[(2u << 4u) + vec_b_idx];
                let va3 = s_X[(3u << 4u) + vec_a_idx]; let vb3 = s_X[(3u << 4u) + vec_b_idx];
                let va4 = s_X[(4u << 4u) + vec_a_idx]; let vb4 = s_X[(4u << 4u) + vec_b_idx];
                let va5 = s_X[(5u << 4u) + vec_a_idx]; let vb5 = s_X[(5u << 4u) + vec_b_idx];
                let va6 = s_X[(6u << 4u) + vec_a_idx]; let vb6 = s_X[(6u << 4u) + vec_b_idx];
                let va7 = s_X[(7u << 4u) + vec_a_idx]; let vb7 = s_X[(7u << 4u) + vec_b_idx];

                acc0 += s * unpack_dot2(w_packed, va0, vb0);
                acc1 += s * unpack_dot2(w_packed, va1, vb1);
                acc2 += s * unpack_dot2(w_packed, va2, vb2);
                acc3 += s * unpack_dot2(w_packed, va3, vb3);
                acc4 += s * unpack_dot2(w_packed, va4, vb4);
                acc5 += s * unpack_dot2(w_packed, va5, vb5);
                acc6 += s * unpack_dot2(w_packed, va6, vb6);
                acc7 += s * unpack_dot2(w_packed, va7, vb7);
            }
        }
        workgroupBarrier();

        k_base_blk += 2u;
    }

    if (!valid_row) { return; }

    // 3. Tree reduction across the 8 lanes in each row team
    // Token 0
    s_reduce[lid.x] = acc0;
    workgroupBarrier();
    if (lane8 < 4u) { s_reduce[lid.x] += s_reduce[lid.x + 4u]; }
    if (lane8 < 2u) { s_reduce[lid.x] += s_reduce[lid.x + 2u]; }
    if (lane8 < 1u) { s_reduce[lid.x] += s_reduce[lid.x + 1u]; }
    if (lane8 == 0u && (t_base + 0u) < pc.N) { Y[(t_base + 0u) * pc.M + row] = s_reduce[lid.x]; }
    workgroupBarrier();

    // Token 1
    s_reduce[lid.x] = acc1;
    workgroupBarrier();
    if (lane8 < 4u) { s_reduce[lid.x] += s_reduce[lid.x + 4u]; }
    if (lane8 < 2u) { s_reduce[lid.x] += s_reduce[lid.x + 2u]; }
    if (lane8 < 1u) { s_reduce[lid.x] += s_reduce[lid.x + 1u]; }
    if (lane8 == 0u && (t_base + 1u) < pc.N) { Y[(t_base + 1u) * pc.M + row] = s_reduce[lid.x]; }
    workgroupBarrier();

    // Token 2
    s_reduce[lid.x] = acc2;
    workgroupBarrier();
    if (lane8 < 4u) { s_reduce[lid.x] += s_reduce[lid.x + 4u]; }
    if (lane8 < 2u) { s_reduce[lid.x] += s_reduce[lid.x + 2u]; }
    if (lane8 < 1u) { s_reduce[lid.x] += s_reduce[lid.x + 1u]; }
    if (lane8 == 0u && (t_base + 2u) < pc.N) { Y[(t_base + 2u) * pc.M + row] = s_reduce[lid.x]; }
    workgroupBarrier();

    // Token 3
    s_reduce[lid.x] = acc3;
    workgroupBarrier();
    if (lane8 < 4u) { s_reduce[lid.x] += s_reduce[lid.x + 4u]; }
    if (lane8 < 2u) { s_reduce[lid.x] += s_reduce[lid.x + 2u]; }
    if (lane8 < 1u) { s_reduce[lid.x] += s_reduce[lid.x + 1u]; }
    if (lane8 == 0u && (t_base + 3u) < pc.N) { Y[(t_base + 3u) * pc.M + row] = s_reduce[lid.x]; }
    workgroupBarrier();

    // Token 4
    s_reduce[lid.x] = acc4;
    workgroupBarrier();
    if (lane8 < 4u) { s_reduce[lid.x] += s_reduce[lid.x + 4u]; }
    if (lane8 < 2u) { s_reduce[lid.x] += s_reduce[lid.x + 2u]; }
    if (lane8 < 1u) { s_reduce[lid.x] += s_reduce[lid.x + 1u]; }
    if (lane8 == 0u && (t_base + 4u) < pc.N) { Y[(t_base + 4u) * pc.M + row] = s_reduce[lid.x]; }
    workgroupBarrier();

    // Token 5
    s_reduce[lid.x] = acc5;
    workgroupBarrier();
    if (lane8 < 4u) { s_reduce[lid.x] += s_reduce[lid.x + 4u]; }
    if (lane8 < 2u) { s_reduce[lid.x] += s_reduce[lid.x + 2u]; }
    if (lane8 < 1u) { s_reduce[lid.x] += s_reduce[lid.x + 1u]; }
    if (lane8 == 0u && (t_base + 5u) < pc.N) { Y[(t_base + 5u) * pc.M + row] = s_reduce[lid.x]; }
    workgroupBarrier();

    // Token 6
    s_reduce[lid.x] = acc6;
    workgroupBarrier();
    if (lane8 < 4u) { s_reduce[lid.x] += s_reduce[lid.x + 4u]; }
    if (lane8 < 2u) { s_reduce[lid.x] += s_reduce[lid.x + 2u]; }
    if (lane8 < 1u) { s_reduce[lid.x] += s_reduce[lid.x + 1u]; }
    if (lane8 == 0u && (t_base + 6u) < pc.N) { Y[(t_base + 6u) * pc.M + row] = s_reduce[lid.x]; }
    workgroupBarrier();

    // Token 7
    s_reduce[lid.x] = acc7;
    workgroupBarrier();
    if (lane8 < 4u) { s_reduce[lid.x] += s_reduce[lid.x + 4u]; }
    if (lane8 < 2u) { s_reduce[lid.x] += s_reduce[lid.x + 2u]; }
    if (lane8 < 1u) { s_reduce[lid.x] += s_reduce[lid.x + 1u]; }
    if (lane8 == 0u && (t_base + 7u) < pc.N) { Y[(t_base + 7u) * pc.M + row] = s_reduce[lid.x]; }
}
