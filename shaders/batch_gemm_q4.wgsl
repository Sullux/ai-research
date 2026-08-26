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

var<workgroup> s0: array<f32, 128>; var<workgroup> s1: array<f32, 128>;
var<workgroup> s2: array<f32, 128>; var<workgroup> s3: array<f32, 128>;
var<workgroup> s4: array<f32, 128>; var<workgroup> s5: array<f32, 128>;
var<workgroup> s6: array<f32, 128>; var<workgroup> s7: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;      // 0..3 (4 waves = 4 rows)
    let lane = lid.x & 31u;            // 0..31 (lane in wave)

    let row = wgid.x * 4u + local_row;
    let t_base = wgid.y * 8u;
    let t0 = t_base + 0u; let t1 = t_base + 1u; let t2 = t_base + 2u; let t3 = t_base + 3u;
    let t4 = t_base + 4u; let t5 = t_base + 5u; let t6 = t_base + 6u; let t7 = t_base + 7u;

    let valid_row = (row < pc.M);
    let has_t0 = (t0 < pc.N); let has_t1 = (t1 < pc.N); let has_t2 = (t2 < pc.N); let has_t3 = (t3 < pc.N);
    let has_t4 = (t4 < pc.N); let has_t5 = (t5 < pc.N); let has_t6 = (t6 < pc.N); let has_t7 = (t7 < pc.N);

    let blk_in_wave = lane >> 2u;     // 0..7 (which of the 8 blocks)
    let word_in_blk = (lane & 3u) + 1u; // 1..4 (which word in the 32-weight block)
    let vec4_base = (lane & 3u) * 2u;   // 0, 2, 4, 6 (vec4 index within 32-float block)

    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 5u;

    let k_vec4 = pc.K >> 2u;
    let x_vec_off0 = t0 * k_vec4; let x_vec_off1 = t1 * k_vec4;
    let x_vec_off2 = t2 * k_vec4; let x_vec_off3 = t3 * k_vec4;
    let x_vec_off4 = t4 * k_vec4; let x_vec_off5 = t5 * k_vec4;
    let x_vec_off6 = t6 * k_vec4; let x_vec_off7 = t7 * k_vec4;

    var acc0: f32 = 0.0; var acc1: f32 = 0.0; var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    var acc4: f32 = 0.0; var acc5: f32 = 0.0; var acc6: f32 = 0.0; var acc7: f32 = 0.0;

    if (valid_row && has_t0) {
        var base_blk = 0u;
        while (base_blk < num_blocks) {
            let cur_blk = base_blk + blk_in_wave;
            let blk_off = row_word_offset + cur_blk * 5u;

            let sm = unpack2x16float(W[blk_off]);
            let w_packed = W[blk_off + word_in_blk];

            let cur_k_vec = (cur_blk * 32u >> 2u) + vec4_base;

            let n0 = f32(w_packed & 0xFu);
            let n1 = f32((w_packed >> 4u) & 0xFu);
            let n2 = f32((w_packed >> 8u) & 0xFu);
            let n3 = f32((w_packed >> 12u) & 0xFu);
            let n4 = f32((w_packed >> 16u) & 0xFu);
            let n5 = f32((w_packed >> 20u) & 0xFu);
            let n6 = f32((w_packed >> 24u) & 0xFu);
            let n7 = f32(w_packed >> 28u);

            // Token 0
            let v0_a = X[x_vec_off0 + cur_k_vec + 0u];
            let v0_b = X[x_vec_off0 + cur_k_vec + 1u];
            let sum_nx0 = n0 * v0_a.x + n1 * v0_a.y + n2 * v0_a.z + n3 * v0_a.w +
                          n4 * v0_b.x + n5 * v0_b.y + n6 * v0_b.z + n7 * v0_b.w;
            let sum_x0  = (v0_a.x + v0_a.y + v0_a.z + v0_a.w) +
                          (v0_b.x + v0_b.y + v0_b.z + v0_b.w);
            acc0 += (sm.x * sum_nx0 + sm.y * sum_x0);

            if (has_t1) {
                let v1_a = X[x_vec_off1 + cur_k_vec + 0u];
                let v1_b = X[x_vec_off1 + cur_k_vec + 1u];
                let sum_nx1 = n0 * v1_a.x + n1 * v1_a.y + n2 * v1_a.z + n3 * v1_a.w +
                              n4 * v1_b.x + n5 * v1_b.y + n6 * v1_b.z + n7 * v1_b.w;
                let sum_x1  = (v1_a.x + v1_a.y + v1_a.z + v1_a.w) +
                              (v1_b.x + v1_b.y + v1_b.z + v1_b.w);
                acc1 += (sm.x * sum_nx1 + sm.y * sum_x1);
            }
            if (has_t2) {
                let v2_a = X[x_vec_off2 + cur_k_vec + 0u];
                let v2_b = X[x_vec_off2 + cur_k_vec + 1u];
                let sum_nx2 = n0 * v2_a.x + n1 * v2_a.y + n2 * v2_a.z + n3 * v2_a.w +
                              n4 * v2_b.x + n5 * v2_b.y + n6 * v2_b.z + n7 * v2_b.w;
                let sum_x2  = (v2_a.x + v2_a.y + v2_a.z + v2_a.w) +
                              (v2_b.x + v2_b.y + v2_b.z + v2_b.w);
                acc2 += (sm.x * sum_nx2 + sm.y * sum_x2);
            }
            if (has_t3) {
                let v3_a = X[x_vec_off3 + cur_k_vec + 0u];
                let v3_b = X[x_vec_off3 + cur_k_vec + 1u];
                let sum_nx3 = n0 * v3_a.x + n1 * v3_a.y + n2 * v3_a.z + n3 * v3_a.w +
                              n4 * v3_b.x + n5 * v3_b.y + n6 * v3_b.z + n7 * v3_b.w;
                let sum_x3  = (v3_a.x + v3_a.y + v3_a.z + v3_a.w) +
                              (v3_b.x + v3_b.y + v3_b.z + v3_b.w);
                acc3 += (sm.x * sum_nx3 + sm.y * sum_x3);
            }
            if (has_t4) {
                let v4_a = X[x_vec_off4 + cur_k_vec + 0u];
                let v4_b = X[x_vec_off4 + cur_k_vec + 1u];
                let sum_nx4 = n0 * v4_a.x + n1 * v4_a.y + n2 * v4_a.z + n3 * v4_a.w +
                              n4 * v4_b.x + n5 * v4_b.y + n6 * v4_b.z + n7 * v4_b.w;
                let sum_x4  = (v4_a.x + v4_a.y + v4_a.z + v4_a.w) +
                              (v4_b.x + v4_b.y + v4_b.z + v4_b.w);
                acc4 += (sm.x * sum_nx4 + sm.y * sum_x4);
            }
            if (has_t5) {
                let v5_a = X[x_vec_off5 + cur_k_vec + 0u];
                let v5_b = X[x_vec_off5 + cur_k_vec + 1u];
                let sum_nx5 = n0 * v5_a.x + n1 * v5_a.y + n2 * v5_a.z + n3 * v5_a.w +
                              n4 * v5_b.x + n5 * v5_b.y + n6 * v5_b.z + n7 * v5_b.w;
                let sum_x5  = (v5_a.x + v5_a.y + v5_a.z + v5_a.w) +
                              (v5_b.x + v5_b.y + v5_b.z + v5_b.w);
                acc5 += (sm.x * sum_nx5 + sm.y * sum_x5);
            }
            if (has_t6) {
                let v6_a = X[x_vec_off6 + cur_k_vec + 0u];
                let v6_b = X[x_vec_off6 + cur_k_vec + 1u];
                let sum_nx6 = n0 * v6_a.x + n1 * v6_a.y + n2 * v6_a.z + n3 * v6_a.w +
                              n4 * v6_b.x + n5 * v6_b.y + n6 * v6_b.z + n7 * v6_b.w;
                let sum_x6  = (v6_a.x + v6_a.y + v6_a.z + v6_a.w) +
                              (v6_b.x + v6_b.y + v6_b.z + v6_b.w);
                acc6 += (sm.x * sum_nx6 + sm.y * sum_x6);
            }
            if (has_t7) {
                let v7_a = X[x_vec_off7 + cur_k_vec + 0u];
                let v7_b = X[x_vec_off7 + cur_k_vec + 1u];
                let sum_nx7 = n0 * v7_a.x + n1 * v7_a.y + n2 * v7_a.z + n3 * v7_a.w +
                              n4 * v7_b.x + n5 * v7_b.y + n6 * v7_b.z + n7 * v7_b.w;
                let sum_x7  = (v7_a.x + v7_a.y + v7_a.z + v7_a.w) +
                              (v7_b.x + v7_b.y + v7_b.z + v7_b.w);
                acc7 += (sm.x * sum_nx7 + sm.y * sum_x7);
            }

            base_blk += 8u;
        }
    }

    s0[lid.x] = acc0; s1[lid.x] = acc1; s2[lid.x] = acc2; s3[lid.x] = acc3;
    s4[lid.x] = acc4; s5[lid.x] = acc5; s6[lid.x] = acc6; s7[lid.x] = acc7;
    workgroupBarrier();

    let base = local_row * 32u;
    if (lane < 16u) {
        let i16 = base + lane + 16u; let i0 = base + lane;
        s0[i0] += s0[i16]; s1[i0] += s1[i16]; s2[i0] += s2[i16]; s3[i0] += s3[i16];
        s4[i0] += s4[i16]; s5[i0] += s5[i16]; s6[i0] += s6[i16]; s7[i0] += s7[i16];
    }
    workgroupBarrier();
    if (lane < 8u) {
        let i8 = base + lane + 8u; let i0 = base + lane;
        s0[i0] += s0[i8]; s1[i0] += s1[i8]; s2[i0] += s2[i8]; s3[i0] += s3[i8];
        s4[i0] += s4[i8]; s5[i0] += s5[i8]; s6[i0] += s6[i8]; s7[i0] += s7[i8];
    }
    workgroupBarrier();
    if (lane < 4u) {
        let i4 = base + lane + 4u; let i0 = base + lane;
        s0[i0] += s0[i4]; s1[i0] += s1[i4]; s2[i0] += s2[i4]; s3[i0] += s3[i4];
        s4[i0] += s4[i4]; s5[i0] += s5[i4]; s6[i0] += s6[i4]; s7[i0] += s7[i4];
    }
    workgroupBarrier();
    if (lane < 2u) {
        let i2 = base + lane + 2u; let i0 = base + lane;
        s0[i0] += s0[i2]; s1[i0] += s1[i2]; s2[i0] += s2[i2]; s3[i0] += s3[i2];
        s4[i0] += s4[i2]; s5[i0] += s5[i2]; s6[i0] += s6[i2]; s7[i0] += s7[i2];
    }
    workgroupBarrier();
    if (lane < 1u) {
        let i1 = base + lane + 1u; let i0 = base + lane;
        s0[i0] += s0[i1]; s1[i0] += s1[i1]; s2[i0] += s2[i1]; s3[i0] += s3[i1];
        s4[i0] += s4[i1]; s5[i0] += s5[i1]; s6[i0] += s6[i1]; s7[i0] += s7[i1];
    }
    workgroupBarrier();

    if (lane == 0u && valid_row && has_t0) {
        Y[t0 * pc.M + row] = s0[base];
        if (has_t1) { Y[t1 * pc.M + row] = s1[base]; }
        if (has_t2) { Y[t2 * pc.M + row] = s2[base]; }
        if (has_t3) { Y[t3 * pc.M + row] = s3[base]; }
        if (has_t4) { Y[t4 * pc.M + row] = s4[base]; }
        if (has_t5) { Y[t5 * pc.M + row] = s5[base]; }
        if (has_t6) { Y[t6 * pc.M + row] = s6[base]; }
        if (has_t7) { Y[t7 * pc.M + row] = s7[base]; }
    }
}
