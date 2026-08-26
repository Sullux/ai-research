struct PushConstants {
    N: u32,
    M: u32,
    K: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W_gate: array<u32>;
@group(0) @binding(1) var<storage, read> W_up: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_g0: array<f32, 128>; var<workgroup> s_u0: array<f32, 128>;
var<workgroup> s_g1: array<f32, 128>; var<workgroup> s_u1: array<f32, 128>;
var<workgroup> s_g2: array<f32, 128>; var<workgroup> s_u2: array<f32, 128>;
var<workgroup> s_g3: array<f32, 128>; var<workgroup> s_u3: array<f32, 128>;
var<workgroup> s_g4: array<f32, 128>; var<workgroup> s_u4: array<f32, 128>;
var<workgroup> s_g5: array<f32, 128>; var<workgroup> s_u5: array<f32, 128>;
var<workgroup> s_g6: array<f32, 128>; var<workgroup> s_u6: array<f32, 128>;
var<workgroup> s_g7: array<f32, 128>; var<workgroup> s_u7: array<f32, 128>;

fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

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

    var g0: f32 = 0.0; var u0: f32 = 0.0;
    var g1: f32 = 0.0; var u1: f32 = 0.0;
    var g2: f32 = 0.0; var u2: f32 = 0.0;
    var g3: f32 = 0.0; var u3: f32 = 0.0;
    var g4: f32 = 0.0; var u4: f32 = 0.0;
    var g5: f32 = 0.0; var u5: f32 = 0.0;
    var g6: f32 = 0.0; var u6: f32 = 0.0;
    var g7: f32 = 0.0; var u7: f32 = 0.0;

    if (valid_row && has_t0) {
        var base_blk = 0u;
        while (base_blk < num_blocks) {
            let cur_blk = base_blk + blk_in_wave;
            let blk_off = row_word_offset + cur_blk * 5u;

            let g_sm = unpack2x16float(W_gate[blk_off]);
            let gw_packed = W_gate[blk_off + word_in_blk];

            let u_sm = unpack2x16float(W_up[blk_off]);
            let uw_packed = W_up[blk_off + word_in_blk];

            let cur_k_vec = (cur_blk * 32u >> 2u) + vec4_base;

            // Gate nibbles
            let gn0 = f32(gw_packed & 0xFu);
            let gn1 = f32((gw_packed >> 4u) & 0xFu);
            let gn2 = f32((gw_packed >> 8u) & 0xFu);
            let gn3 = f32((gw_packed >> 12u) & 0xFu);
            let gn4 = f32((gw_packed >> 16u) & 0xFu);
            let gn5 = f32((gw_packed >> 20u) & 0xFu);
            let gn6 = f32((gw_packed >> 24u) & 0xFu);
            let gn7 = f32(gw_packed >> 28u);

            // Up nibbles
            let un0 = f32(uw_packed & 0xFu);
            let un1 = f32((uw_packed >> 4u) & 0xFu);
            let un2 = f32((uw_packed >> 8u) & 0xFu);
            let un3 = f32((uw_packed >> 12u) & 0xFu);
            let un4 = f32((uw_packed >> 16u) & 0xFu);
            let un5 = f32((uw_packed >> 20u) & 0xFu);
            let un6 = f32((uw_packed >> 24u) & 0xFu);
            let un7 = f32(uw_packed >> 28u);

            // Token 0
            let v0_a = X[x_vec_off0 + cur_k_vec + 0u];
            let v0_b = X[x_vec_off0 + cur_k_vec + 1u];
            let sum_gn_x0 = gn0 * v0_a.x + gn1 * v0_a.y + gn2 * v0_a.z + gn3 * v0_a.w +
                            gn4 * v0_b.x + gn5 * v0_b.y + gn6 * v0_b.z + gn7 * v0_b.w;
            let sum_un_x0 = un0 * v0_a.x + un1 * v0_a.y + un2 * v0_a.z + un3 * v0_a.w +
                            un4 * v0_b.x + un5 * v0_b.y + un6 * v0_b.z + un7 * v0_b.w;
            let sum_x0    = (v0_a.x + v0_a.y + v0_a.z + v0_a.w) +
                            (v0_b.x + v0_b.y + v0_b.z + v0_b.w);
            g0 += (g_sm.x * sum_gn_x0 + g_sm.y * sum_x0);
            u0 += (u_sm.x * sum_un_x0 + u_sm.y * sum_x0);

            if (has_t1) {
                let v1_a = X[x_vec_off1 + cur_k_vec + 0u];
                let v1_b = X[x_vec_off1 + cur_k_vec + 1u];
                let sum_gn_x1 = gn0 * v1_a.x + gn1 * v1_a.y + gn2 * v1_a.z + gn3 * v1_a.w +
                                gn4 * v1_b.x + gn5 * v1_b.y + gn6 * v1_b.z + gn7 * v1_b.w;
                let sum_un_x1 = un0 * v1_a.x + un1 * v1_a.y + un2 * v1_a.z + un3 * v1_a.w +
                                un4 * v1_b.x + un5 * v1_b.y + un6 * v1_b.z + un7 * v1_b.w;
                let sum_x1    = (v1_a.x + v1_a.y + v1_a.z + v1_a.w) +
                                (v1_b.x + v1_b.y + v1_b.z + v1_b.w);
                g1 += (g_sm.x * sum_gn_x1 + g_sm.y * sum_x1);
                u1 += (u_sm.x * sum_un_x1 + u_sm.y * sum_x1);
            }
            if (has_t2) {
                let v2_a = X[x_vec_off2 + cur_k_vec + 0u];
                let v2_b = X[x_vec_off2 + cur_k_vec + 1u];
                let sum_gn_x2 = gn0 * v2_a.x + gn1 * v2_a.y + gn2 * v2_a.z + gn3 * v2_a.w +
                                gn4 * v2_b.x + gn5 * v2_b.y + gn6 * v2_b.z + gn7 * v2_b.w;
                let sum_un_x2 = un0 * v2_a.x + un1 * v2_a.y + un2 * v2_a.z + un3 * v2_a.w +
                                un4 * v2_b.x + un5 * v2_b.y + un6 * v2_b.z + un7 * v2_b.w;
                let sum_x2    = (v2_a.x + v2_a.y + v2_a.z + v2_a.w) +
                                (v2_b.x + v2_b.y + v2_b.z + v2_b.w);
                g2 += (g_sm.x * sum_gn_x2 + g_sm.y * sum_x2);
                u2 += (u_sm.x * sum_un_x2 + u_sm.y * sum_x2);
            }
            if (has_t3) {
                let v3_a = X[x_vec_off3 + cur_k_vec + 0u];
                let v3_b = X[x_vec_off3 + cur_k_vec + 1u];
                let sum_gn_x3 = gn0 * v3_a.x + gn1 * v3_a.y + gn2 * v3_a.z + gn3 * v3_a.w +
                                gn4 * v3_b.x + gn5 * v3_b.y + gn6 * v3_b.z + gn7 * v3_b.w;
                let sum_un_x3 = un0 * v3_a.x + un1 * v3_a.y + un2 * v3_a.z + un3 * v3_a.w +
                                un4 * v3_b.x + un5 * v3_b.y + un6 * v3_b.z + un7 * v3_b.w;
                let sum_x3    = (v3_a.x + v3_a.y + v3_a.z + v3_a.w) +
                                (v3_b.x + v3_b.y + v3_b.z + v3_b.w);
                g3 += (g_sm.x * sum_gn_x3 + g_sm.y * sum_x3);
                u3 += (u_sm.x * sum_un_x3 + u_sm.y * sum_x3);
            }
            if (has_t4) {
                let v4_a = X[x_vec_off4 + cur_k_vec + 0u];
                let v4_b = X[x_vec_off4 + cur_k_vec + 1u];
                let sum_gn_x4 = gn0 * v4_a.x + gn1 * v4_a.y + gn2 * v4_a.z + gn3 * v4_a.w +
                                gn4 * v4_b.x + gn5 * v4_b.y + gn6 * v4_b.z + gn7 * v4_b.w;
                let sum_un_x4 = un0 * v4_a.x + un1 * v4_a.y + un2 * v4_a.z + un3 * v4_a.w +
                                un4 * v4_b.x + un5 * v4_b.y + un6 * v4_b.z + un7 * v4_b.w;
                let sum_x4    = (v4_a.x + v4_a.y + v4_a.z + v4_a.w) +
                                (v4_b.x + v4_b.y + v4_b.z + v4_b.w);
                g4 += (g_sm.x * sum_gn_x4 + g_sm.y * sum_x4);
                u4 += (u_sm.x * sum_un_x4 + u_sm.y * sum_x4);
            }
            if (has_t5) {
                let v5_a = X[x_vec_off5 + cur_k_vec + 0u];
                let v5_b = X[x_vec_off5 + cur_k_vec + 1u];
                let sum_gn_x5 = gn0 * v5_a.x + gn1 * v5_a.y + gn2 * v5_a.z + gn3 * v5_a.w +
                                gn4 * v5_b.x + gn5 * v5_b.y + gn6 * v5_b.z + gn7 * v5_b.w;
                let sum_un_x5 = un0 * v5_a.x + un1 * v5_a.y + un2 * v5_a.z + un3 * v5_a.w +
                                un4 * v5_b.x + un5 * v5_b.y + un6 * v5_b.z + un7 * v5_b.w;
                let sum_x5    = (v5_a.x + v5_a.y + v5_a.z + v5_a.w) +
                                (v5_b.x + v5_b.y + v5_b.z + v5_b.w);
                g5 += (g_sm.x * sum_gn_x5 + g_sm.y * sum_x5);
                u5 += (u_sm.x * sum_un_x5 + u_sm.y * sum_x5);
            }
            if (has_t6) {
                let v6_a = X[x_vec_off6 + cur_k_vec + 0u];
                let v6_b = X[x_vec_off6 + cur_k_vec + 1u];
                let sum_gn_x6 = gn0 * v6_a.x + gn1 * v6_a.y + gn2 * v6_a.z + gn3 * v6_a.w +
                                gn4 * v6_b.x + gn5 * v6_b.y + gn6 * v6_b.z + gn7 * v6_b.w;
                let sum_un_x6 = un0 * v6_a.x + un1 * v6_a.y + un2 * v6_a.z + un3 * v6_a.w +
                                un4 * v6_b.x + un5 * v6_b.y + un6 * v6_b.z + un7 * v6_b.w;
                let sum_x6    = (v6_a.x + v6_a.y + v6_a.z + v6_a.w) +
                                (v6_b.x + v6_b.y + v6_b.z + v6_b.w);
                g6 += (g_sm.x * sum_gn_x6 + g_sm.y * sum_x6);
                u6 += (u_sm.x * sum_un_x6 + u_sm.y * sum_x6);
            }
            if (has_t7) {
                let v7_a = X[x_vec_off7 + cur_k_vec + 0u];
                let v7_b = X[x_vec_off7 + cur_k_vec + 1u];
                let sum_gn_x7 = gn0 * v7_a.x + gn1 * v7_a.y + gn2 * v7_a.z + gn3 * v7_a.w +
                                gn4 * v7_b.x + gn5 * v7_b.y + gn6 * v7_b.z + gn7 * v7_b.w;
                let sum_un_x7 = un0 * v7_a.x + un1 * v7_a.y + un2 * v7_a.z + un3 * v7_a.w +
                                un4 * v7_b.x + un5 * v7_b.y + un6 * v7_b.z + un7 * v7_b.w;
                let sum_x7    = (v7_a.x + v7_a.y + v7_a.z + v7_a.w) +
                                (v7_b.x + v7_b.y + v7_b.z + v7_b.w);
                g7 += (g_sm.x * sum_gn_x7 + g_sm.y * sum_x7);
                u7 += (u_sm.x * sum_un_x7 + u_sm.y * sum_x7);
            }

            base_blk += 8u;
        }
    }

    s_g0[lid.x] = g0; s_u0[lid.x] = u0;
    s_g1[lid.x] = g1; s_u1[lid.x] = u1;
    s_g2[lid.x] = g2; s_u2[lid.x] = u2;
    s_g3[lid.x] = g3; s_u3[lid.x] = u3;
    s_g4[lid.x] = g4; s_u4[lid.x] = u4;
    s_g5[lid.x] = g5; s_u5[lid.x] = u5;
    s_g6[lid.x] = g6; s_u6[lid.x] = u6;
    s_g7[lid.x] = g7; s_u7[lid.x] = u7;
    workgroupBarrier();

    let base = local_row * 32u;
    if (lane < 16u) {
        let i16 = base + lane + 16u; let i0 = base + lane;
        s_g0[i0] += s_g0[i16]; s_u0[i0] += s_u0[i16];
        s_g1[i0] += s_g1[i16]; s_u1[i0] += s_u1[i16];
        s_g2[i0] += s_g2[i16]; s_u2[i0] += s_u2[i16];
        s_g3[i0] += s_g3[i16]; s_u3[i0] += s_u3[i16];
        s_g4[i0] += s_g4[i16]; s_u4[i0] += s_u4[i16];
        s_g5[i0] += s_g5[i16]; s_u5[i0] += s_u5[i16];
        s_g6[i0] += s_g6[i16]; s_u6[i0] += s_u6[i16];
        s_g7[i0] += s_g7[i16]; s_u7[i0] += s_u7[i16];
    }
    workgroupBarrier();
    if (lane < 8u) {
        let i8 = base + lane + 8u; let i0 = base + lane;
        s_g0[i0] += s_g0[i8]; s_u0[i0] += s_u0[i8];
        s_g1[i0] += s_g1[i8]; s_u1[i0] += s_u1[i8];
        s_g2[i0] += s_g2[i8]; s_u2[i0] += s_u2[i8];
        s_g3[i0] += s_g3[i8]; s_u3[i0] += s_u3[i8];
        s_g4[i0] += s_g4[i8]; s_u4[i0] += s_u4[i8];
        s_g5[i0] += s_g5[i8]; s_u5[i0] += s_u5[i8];
        s_g6[i0] += s_g6[i8]; s_u6[i0] += s_u6[i8];
        s_g7[i0] += s_g7[i8]; s_u7[i0] += s_u7[i8];
    }
    workgroupBarrier();
    if (lane < 4u) {
        let i4 = base + lane + 4u; let i0 = base + lane;
        s_g0[i0] += s_g0[i4]; s_u0[i0] += s_u0[i4];
        s_g1[i0] += s_g1[i4]; s_u1[i0] += s_u1[i4];
        s_g2[i0] += s_g2[i4]; s_u2[i0] += s_u2[i4];
        s_g3[i0] += s_g3[i4]; s_u3[i0] += s_u3[i4];
        s_g4[i0] += s_g4[i4]; s_u4[i0] += s_u4[i4];
        s_g5[i0] += s_g5[i4]; s_u5[i0] += s_u5[i4];
        s_g6[i0] += s_g6[i4]; s_u6[i0] += s_u6[i4];
        s_g7[i0] += s_g7[i4]; s_u7[i0] += s_u7[i4];
    }
    workgroupBarrier();
    if (lane < 2u) {
        let i2 = base + lane + 2u; let i0 = base + lane;
        s_g0[i0] += s_g0[i2]; s_u0[i0] += s_u0[i2];
        s_g1[i0] += s_g1[i2]; s_u1[i0] += s_u1[i2];
        s_g2[i0] += s_g2[i2]; s_u2[i0] += s_u2[i2];
        s_g3[i0] += s_g3[i2]; s_u3[i0] += s_u3[i2];
        s_g4[i0] += s_g4[i2]; s_u4[i0] += s_u4[i2];
        s_g5[i0] += s_g5[i2]; s_u5[i0] += s_u5[i2];
        s_g6[i0] += s_g6[i2]; s_u6[i0] += s_u6[i2];
        s_g7[i0] += s_g7[i2]; s_u7[i0] += s_u7[i2];
    }
    workgroupBarrier();
    if (lane < 1u) {
        let i1 = base + lane + 1u; let i0 = base + lane;
        s_g0[i0] += s_g0[i1]; s_u0[i0] += s_u0[i1];
        s_g1[i0] += s_g1[i1]; s_u1[i0] += s_u1[i1];
        s_g2[i0] += s_g2[i1]; s_u2[i0] += s_u2[i1];
        s_g3[i0] += s_g3[i1]; s_u3[i0] += s_u3[i1];
        s_g4[i0] += s_g4[i1]; s_u4[i0] += s_u4[i1];
        s_g5[i0] += s_g5[i1]; s_u5[i0] += s_u5[i1];
        s_g6[i0] += s_g6[i1]; s_u6[i0] += s_u6[i1];
        s_g7[i0] += s_g7[i1]; s_u7[i0] += s_u7[i1];
    }
    workgroupBarrier();

    if (lane == 0u && valid_row && has_t0) {
        Y[t0 * pc.M + row] = gelu_tanh(s_g0[base]) * s_u0[base];
        if (has_t1) { Y[t1 * pc.M + row] = gelu_tanh(s_g1[base]) * s_u1[base]; }
        if (has_t2) { Y[t2 * pc.M + row] = gelu_tanh(s_g2[base]) * s_u2[base]; }
        if (has_t3) { Y[t3 * pc.M + row] = gelu_tanh(s_g3[base]) * s_u3[base]; }
        if (has_t4) { Y[t4 * pc.M + row] = gelu_tanh(s_g4[base]) * s_u4[base]; }
        if (has_t5) { Y[t5 * pc.M + row] = gelu_tanh(s_g5[base]) * s_u5[base]; }
        if (has_t6) { Y[t6 * pc.M + row] = gelu_tanh(s_g6[base]) * s_u6[base]; }
        if (has_t7) { Y[t7 * pc.M + row] = gelu_tanh(s_g7[base]) * s_u7[base]; }
    }
}
