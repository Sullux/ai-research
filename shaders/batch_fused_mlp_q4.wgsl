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

var<workgroup> s_gate0: array<f32, 256>; var<workgroup> s_up0: array<f32, 256>;
var<workgroup> s_gate1: array<f32, 256>; var<workgroup> s_up1: array<f32, 256>;
var<workgroup> s_gate2: array<f32, 256>; var<workgroup> s_up2: array<f32, 256>;
var<workgroup> s_gate3: array<f32, 256>; var<workgroup> s_up3: array<f32, 256>;
var<workgroup> s_gate4: array<f32, 256>; var<workgroup> s_up4: array<f32, 256>;
var<workgroup> s_gate5: array<f32, 256>; var<workgroup> s_up5: array<f32, 256>;
var<workgroup> s_gate6: array<f32, 256>; var<workgroup> s_up6: array<f32, 256>;
var<workgroup> s_gate7: array<f32, 256>; var<workgroup> s_up7: array<f32, 256>;

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
    let t_base = wgid.y * 8u;

    let blk_in_wave = lane >> 2u;     // 0..7 (which of the 8 blocks)
    let word_in_blk = (lane & 3u) + 1u; // 1..4 (which word in the 32-weight block)
    let vec4_base = (lane & 3u) * 2u;   // 0, 2, 4, 6 (vec4 offset in 32-float block)

    let num_blocks = pc.K / 32u;
    let valid_row = (row < pc.M);

    let t0 = t_base + 0u; let t1 = t_base + 1u; let t2 = t_base + 2u; let t3 = t_base + 3u;
    let t4 = t_base + 4u; let t5 = t_base + 5u; let t6 = t_base + 6u; let t7 = t_base + 7u;

    let has_t0 = (t0 < pc.N); let has_t1 = (t1 < pc.N); let has_t2 = (t2 < pc.N); let has_t3 = (t3 < pc.N);
    let has_t4 = (t4 < pc.N); let has_t5 = (t5 < pc.N); let has_t6 = (t6 < pc.N); let has_t7 = (t7 < pc.N);

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

            // Token 0
            let v0_a = X[x_vec_off0 + cur_k_vec + 0u];
            let v0_b = X[x_vec_off0 + cur_k_vec + 1u];
            let sum_gn0 = gn0 * v0_a.x + gn1 * v0_a.y + gn2 * v0_a.z + gn3 * v0_a.w +
                          gn4 * v0_b.x + gn5 * v0_b.y + gn6 * v0_b.z + gn7 * v0_b.w;
            let sum_un0 = un0 * v0_a.x + un1 * v0_a.y + un2 * v0_a.z + un3 * v0_a.w +
                          un4 * v0_b.x + un5 * v0_b.y + un6 * v0_b.z + un7 * v0_b.w;
            g0 += (g_s * sum_gn0);
            u0 += (u_s * sum_un0);

            if (has_t1) {
                let v1_a = X[x_vec_off1 + cur_k_vec + 0u];
                let v1_b = X[x_vec_off1 + cur_k_vec + 1u];
                let sum_gn1 = gn0 * v1_a.x + gn1 * v1_a.y + gn2 * v1_a.z + gn3 * v1_a.w +
                              gn4 * v1_b.x + gn5 * v1_b.y + gn6 * v1_b.z + gn7 * v1_b.w;
                let sum_un1 = un0 * v1_a.x + un1 * v1_a.y + un2 * v1_a.z + un3 * v1_a.w +
                              un4 * v1_b.x + un5 * v1_b.y + un6 * v1_b.z + un7 * v1_b.w;
                g1 += (g_s * sum_gn1);
                u1 += (u_s * sum_un1);
            }
            if (has_t2) {
                let v2_a = X[x_vec_off2 + cur_k_vec + 0u];
                let v2_b = X[x_vec_off2 + cur_k_vec + 1u];
                let sum_gn2 = gn0 * v2_a.x + gn1 * v2_a.y + gn2 * v2_a.z + gn3 * v2_a.w +
                              gn4 * v2_b.x + gn5 * v2_b.y + gn6 * v2_b.z + gn7 * v2_b.w;
                let sum_un2 = un0 * v2_a.x + un1 * v2_a.y + un2 * v2_a.z + un3 * v2_a.w +
                              un4 * v2_b.x + un5 * v2_b.y + un6 * v2_b.z + un7 * v2_b.w;
                g2 += (g_s * sum_gn2);
                u2 += (u_s * sum_un2);
            }
            if (has_t3) {
                let v3_a = X[x_vec_off3 + cur_k_vec + 0u];
                let v3_b = X[x_vec_off3 + cur_k_vec + 1u];
                let sum_gn3 = gn0 * v3_a.x + gn1 * v3_a.y + gn2 * v3_a.z + gn3 * v3_a.w +
                              gn4 * v3_b.x + gn5 * v3_b.y + gn6 * v3_b.z + gn7 * v3_b.w;
                let sum_un3 = un0 * v3_a.x + un1 * v3_a.y + un2 * v3_a.z + un3 * v3_a.w +
                              un4 * v3_b.x + un5 * v3_b.y + un6 * v3_b.z + un7 * v3_b.w;
                g3 += (g_s * sum_gn3);
                u3 += (u_s * sum_un3);
            }
            if (has_t4) {
                let v4_a = X[x_vec_off4 + cur_k_vec + 0u];
                let v4_b = X[x_vec_off4 + cur_k_vec + 1u];
                let sum_gn4 = gn0 * v4_a.x + gn1 * v4_a.y + gn2 * v4_a.z + gn3 * v4_a.w +
                              gn4 * v4_b.x + gn5 * v4_b.y + gn6 * v4_b.z + gn7 * v4_b.w;
                let sum_un4 = un0 * v4_a.x + un1 * v4_a.y + un2 * v4_a.z + un3 * v4_a.w +
                              un4 * v4_b.x + un5 * v4_b.y + un6 * v4_b.z + un7 * v4_b.w;
                g4 += (g_s * sum_gn4);
                u4 += (u_s * sum_un4);
            }
            if (has_t5) {
                let v5_a = X[x_vec_off5 + cur_k_vec + 0u];
                let v5_b = X[x_vec_off5 + cur_k_vec + 1u];
                let sum_gn5 = gn0 * v5_a.x + gn1 * v5_a.y + gn2 * v5_a.z + gn3 * v5_a.w +
                              gn4 * v5_b.x + gn5 * v5_b.y + gn6 * v5_b.z + gn7 * v5_b.w;
                let sum_un5 = un0 * v5_a.x + un1 * v5_a.y + un2 * v5_a.z + un3 * v5_a.w +
                              un4 * v5_b.x + un5 * v5_b.y + un6 * v5_b.z + un7 * v5_b.w;
                g5 += (g_s * sum_gn5);
                u5 += (u_s * sum_un5);
            }
            if (has_t6) {
                let v6_a = X[x_vec_off6 + cur_k_vec + 0u];
                let v6_b = X[x_vec_off6 + cur_k_vec + 1u];
                let sum_gn6 = gn0 * v6_a.x + gn1 * v6_a.y + gn2 * v6_a.z + gn3 * v6_a.w +
                              gn4 * v6_b.x + gn5 * v6_b.y + gn6 * v6_b.z + gn7 * v6_b.w;
                let sum_un6 = un0 * v6_a.x + un1 * v6_a.y + un2 * v6_a.z + un3 * v6_a.w +
                              un4 * v6_b.x + un5 * v6_b.y + un6 * v6_b.z + un7 * v6_b.w;
                g6 += (g_s * sum_gn6);
                u6 += (u_s * sum_un6);
            }
            if (has_t7) {
                let v7_a = X[x_vec_off7 + cur_k_vec + 0u];
                let v7_b = X[x_vec_off7 + cur_k_vec + 1u];
                let sum_gn7 = gn0 * v7_a.x + gn1 * v7_a.y + gn2 * v7_a.z + gn3 * v7_a.w +
                              gn4 * v7_b.x + gn5 * v7_b.y + gn6 * v7_b.z + gn7 * v7_b.w;
                let sum_un7 = un0 * v7_a.x + un1 * v7_a.y + un2 * v7_a.z + un3 * v7_a.w +
                              un4 * v7_b.x + un5 * v7_b.y + un6 * v7_b.z + un7 * v7_b.w;
                g7 += (g_s * sum_gn7);
                u7 += (u_s * sum_un7);
            }

            base_blk += 8u;
        }
    }

    s_gate0[lid.x] = g0; s_up0[lid.x] = u0;
    if (has_t1) { s_gate1[lid.x] = g1; s_up1[lid.x] = u1; }
    if (has_t2) { s_gate2[lid.x] = g2; s_up2[lid.x] = u2; }
    if (has_t3) { s_gate3[lid.x] = g3; s_up3[lid.x] = u3; }
    if (has_t4) { s_gate4[lid.x] = g4; s_up4[lid.x] = u4; }
    if (has_t5) { s_gate5[lid.x] = g5; s_up5[lid.x] = u5; }
    if (has_t6) { s_gate6[lid.x] = g6; s_up6[lid.x] = u6; }
    if (has_t7) { s_gate7[lid.x] = g7; s_up7[lid.x] = u7; }
    workgroupBarrier();

    let base = local_row * 32u;
    if (lane < 16u) {
        s_gate0[base + lane] += s_gate0[base + lane + 16u]; s_up0[base + lane] += s_up0[base + lane + 16u];
        if (has_t1) { s_gate1[base + lane] += s_gate1[base + lane + 16u]; s_up1[base + lane] += s_up1[base + lane + 16u]; }
        if (has_t2) { s_gate2[base + lane] += s_gate2[base + lane + 16u]; s_up2[base + lane] += s_up2[base + lane + 16u]; }
        if (has_t3) { s_gate3[base + lane] += s_gate3[base + lane + 16u]; s_up3[base + lane] += s_up3[base + lane + 16u]; }
        if (has_t4) { s_gate4[base + lane] += s_gate4[base + lane + 16u]; s_up4[base + lane] += s_up4[base + lane + 16u]; }
        if (has_t5) { s_gate5[base + lane] += s_gate5[base + lane + 16u]; s_up5[base + lane] += s_up5[base + lane + 16u]; }
        if (has_t6) { s_gate6[base + lane] += s_gate6[base + lane + 16u]; s_up6[base + lane] += s_up6[base + lane + 16u]; }
        if (has_t7) { s_gate7[base + lane] += s_gate7[base + lane + 16u]; s_up7[base + lane] += s_up7[base + lane + 16u]; }
    }
    workgroupBarrier();
    if (lane < 8u) {
        s_gate0[base + lane] += s_gate0[base + lane + 8u]; s_up0[base + lane] += s_up0[base + lane + 8u];
        if (has_t1) { s_gate1[base + lane] += s_gate1[base + lane + 8u]; s_up1[base + lane] += s_up1[base + lane + 8u]; }
        if (has_t2) { s_gate2[base + lane] += s_gate2[base + lane + 8u]; s_up2[base + lane] += s_up2[base + lane + 8u]; }
        if (has_t3) { s_gate3[base + lane] += s_gate3[base + lane + 8u]; s_up3[base + lane] += s_up3[base + lane + 8u]; }
        if (has_t4) { s_gate4[base + lane] += s_gate4[base + lane + 8u]; s_up4[base + lane] += s_up4[base + lane + 8u]; }
        if (has_t5) { s_gate5[base + lane] += s_gate5[base + lane + 8u]; s_up5[base + lane] += s_up5[base + lane + 8u]; }
        if (has_t6) { s_gate6[base + lane] += s_gate6[base + lane + 8u]; s_up6[base + lane] += s_up6[base + lane + 8u]; }
        if (has_t7) { s_gate7[base + lane] += s_gate7[base + lane + 8u]; s_up7[base + lane] += s_up7[base + lane + 8u]; }
    }
    workgroupBarrier();
    if (lane < 4u) {
        s_gate0[base + lane] += s_gate0[base + lane + 4u]; s_up0[base + lane] += s_up0[base + lane + 4u];
        if (has_t1) { s_gate1[base + lane] += s_gate1[base + lane + 4u]; s_up1[base + lane] += s_up1[base + lane + 4u]; }
        if (has_t2) { s_gate2[base + lane] += s_gate2[base + lane + 4u]; s_up2[base + lane] += s_up2[base + lane + 4u]; }
        if (has_t3) { s_gate3[base + lane] += s_gate3[base + lane + 4u]; s_up3[base + lane] += s_up3[base + lane + 4u]; }
        if (has_t4) { s_gate4[base + lane] += s_gate4[base + lane + 4u]; s_up4[base + lane] += s_up4[base + lane + 4u]; }
        if (has_t5) { s_gate5[base + lane] += s_gate5[base + lane + 4u]; s_up5[base + lane] += s_up5[base + lane + 4u]; }
        if (has_t6) { s_gate6[base + lane] += s_gate6[base + lane + 4u]; s_up6[base + lane] += s_up6[base + lane + 4u]; }
        if (has_t7) { s_gate7[base + lane] += s_gate7[base + lane + 4u]; s_up7[base + lane] += s_up7[base + lane + 4u]; }
    }
    workgroupBarrier();
    if (lane < 2u) {
        s_gate0[base + lane] += s_gate0[base + lane + 2u]; s_up0[base + lane] += s_up0[base + lane + 2u];
        if (has_t1) { s_gate1[base + lane] += s_gate1[base + lane + 2u]; s_up1[base + lane] += s_up1[base + lane + 2u]; }
        if (has_t2) { s_gate2[base + lane] += s_gate2[base + lane + 2u]; s_up2[base + lane] += s_up2[base + lane + 2u]; }
        if (has_t3) { s_gate3[base + lane] += s_gate3[base + lane + 2u]; s_up3[base + lane] += s_up3[base + lane + 2u]; }
        if (has_t4) { s_gate4[base + lane] += s_gate4[base + lane + 2u]; s_up4[base + lane] += s_up4[base + lane + 2u]; }
        if (has_t5) { s_gate5[base + lane] += s_gate5[base + lane + 2u]; s_up5[base + lane] += s_up5[base + lane + 2u]; }
        if (has_t6) { s_gate6[base + lane] += s_gate6[base + lane + 2u]; s_up6[base + lane] += s_up6[base + lane + 2u]; }
        if (has_t7) { s_gate7[base + lane] += s_gate7[base + lane + 2u]; s_up7[base + lane] += s_up7[base + lane + 2u]; }
    }
    workgroupBarrier();
    if (lane < 1u) {
        s_gate0[base + lane] += s_gate0[base + lane + 1u]; s_up0[base + lane] += s_up0[base + lane + 1u];
        if (has_t1) { s_gate1[base + lane] += s_gate1[base + lane + 1u]; s_up1[base + lane] += s_up1[base + lane + 1u]; }
        if (has_t2) { s_gate2[base + lane] += s_gate2[base + lane + 1u]; s_up2[base + lane] += s_up2[base + lane + 1u]; }
        if (has_t3) { s_gate3[base + lane] += s_gate3[base + lane + 1u]; s_up3[base + lane] += s_up3[base + lane + 1u]; }
        if (has_t4) { s_gate4[base + lane] += s_gate4[base + lane + 1u]; s_up4[base + lane] += s_up4[base + lane + 1u]; }
        if (has_t5) { s_gate5[base + lane] += s_gate5[base + lane + 1u]; s_up5[base + lane] += s_up5[base + lane + 1u]; }
        if (has_t6) { s_gate6[base + lane] += s_gate6[base + lane + 1u]; s_up6[base + lane] += s_up6[base + lane + 1u]; }
        if (has_t7) { s_gate7[base + lane] += s_gate7[base + lane + 1u]; s_up7[base + lane] += s_up7[base + lane + 1u]; }
    }
    workgroupBarrier();

    if (lane == 0u && valid_row) {
        Y[t0 * pc.M + row] = gelu_tanh(s_gate0[base]) * s_up0[base];
        if (has_t1) { Y[t1 * pc.M + row] = gelu_tanh(s_gate1[base]) * s_up1[base]; }
        if (has_t2) { Y[t2 * pc.M + row] = gelu_tanh(s_gate2[base]) * s_up2[base]; }
        if (has_t3) { Y[t3 * pc.M + row] = gelu_tanh(s_gate3[base]) * s_up3[base]; }
        if (has_t4) { Y[t4 * pc.M + row] = gelu_tanh(s_gate4[base]) * s_up4[base]; }
        if (has_t5) { Y[t5 * pc.M + row] = gelu_tanh(s_gate5[base]) * s_up5[base]; }
        if (has_t6) { Y[t6 * pc.M + row] = gelu_tanh(s_gate6[base]) * s_up6[base]; }
        if (has_t7) { Y[t7 * pc.M + row] = gelu_tanh(s_gate7[base]) * s_up7[base]; }
    }
}
