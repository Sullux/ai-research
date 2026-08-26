struct PushConstants {
    head_dim: u32,
    kv_dim: u32,
    gqa_ratio: u32,
    inv_sqrt_dim: f32,
};

@group(0) @binding(0) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> K_cache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> V_cache: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> Active_slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> Attn_out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> Step_params: array<u32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata_dot: array<f32, 32>;
var<workgroup> s_scores: array<f32, 4096>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let q_head = wgid.x;
    let lane = lid.x;
    let D_vec4 = pc.head_dim >> 2u;
    let kv_vec4 = pc.kv_dim >> 2u;
    let kv_h = q_head / pc.gqa_ratio;
    let q_offset = q_head * D_vec4;

    let S = Step_params[2];

    // Pre-load Q into registers once for the entire slot sequence
    let q0 = Q[q_offset + lane];
    let q1 = Q[q_offset + lane + 32u];
    var q2 = vec4<f32>(0.0);
    var q3 = vec4<f32>(0.0);
    if (D_vec4 == 128u) {
        q2 = Q[q_offset + lane + 64u];
        q3 = Q[q_offset + lane + 96u];
    }

    // 1. Compute dot product scores for all active slots
    for (var slot_i = 0u; slot_i < S; slot_i = slot_i + 1u) {
        let physical_slot = Active_slots[slot_i];
        let kv_offset = physical_slot * kv_vec4 + kv_h * D_vec4;

        let k0 = K_cache[kv_offset + lane];
        let k1 = K_cache[kv_offset + lane + 32u];
        var dot_val = dot(q0, k0) + dot(q1, k1);
        if (D_vec4 == 128u) {
            let k2 = K_cache[kv_offset + lane + 64u];
            let k3 = K_cache[kv_offset + lane + 96u];
            dot_val = dot_val + dot(q2, k2) + dot(q3, k3);
        }

        sdata_dot[lane] = dot_val;
        if (lane < 16u) { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 16u]; }
        if (lane < 8u)  { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 8u]; }
        if (lane < 4u)  { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 4u]; }
        if (lane < 2u)  { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 2u]; }
        if (lane < 1u)  { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 1u]; }

        if (lane == 0u) {
            s_scores[slot_i] = sdata_dot[0] * pc.inv_sqrt_dim;
        }
    }
    workgroupBarrier();

    // 2. Softmax over s_scores[0..S-1]
    if (lane == 0u) {
        var max_val: f32 = -1e9;
        for (var i = 0u; i < S; i = i + 1u) {
            if (s_scores[i] > max_val) {
                max_val = s_scores[i];
            }
        }
        var sum_exp: f32 = 0.0;
        for (var i = 0u; i < S; i = i + 1u) {
            let exp_v = exp(s_scores[i] - max_val);
            s_scores[i] = exp_v;
            sum_exp = sum_exp + exp_v;
        }
        let inv_sum = 1.0 / (sum_exp + 1e-9);
        for (var i = 0u; i < S; i = i + 1u) {
            s_scores[i] = s_scores[i] * inv_sum;
        }
    }
    workgroupBarrier();

    // 3. Weighted sum of V_cache vectors
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    for (var slot_i = 0u; slot_i < S; slot_i = slot_i + 1u) {
        let physical_slot = Active_slots[slot_i];
        let kv_offset = physical_slot * kv_vec4 + kv_h * D_vec4;
        let weight = vec4<f32>(s_scores[slot_i]);

        let v0 = V_cache[kv_offset + lane];
        let v1 = V_cache[kv_offset + lane + 32u];
        acc0 = fma(weight, v0, acc0);
        acc1 = fma(weight, v1, acc1);

        if (D_vec4 == 128u) {
            let v2 = V_cache[kv_offset + lane + 64u];
            let v3 = V_cache[kv_offset + lane + 96u];
            acc2 = fma(weight, v2, acc2);
            acc3 = fma(weight, v3, acc3);
        }
    }

    Attn_out[q_offset + lane] = acc0;
    Attn_out[q_offset + lane + 32u] = acc1;
    if (D_vec4 == 128u) {
        Attn_out[q_offset + lane + 64u] = acc2;
        Attn_out[q_offset + lane + 96u] = acc3;
    }
}
