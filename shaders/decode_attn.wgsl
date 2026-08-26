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

var<workgroup> s_Q: array<vec4<f32>, 128>;
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

    // Pre-load Q for this head into shared memory (32 threads load 4 vec4s each)
    s_Q[lane] = Q[q_offset + lane];
    s_Q[lane + 32u] = Q[q_offset + lane + 32u];
    if (D_vec4 == 128u) {
        s_Q[lane + 64u] = Q[q_offset + lane + 64u];
        s_Q[lane + 96u] = Q[q_offset + lane + 96u];
    }
    workgroupBarrier();

    // 1. Compute dot product scores for all active slots (4x unrolled burst reads)
    for (var slot_i = lane; slot_i < S; slot_i = slot_i + 32u) {
        let physical_slot = Active_slots[slot_i];
        let kv_offset = physical_slot * kv_vec4 + kv_h * D_vec4;

        var dot_sum = 0.0;
        var d = 0u;
        while (d + 4u <= D_vec4) {
            let q0 = s_Q[d];
            let q1 = s_Q[d + 1u];
            let q2 = s_Q[d + 2u];
            let q3 = s_Q[d + 3u];

            let k0 = K_cache[kv_offset + d];
            let k1 = K_cache[kv_offset + d + 1u];
            let k2 = K_cache[kv_offset + d + 2u];
            let k3 = K_cache[kv_offset + d + 3u];

            dot_sum = dot_sum + dot(q0, k0) + dot(q1, k1) + dot(q2, k2) + dot(q3, k3);
            d = d + 4u;
        }
        while (d < D_vec4) {
            dot_sum = dot_sum + dot(s_Q[d], K_cache[kv_offset + d]);
            d = d + 1u;
        }
        s_scores[slot_i] = dot_sum * pc.inv_sqrt_dim;
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

    // 3. Weighted sum of V_cache vectors (4x unrolled for pipelined memory loads)
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    var slot_i = 0u;
    while (slot_i + 4u <= S) {
        let s0 = Active_slots[slot_i];
        let s1 = Active_slots[slot_i + 1u];
        let s2 = Active_slots[slot_i + 2u];
        let s3 = Active_slots[slot_i + 3u];

        let off0 = s0 * kv_vec4 + kv_h * D_vec4;
        let off1 = s1 * kv_vec4 + kv_h * D_vec4;
        let off2 = s2 * kv_vec4 + kv_h * D_vec4;
        let off3 = s3 * kv_vec4 + kv_h * D_vec4;

        let w0 = vec4<f32>(s_scores[slot_i]);
        let w1 = vec4<f32>(s_scores[slot_i + 1u]);
        let w2 = vec4<f32>(s_scores[slot_i + 2u]);
        let w3 = vec4<f32>(s_scores[slot_i + 3u]);

        acc0 = fma(w0, V_cache[off0 + lane], acc0);
        acc0 = fma(w1, V_cache[off1 + lane], acc0);
        acc0 = fma(w2, V_cache[off2 + lane], acc0);
        acc0 = fma(w3, V_cache[off3 + lane], acc0);

        acc1 = fma(w0, V_cache[off0 + lane + 32u], acc1);
        acc1 = fma(w1, V_cache[off1 + lane + 32u], acc1);
        acc1 = fma(w2, V_cache[off2 + lane + 32u], acc1);
        acc1 = fma(w3, V_cache[off3 + lane + 32u], acc1);

        if (D_vec4 == 128u) {
            acc2 = fma(w0, V_cache[off0 + lane + 64u], acc2);
            acc2 = fma(w1, V_cache[off1 + lane + 64u], acc2);
            acc2 = fma(w2, V_cache[off2 + lane + 64u], acc2);
            acc2 = fma(w3, V_cache[off3 + lane + 64u], acc2);

            acc3 = fma(w0, V_cache[off0 + lane + 96u], acc3);
            acc3 = fma(w1, V_cache[off1 + lane + 96u], acc3);
            acc3 = fma(w2, V_cache[off2 + lane + 96u], acc3);
            acc3 = fma(w3, V_cache[off3 + lane + 96u], acc3);
        }

        slot_i = slot_i + 4u;
    }

    while (slot_i < S) {
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
        slot_i = slot_i + 1u;
    }

    Attn_out[q_offset + lane] = acc0;
    Attn_out[q_offset + lane + 32u] = acc1;
    if (D_vec4 == 128u) {
        Attn_out[q_offset + lane + 64u] = acc2;
        Attn_out[q_offset + lane + 96u] = acc3;
    }
}
