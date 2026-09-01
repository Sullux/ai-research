struct PushConstants {
    head_dim: u32,
    kv_dim: u32,
    gqa_ratio: u32,
    inv_sqrt_dim: f32,
    num_q_heads: u32,
    N: u32,
    num_prev_slots: u32,
    is_sliding: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> K_cache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> V_cache: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> Slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> Attn_out: array<vec4<f32>>;
var<push_constant> pc: PushConstants;

var<workgroup> s_Q: array<vec4<f32>, 128>;
var<workgroup> s_scores: array<f32, 4096>;
var<workgroup> s_max_val: f32;
var<workgroup> s_inv_sum: f32;
var<workgroup> s_reduce: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let q_head = wgid.x;
    let t = wgid.y;
    if (q_head >= pc.num_q_heads || t >= pc.N) {
        return;
    }
    let lane = lid.x;
    let kv_h = q_head / pc.gqa_ratio;

    let D_vec4 = pc.head_dim >> 2u;
    let q_dim_vec4 = (pc.num_q_heads * pc.head_dim) >> 2u;
    let kv_dim_vec4 = pc.kv_dim >> 2u;
    let q_offset = t * q_dim_vec4 + q_head * D_vec4;

    // Load Q into shared memory
    s_Q[lane] = select(vec4<f32>(0.0), Q[q_offset + lane], lane < D_vec4);
    s_Q[lane + 32u] = select(vec4<f32>(0.0), Q[q_offset + lane + 32u], (lane + 32u) < D_vec4);
    if (D_vec4 > 64u) {
        s_Q[lane + 64u] = select(vec4<f32>(0.0), Q[q_offset + lane + 64u], (lane + 64u) < D_vec4);
        s_Q[lane + 96u] = select(vec4<f32>(0.0), Q[q_offset + lane + 96u], (lane + 96u) < D_vec4);
    }
    workgroupBarrier();

    let S = pc.num_prev_slots + t + 1u;
    let start_slot = select(0u, select(0u, S - 1024u, S > 1024u), pc.is_sliding != 0u);

    // 1. Compute dot product scores for all slots in window
    for (var slot_i = start_slot + lane; slot_i < S; slot_i = slot_i + 32u) {
        let physical_slot = Slots[slot_i];
        let kv_offset = physical_slot * kv_dim_vec4 + kv_h * D_vec4;

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

    // 2. Softmax over scores
    var local_max: f32 = -1e9;
    var i = start_slot + lane;
    while (i < S) {
        if (s_scores[i] > local_max) { local_max = s_scores[i]; }
        i += 32u;
    }
    s_reduce[lane] = local_max;
    workgroupBarrier();

    if (lane < 16u) { if (s_reduce[lane + 16u] > s_reduce[lane]) { s_reduce[lane] = s_reduce[lane + 16u]; } }
    if (lane < 8u)  { if (s_reduce[lane + 8u] > s_reduce[lane])  { s_reduce[lane] = s_reduce[lane + 8u]; } }
    if (lane < 4u)  { if (s_reduce[lane + 4u] > s_reduce[lane])  { s_reduce[lane] = s_reduce[lane + 4u]; } }
    if (lane < 2u)  { if (s_reduce[lane + 2u] > s_reduce[lane])  { s_reduce[lane] = s_reduce[lane + 2u]; } }
    if (lane < 1u)  { if (s_reduce[lane + 1u] > s_reduce[lane])  { s_reduce[lane] = s_reduce[lane + 1u]; } }

    if (lane == 0u) { s_max_val = s_reduce[0]; }
    workgroupBarrier();

    let max_val = s_max_val;
    var local_sum: f32 = 0.0;
    i = start_slot + lane;
    while (i < S) {
        let exp_v = exp(s_scores[i] - max_val);
        s_scores[i] = exp_v;
        local_sum += exp_v;
        i += 32u;
    }
    s_reduce[lane] = local_sum;
    workgroupBarrier();

    if (lane < 16u) { s_reduce[lane] += s_reduce[lane + 16u]; }
    if (lane < 8u)  { s_reduce[lane] += s_reduce[lane + 8u]; }
    if (lane < 4u)  { s_reduce[lane] += s_reduce[lane + 4u]; }
    if (lane < 2u)  { s_reduce[lane] += s_reduce[lane + 2u]; }
    if (lane < 1u)  { s_reduce[lane] += s_reduce[lane + 1u]; }

    if (lane == 0u) { s_inv_sum = 1.0 / (s_reduce[0] + 1e-9); }
    workgroupBarrier();

    let inv_sum = s_inv_sum;
    i = start_slot + lane;
    while (i < S) {
        s_scores[i] *= inv_sum;
        i += 32u;
    }
    workgroupBarrier();

    // 3. Value accumulation
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    for (var s = start_slot; s < S; s = s + 1u) {
        let physical_slot = Slots[s];
        let kv_offset = physical_slot * kv_dim_vec4 + kv_h * D_vec4;
        let weight = s_scores[s];

        acc0 += weight * V_cache[kv_offset + lane];
        acc1 += weight * V_cache[kv_offset + lane + 32u];
        if (D_vec4 > 64u) {
            acc2 += weight * V_cache[kv_offset + lane + 64u];
            acc3 += weight * V_cache[kv_offset + lane + 96u];
        }
    }

    let out_offset = t * q_dim_vec4 + q_head * D_vec4;
    Attn_out[out_offset + lane] = acc0;
    Attn_out[out_offset + lane + 32u] = acc1;
    if (D_vec4 > 64u) {
        Attn_out[out_offset + lane + 64u] = acc2;
        Attn_out[out_offset + lane + 96u] = acc3;
    }
}
