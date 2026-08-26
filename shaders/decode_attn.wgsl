struct PushConstants {
    head_dim: u32,
    kv_dim: u32,
    gqa_ratio: u32,
    inv_sqrt_dim: f32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K_cache: array<f32>;
@group(0) @binding(2) var<storage, read> V_cache: array<f32>;
@group(0) @binding(3) var<storage, read> Active_slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> Attn_out: array<f32>;
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
    let D = pc.head_dim;
    let kv_h = q_head / pc.gqa_ratio;
    let q_offset = q_head * D;

    let S = Step_params[2];

    // 1. Compute dot product scores for all active slots
    for (var slot_i = 0u; slot_i < S; slot_i = slot_i + 1u) {
        let physical_slot = Active_slots[slot_i];
        let kv_offset = physical_slot * pc.kv_dim + kv_h * D;

        var dot: f32 = 0.0;
        var d = lane;
        while (d < D) {
            dot = dot + Q[q_offset + d] * K_cache[kv_offset + d]
                      + Q[q_offset + d + 32u] * K_cache[kv_offset + d + 32u]
                      + Q[q_offset + d + 64u] * K_cache[kv_offset + d + 64u]
                      + Q[q_offset + d + 96u] * K_cache[kv_offset + d + 96u];
            d = d + 128u;
        }

        sdata_dot[lane] = dot;
        workgroupBarrier();

        if (lane < 16u) { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 16u]; }
        workgroupBarrier();
        if (lane < 8u) { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 8u]; }
        workgroupBarrier();
        if (lane < 4u) { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 4u]; }
        workgroupBarrier();
        if (lane < 2u) { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 2u]; }
        workgroupBarrier();
        if (lane < 1u) { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 1u]; }
        workgroupBarrier();

        if (lane == 0u) {
            s_scores[slot_i] = sdata_dot[0] * pc.inv_sqrt_dim;
        }
        workgroupBarrier();
    }

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
    var d = lane;
    while (d < D) {
        var acc0: f32 = 0.0;
        var acc1: f32 = 0.0;
        var acc2: f32 = 0.0;
        var acc3: f32 = 0.0;
        for (var i = 0u; i < S; i = i + 1u) {
            let physical_slot = Active_slots[i];
            let kv_offset = physical_slot * pc.kv_dim + kv_h * D;
            let weight = s_scores[i];
            acc0 = acc0 + weight * V_cache[kv_offset + d];
            acc1 = acc1 + weight * V_cache[kv_offset + d + 32u];
            acc2 = acc2 + weight * V_cache[kv_offset + d + 64u];
            acc3 = acc3 + weight * V_cache[kv_offset + d + 96u];
        }
        Attn_out[q_offset + d] = acc0;
        Attn_out[q_offset + d + 32u] = acc1;
        Attn_out[q_offset + d + 64u] = acc2;
        Attn_out[q_offset + d + 96u] = acc3;
        d = d + 128u;
    }
}
