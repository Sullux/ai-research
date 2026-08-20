struct PushConstants {
    n_active: u32,
    head_dim: u32,
    gqa_ratio: u32,
    inv_sqrt_dim: f32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K_cache: array<f32>;
@group(0) @binding(2) var<storage, read> V_cache: array<f32>;
@group(0) @binding(3) var<storage, read> Active_slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> Attn_out: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata_dot: array<f32, 32>;
var<workgroup> s_scores: array<f32, 512>;

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

    let S = pc.n_active;

    // 1. Compute dot product scores for all active slots
    for (var slot_i = 0u; slot_i < S; slot_i = slot_i + 1u) {
        let physical_slot = Active_slots[slot_i];
        let kv_offset = physical_slot * D;

        var dot: f32 = 0.0;
        var d = lane;
        while (d < D) {
            dot = dot + Q[q_offset + d] * K_cache[kv_offset + d];
            d = d + 32u;
        }

        sdata_dot[lane] = dot;
        workgroupBarrier();

        if (lane == 0u) {
            var sum: f32 = 0.0;
            for (var i = 0u; i < 32u; i = i + 1u) {
                sum = sum + sdata_dot[i];
            }
            s_scores[slot_i] = sum * pc.inv_sqrt_dim;
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
            let e = exp(s_scores[i] - max_val);
            s_scores[i] = e;
            sum_exp = sum_exp + e;
        }
        let inv_sum = 1.0 / sum_exp;
        for (var i = 0u; i < S; i = i + 1u) {
            s_scores[i] = s_scores[i] * inv_sum;
        }
    }
    workgroupBarrier();

    // 3. Weighted accumulation of V
    var d = lane;
    while (d < D) {
        var out_val: f32 = 0.0;
        for (var slot_i = 0u; slot_i < S; slot_i = slot_i + 1u) {
            let physical_slot = Active_slots[slot_i];
            let kv_offset = physical_slot * D;
            out_val = out_val + s_scores[slot_i] * V_cache[kv_offset + d];
        }
        Attn_out[q_offset + d] = out_val;
        d = d + 32u;
    }
}
