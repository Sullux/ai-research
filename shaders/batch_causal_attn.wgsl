struct PushConstants {
    head_dim: u32,
    kv_dim: u32,
    gqa_ratio: u32,
    inv_sqrt_dim: f32,
    num_q_heads: u32,
    N: u32,
    num_prev_slots: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> K_cache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> V_cache: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> Slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> Attn_out: array<vec4<f32>>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata: array<f32, 64>;
var<workgroup> s_scores: array<f32, 4096>;
var<workgroup> s_max_val: f32;
var<workgroup> s_inv_sum: f32;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let q_head = wgid.x;
    let t = wgid.y;
    if (q_head >= pc.num_q_heads || t >= pc.N) {
        return;
    }
    let d = lid.x; // 0..63 (64 x vec4 = 256 dimensions)
    let kv_h = q_head / pc.gqa_ratio;

    // Offsets in units of vec4<f32>
    let d_vec4 = pc.head_dim >> 2u;
    let q_dim_vec4 = (pc.num_q_heads * pc.head_dim) >> 2u;
    let kv_dim_vec4 = pc.kv_dim >> 2u;

    let q_vec_off = t * q_dim_vec4 + q_head * d_vec4 + d;
    let q_vec = select(vec4<f32>(0.0), Q[q_vec_off], d < d_vec4);

    let S = pc.num_prev_slots + t + 1u;

    // 1. Vectorized Q-K Dot Products
    for (var slot_idx = 0u; slot_idx < S; slot_idx = slot_idx + 1u) {
        let physical_slot = Slots[slot_idx];
        let k_vec_off = physical_slot * kv_dim_vec4 + kv_h * d_vec4 + d;
        let k_vec = select(vec4<f32>(0.0), K_cache[k_vec_off], d < d_vec4);

        let dot = q_vec.x * k_vec.x + q_vec.y * k_vec.y + q_vec.z * k_vec.z + q_vec.w * k_vec.w;
        sdata[d] = dot;
        workgroupBarrier();

        if (d < 32u) { sdata[d] += sdata[d + 32u]; }
        if (d < 16u) { sdata[d] += sdata[d + 16u]; }
        if (d < 8u)  { sdata[d] += sdata[d + 8u];  }
        if (d < 4u)  { sdata[d] += sdata[d + 4u];  }
        if (d < 2u)  { sdata[d] += sdata[d + 2u];  }
        if (d < 1u)  { sdata[d] += sdata[d + 1u];  }

        if (d == 0u) {
            s_scores[slot_idx] = sdata[0] * pc.inv_sqrt_dim;
        }
        workgroupBarrier();
    }

    // 2. Softmax over scores
    if (d == 0u) {
        var max_val: f32 = -1e9;
        for (var i = 0u; i < S; i = i + 1u) {
            if (s_scores[i] > max_val) { max_val = s_scores[i]; }
        }
        s_max_val = max_val;
    }
    workgroupBarrier();

    let max_val = s_max_val;
    var local_sum_exp: f32 = 0.0;
    var idx = d;
    while (idx < S) {
        let exp_v = exp(s_scores[idx] - max_val);
        s_scores[idx] = exp_v;
        local_sum_exp += exp_v;
        idx += 64u;
    }
    sdata[d] = local_sum_exp;
    workgroupBarrier();

    if (d < 32u) { sdata[d] += sdata[d + 32u]; }
    if (d < 16u) { sdata[d] += sdata[d + 16u]; }
    if (d < 8u)  { sdata[d] += sdata[d + 8u];  }
    if (d < 4u)  { sdata[d] += sdata[d + 4u];  }
    if (d < 2u)  { sdata[d] += sdata[d + 2u];  }
    if (d < 1u)  { sdata[d] += sdata[d + 1u];  }

    if (d == 0u) {
        s_inv_sum = 1.0 / (sdata[0] + 1e-9);
    }
    workgroupBarrier();

    let inv_sum = s_inv_sum;
    idx = d;
    while (idx < S) {
        s_scores[idx] *= inv_sum;
        idx += 64u;
    }
    workgroupBarrier();

    // 3. Weighted Sum of V cache (64 threads compute 64 x vec4 = 256 dimensions)
    if (d < d_vec4) {
        var acc = vec4<f32>(0.0);
        for (var i = 0u; i < S; i = i + 1u) {
            let physical_slot = Slots[i];
            let v_vec_off = physical_slot * kv_dim_vec4 + kv_h * d_vec4 + d;
            let v_val = V_cache[v_vec_off];
            acc += s_scores[i] * v_val;
        }
        Attn_out[q_vec_off] = acc;
    }
}
