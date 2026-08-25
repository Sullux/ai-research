struct PushConstants {
    N: u32,
    M: u32,
    K: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W_gate: array<u32>;
@group(0) @binding(1) var<storage, read> W_up: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata_gate0: array<f32, 128>;
var<workgroup> sdata_gate1: array<f32, 128>;
var<workgroup> sdata_gate2: array<f32, 128>;
var<workgroup> sdata_gate3: array<f32, 128>;

var<workgroup> sdata_up0: array<f32, 128>;
var<workgroup> sdata_up1: array<f32, 128>;
var<workgroup> sdata_up2: array<f32, 128>;
var<workgroup> sdata_up3: array<f32, 128>;

fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;
    let lane = lid.x & 31u;
    let row = wgid.x * 4u + local_row;
    let t_base = wgid.y * 4u;
    if (t_base >= pc.N) {
        return;
    }
    let tid = lid.x;
    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 5u;
    let lane_word_idx = 1u + (lane >> 3u);
    let nib_shift = (lane & 7u) * 4u;

    let has_t1 = (t_base + 1u < pc.N);
    let has_t2 = (t_base + 2u < pc.N);
    let has_t3 = (t_base + 3u < pc.N);

    let x_off0 = t_base * pc.K;
    let x_off1 = (t_base + 1u) * pc.K;
    let x_off2 = (t_base + 2u) * pc.K;
    let x_off3 = (t_base + 3u) * pc.K;

    var g_acc0: f32 = 0.0;
    var g_acc1: f32 = 0.0;
    var g_acc2: f32 = 0.0;
    var g_acc3: f32 = 0.0;

    var u_acc0: f32 = 0.0;
    var u_acc1: f32 = 0.0;
    var u_acc2: f32 = 0.0;
    var u_acc3: f32 = 0.0;

    if (row < pc.M) {
        var b = 0u;
        while (b < num_blocks) {
            let blk_off = row_word_offset + b * 5u;
            let gs = bitcast<f32>(W_gate[blk_off]);
            let gp = W_gate[blk_off + lane_word_idx];
            let gn = (gp >> nib_shift) & 0x0Fu;
            let g_val = (f32(gn) - 8.0) * gs;

            let us = bitcast<f32>(W_up[blk_off]);
            let up = W_up[blk_off + lane_word_idx];
            let un = (up >> nib_shift) & 0x0Fu;
            let u_val = (f32(un) - 8.0) * us;

            let k_idx = b * 32u + lane;
            let x0 = X[x_off0 + k_idx];
            g_acc0 = g_acc0 + g_val * x0;
            u_acc0 = u_acc0 + u_val * x0;

            if (has_t1) {
                let x1 = X[x_off1 + k_idx];
                g_acc1 = g_acc1 + g_val * x1;
                u_acc1 = u_acc1 + u_val * x1;
            }
            if (has_t2) {
                let x2 = X[x_off2 + k_idx];
                g_acc2 = g_acc2 + g_val * x2;
                u_acc2 = u_acc2 + u_val * x2;
            }
            if (has_t3) {
                let x3 = X[x_off3 + k_idx];
                g_acc3 = g_acc3 + g_val * x3;
                u_acc3 = u_acc3 + u_val * x3;
            }

            b = b + 1u;
        }
    }

    sdata_gate0[tid] = g_acc0;
    sdata_gate1[tid] = g_acc1;
    sdata_gate2[tid] = g_acc2;
    sdata_gate3[tid] = g_acc3;

    sdata_up0[tid] = u_acc0;
    sdata_up1[tid] = u_acc1;
    sdata_up2[tid] = u_acc2;
    sdata_up3[tid] = u_acc3;
    workgroupBarrier();

    if (lane < 16u) {
        sdata_gate0[tid] = sdata_gate0[tid] + sdata_gate0[tid + 16u];
        sdata_gate1[tid] = sdata_gate1[tid] + sdata_gate1[tid + 16u];
        sdata_gate2[tid] = sdata_gate2[tid] + sdata_gate2[tid + 16u];
        sdata_gate3[tid] = sdata_gate3[tid] + sdata_gate3[tid + 16u];

        sdata_up0[tid] = sdata_up0[tid] + sdata_up0[tid + 16u];
        sdata_up1[tid] = sdata_up1[tid] + sdata_up1[tid + 16u];
        sdata_up2[tid] = sdata_up2[tid] + sdata_up2[tid + 16u];
        sdata_up3[tid] = sdata_up3[tid] + sdata_up3[tid + 16u];
    }
    workgroupBarrier();
    if (lane < 8u) {
        sdata_gate0[tid] = sdata_gate0[tid] + sdata_gate0[tid + 8u];
        sdata_gate1[tid] = sdata_gate1[tid] + sdata_gate1[tid + 8u];
        sdata_gate2[tid] = sdata_gate2[tid] + sdata_gate2[tid + 8u];
        sdata_gate3[tid] = sdata_gate3[tid] + sdata_gate3[tid + 8u];

        sdata_up0[tid] = sdata_up0[tid] + sdata_up0[tid + 8u];
        sdata_up1[tid] = sdata_up1[tid] + sdata_up1[tid + 8u];
        sdata_up2[tid] = sdata_up2[tid] + sdata_up2[tid + 8u];
        sdata_up3[tid] = sdata_up3[tid] + sdata_up3[tid + 8u];
    }
    workgroupBarrier();
    if (lane < 4u) {
        sdata_gate0[tid] = sdata_gate0[tid] + sdata_gate0[tid + 4u];
        sdata_gate1[tid] = sdata_gate1[tid] + sdata_gate1[tid + 4u];
        sdata_gate2[tid] = sdata_gate2[tid] + sdata_gate2[tid + 4u];
        sdata_gate3[tid] = sdata_gate3[tid] + sdata_gate3[tid + 4u];

        sdata_up0[tid] = sdata_up0[tid] + sdata_up0[tid + 4u];
        sdata_up1[tid] = sdata_up1[tid] + sdata_up1[tid + 4u];
        sdata_up2[tid] = sdata_up2[tid] + sdata_up2[tid + 4u];
        sdata_up3[tid] = sdata_up3[tid] + sdata_up3[tid + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        sdata_gate0[tid] = sdata_gate0[tid] + sdata_gate0[tid + 2u];
        sdata_gate1[tid] = sdata_gate1[tid] + sdata_gate1[tid + 2u];
        sdata_gate2[tid] = sdata_gate2[tid] + sdata_gate2[tid + 2u];
        sdata_gate3[tid] = sdata_gate3[tid] + sdata_gate3[tid + 2u];

        sdata_up0[tid] = sdata_up0[tid] + sdata_up0[tid + 2u];
        sdata_up1[tid] = sdata_up1[tid] + sdata_up1[tid + 2u];
        sdata_up2[tid] = sdata_up2[tid] + sdata_up2[tid + 2u];
        sdata_up3[tid] = sdata_up3[tid] + sdata_up3[tid + 2u];
    }
    workgroupBarrier();
    if (lane < 1u) {
        sdata_gate0[tid] = sdata_gate0[tid] + sdata_gate0[tid + 1u];
        sdata_gate1[tid] = sdata_gate1[tid] + sdata_gate1[tid + 1u];
        sdata_gate2[tid] = sdata_gate2[tid] + sdata_gate2[tid + 1u];
        sdata_gate3[tid] = sdata_gate3[tid] + sdata_gate3[tid + 1u];

        sdata_up0[tid] = sdata_up0[tid] + sdata_up0[tid + 1u];
        sdata_up1[tid] = sdata_up1[tid] + sdata_up1[tid + 1u];
        sdata_up2[tid] = sdata_up2[tid] + sdata_up2[tid + 1u];
        sdata_up3[tid] = sdata_up3[tid] + sdata_up3[tid + 1u];
    }
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        Y[t_base * pc.M + row] = gelu_tanh(sdata_gate0[tid]) * sdata_up0[tid];
        if (has_t1) { Y[(t_base + 1u) * pc.M + row] = gelu_tanh(sdata_gate1[tid]) * sdata_up1[tid]; }
        if (has_t2) { Y[(t_base + 2u) * pc.M + row] = gelu_tanh(sdata_gate2[tid]) * sdata_up2[tid]; }
        if (has_t3) { Y[(t_base + 3u) * pc.M + row] = gelu_tanh(sdata_gate3[tid]) * sdata_up3[tid]; }
    }
}
