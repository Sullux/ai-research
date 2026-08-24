struct PushConstants {
    N: u32,
    M: u32,
    K: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata0: array<f32, 32>;
var<workgroup> sdata1: array<f32, 32>;
var<workgroup> sdata2: array<f32, 32>;
var<workgroup> sdata3: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wgid.x;
    let t_base = wgid.y * 4u;
    if (row >= pc.M || t_base >= pc.N) {
        return;
    }
    let lane = lid.x;
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

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    var b = 0u;
    while (b < num_blocks) {
        let blk_off = row_word_offset + b * 5u;
        let s = bitcast<f32>(W[blk_off]);
        let packed_word = W[blk_off + lane_word_idx];
        let nibble = (packed_word >> nib_shift) & 0x0Fu;
        let w_val = (f32(nibble) - 8.0) * s;
        let k_idx = b * 32u + lane;

        acc0 = acc0 + w_val * X[x_off0 + k_idx];
        if (has_t1) { acc1 = acc1 + w_val * X[x_off1 + k_idx]; }
        if (has_t2) { acc2 = acc2 + w_val * X[x_off2 + k_idx]; }
        if (has_t3) { acc3 = acc3 + w_val * X[x_off3 + k_idx]; }

        b = b + 1u;
    }

    sdata0[lane] = acc0;
    sdata1[lane] = acc1;
    sdata2[lane] = acc2;
    sdata3[lane] = acc3;
    workgroupBarrier();

    if (lane < 16u) {
        sdata0[lane] = sdata0[lane] + sdata0[lane + 16u];
        sdata1[lane] = sdata1[lane] + sdata1[lane + 16u];
        sdata2[lane] = sdata2[lane] + sdata2[lane + 16u];
        sdata3[lane] = sdata3[lane] + sdata3[lane + 16u];
    }
    workgroupBarrier();
    if (lane < 8u) {
        sdata0[lane] = sdata0[lane] + sdata0[lane + 8u];
        sdata1[lane] = sdata1[lane] + sdata1[lane + 8u];
        sdata2[lane] = sdata2[lane] + sdata2[lane + 8u];
        sdata3[lane] = sdata3[lane] + sdata3[lane + 8u];
    }
    workgroupBarrier();
    if (lane < 4u) {
        sdata0[lane] = sdata0[lane] + sdata0[lane + 4u];
        sdata1[lane] = sdata1[lane] + sdata1[lane + 4u];
        sdata2[lane] = sdata2[lane] + sdata2[lane + 4u];
        sdata3[lane] = sdata3[lane] + sdata3[lane + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        sdata0[lane] = sdata0[lane] + sdata0[lane + 2u];
        sdata1[lane] = sdata1[lane] + sdata1[lane + 2u];
        sdata2[lane] = sdata2[lane] + sdata2[lane + 2u];
        sdata3[lane] = sdata3[lane] + sdata3[lane + 2u];
    }
    workgroupBarrier();
    if (lane < 1u) {
        sdata0[lane] = sdata0[lane] + sdata0[lane + 1u];
        sdata1[lane] = sdata1[lane] + sdata1[lane + 1u];
        sdata2[lane] = sdata2[lane] + sdata2[lane + 1u];
        sdata3[lane] = sdata3[lane] + sdata3[lane + 1u];
    }
    workgroupBarrier();

    if (lane == 0u) {
        Y[t_base * pc.M + row] = sdata0[0];
        if (has_t1) { Y[(t_base + 1u) * pc.M + row] = sdata1[0]; }
        if (has_t2) { Y[(t_base + 2u) * pc.M + row] = sdata2[0]; }
        if (has_t3) { Y[(t_base + 3u) * pc.M + row] = sdata3[0]; }
    }
}
