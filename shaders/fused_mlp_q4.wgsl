struct PushConstants {
    M: u32,
    K: u32,
};

@group(0) @binding(0) var<storage, read> W_gate: array<u32>;
@group(0) @binding(1) var<storage, read> W_up: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata_gate: array<f32, 128>;
var<workgroup> sdata_up: array<f32, 128>;

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

    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 5u;
    let lane_word_idx = 1u + (lane >> 3u);
    let nib_shift = (lane & 7u) * 4u;

    var gate_acc: f32 = 0.0;
    var up_acc: f32 = 0.0;

    if (row < pc.M) {
        var b = 0u;
        while (b < num_blocks) {
            let bb0 = row_word_offset + b * 5u;
            let x0 = X[b * 32u + lane];
            let g_sm0 = unpack2x16float(W_gate[bb0]);
            let gp0 = W_gate[bb0 + lane_word_idx];
            let gn0 = (gp0 >> nib_shift) & 0x0Fu;
            gate_acc = gate_acc + (f32(gn0) * g_sm0.x + g_sm0.y) * x0;
            let u_sm0 = unpack2x16float(W_up[bb0]);
            let up0 = W_up[bb0 + lane_word_idx];
            let un0 = (up0 >> nib_shift) & 0x0Fu;
            up_acc = up_acc + (f32(un0) * u_sm0.x + u_sm0.y) * x0;

            let bb1 = bb0 + 5u;
            let x1 = X[(b + 1u) * 32u + lane];
            let g_sm1 = unpack2x16float(W_gate[bb1]);
            let gp1 = W_gate[bb1 + lane_word_idx];
            let gn1 = (gp1 >> nib_shift) & 0x0Fu;
            gate_acc = gate_acc + (f32(gn1) * g_sm1.x + g_sm1.y) * x1;
            let u_sm1 = unpack2x16float(W_up[bb1]);
            let up1 = W_up[bb1 + lane_word_idx];
            let un1 = (up1 >> nib_shift) & 0x0Fu;
            up_acc = up_acc + (f32(un1) * u_sm1.x + u_sm1.y) * x1;

            let bb2 = bb0 + 10u;
            let x2 = X[(b + 2u) * 32u + lane];
            let g_sm2 = unpack2x16float(W_gate[bb2]);
            let gp2 = W_gate[bb2 + lane_word_idx];
            let gn2 = (gp2 >> nib_shift) & 0x0Fu;
            gate_acc = gate_acc + (f32(gn2) * g_sm2.x + g_sm2.y) * x2;
            let u_sm2 = unpack2x16float(W_up[bb2]);
            let up2 = W_up[bb2 + lane_word_idx];
            let un2 = (up2 >> nib_shift) & 0x0Fu;
            up_acc = up_acc + (f32(un2) * u_sm2.x + u_sm2.y) * x2;

            let bb3 = bb0 + 15u;
            let x3 = X[(b + 3u) * 32u + lane];
            let g_sm3 = unpack2x16float(W_gate[bb3]);
            let gp3 = W_gate[bb3 + lane_word_idx];
            let gn3 = (gp3 >> nib_shift) & 0x0Fu;
            gate_acc = gate_acc + (f32(gn3) * g_sm3.x + g_sm3.y) * x3;
            let u_sm3 = unpack2x16float(W_up[bb3]);
            let up3 = W_up[bb3 + lane_word_idx];
            let un3 = (up3 >> nib_shift) & 0x0Fu;
            up_acc = up_acc + (f32(un3) * u_sm3.x + u_sm3.y) * x3;

            b = b + 4u;
        }
    }

    sdata_gate[lid.x] = gate_acc;
    sdata_up[lid.x] = up_acc;
    workgroupBarrier();

    if (lane < 16u) {
        sdata_gate[lid.x] = sdata_gate[lid.x] + sdata_gate[lid.x + 16u];
        sdata_up[lid.x] = sdata_up[lid.x] + sdata_up[lid.x + 16u];
    }
    workgroupBarrier();
    if (lane < 8u) {
        sdata_gate[lid.x] = sdata_gate[lid.x] + sdata_gate[lid.x + 8u];
        sdata_up[lid.x] = sdata_up[lid.x] + sdata_up[lid.x + 8u];
    }
    workgroupBarrier();
    if (lane < 4u) {
        sdata_gate[lid.x] = sdata_gate[lid.x] + sdata_gate[lid.x + 4u];
        sdata_up[lid.x] = sdata_up[lid.x] + sdata_up[lid.x + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        sdata_gate[lid.x] = sdata_gate[lid.x] + sdata_gate[lid.x + 2u];
        sdata_up[lid.x] = sdata_up[lid.x] + sdata_up[lid.x + 2u];
    }
    workgroupBarrier();
    if (lane < 1u) {
        sdata_gate[lid.x] = sdata_gate[lid.x] + sdata_gate[lid.x + 1u];
        sdata_up[lid.x] = sdata_up[lid.x] + sdata_up[lid.x + 1u];
    }
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        let g_final = sdata_gate[lid.x];
        let u_final = sdata_up[lid.x];
        let act = gelu_tanh(g_final);
        Y[row] = act * u_final;
    }
}
