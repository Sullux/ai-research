struct PushConstants {
    M: u32,
    K: u32,
};

@group(0) @binding(0) var<storage, read> W_gate: array<u32>;
@group(0) @binding(1) var<storage, read> W_up: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_reduce_gate: array<f32, 128>;
var<workgroup> s_reduce_up: array<f32, 128>;

fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;      // 0..3 (4 rows per workgroup)
    let lane = lid.x & 31u;            // 0..31 (lane in wave)
    let row = wgid.x * 4u + local_row;

    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 5u;

    var gate_acc: f32 = 0.0;
    var up_acc: f32 = 0.0;

    if (row < pc.M) {
        var blk = lane;
        while (blk < num_blocks) {
            let blk_word_off = row_word_offset + blk * 5u;

            let g_sm = unpack2x16float(W_gate[blk_word_off]);
            let gw0 = W_gate[blk_word_off + 1u];
            let gw1 = W_gate[blk_word_off + 2u];
            let gw2 = W_gate[blk_word_off + 3u];
            let gw3 = W_gate[blk_word_off + 4u];

            let u_sm = unpack2x16float(W_up[blk_word_off]);
            let uw0 = W_up[blk_word_off + 1u];
            let uw1 = W_up[blk_word_off + 2u];
            let uw2 = W_up[blk_word_off + 3u];
            let uw3 = W_up[blk_word_off + 4u];

            let x_base = blk * 32u;

            var sum_g: f32 = 0.0;
            var sum_u: f32 = 0.0;
            var sum_x: f32 = 0.0;

            // Word 0 (0..7)
            var cur_gw = gw0;
            var cur_uw = uw0;
            for (var j = 0u; j < 8u; j = j + 1u) {
                let gn = f32(cur_gw & 0x0Fu);
                let un = f32(cur_uw & 0x0Fu);
                let xv = X[x_base + j];
                sum_g = sum_g + gn * xv;
                sum_u = sum_u + un * xv;
                sum_x = sum_x + xv;
                cur_gw = cur_gw >> 4u;
                cur_uw = cur_uw >> 4u;
            }

            // Word 1 (8..15)
            cur_gw = gw1;
            cur_uw = uw1;
            for (var j = 0u; j < 8u; j = j + 1u) {
                let gn = f32(cur_gw & 0x0Fu);
                let un = f32(cur_uw & 0x0Fu);
                let xv = X[x_base + 8u + j];
                sum_g = sum_g + gn * xv;
                sum_u = sum_u + un * xv;
                sum_x = sum_x + xv;
                cur_gw = cur_gw >> 4u;
                cur_uw = cur_uw >> 4u;
            }

            // Word 2 (16..23)
            cur_gw = gw2;
            cur_uw = uw2;
            for (var j = 0u; j < 8u; j = j + 1u) {
                let gn = f32(cur_gw & 0x0Fu);
                let un = f32(cur_uw & 0x0Fu);
                let xv = X[x_base + 16u + j];
                sum_g = sum_g + gn * xv;
                sum_u = sum_u + un * xv;
                sum_x = sum_x + xv;
                cur_gw = cur_gw >> 4u;
                cur_uw = cur_uw >> 4u;
            }

            // Word 3 (24..31)
            cur_gw = gw3;
            cur_uw = uw3;
            for (var j = 0u; j < 8u; j = j + 1u) {
                let gn = f32(cur_gw & 0x0Fu);
                let un = f32(cur_uw & 0x0Fu);
                let xv = X[x_base + 24u + j];
                sum_g = sum_g + gn * xv;
                sum_u = sum_u + un * xv;
                sum_x = sum_x + xv;
                cur_gw = cur_gw >> 4u;
                cur_uw = cur_uw >> 4u;
            }

            gate_acc = gate_acc + (g_sm.x * sum_g + g_sm.y * sum_x);
            up_acc = up_acc + (u_sm.x * sum_u + u_sm.y * sum_x);
            blk = blk + 32u;
        }
    }

    s_reduce_gate[lid.x] = gate_acc;
    s_reduce_up[lid.x] = up_acc;
    workgroupBarrier();

    let base = local_row * 32u;
    if (lane < 16u) {
        s_reduce_gate[base + lane] = s_reduce_gate[base + lane] + s_reduce_gate[base + lane + 16u];
        s_reduce_up[base + lane] = s_reduce_up[base + lane] + s_reduce_up[base + lane + 16u];
    }
    workgroupBarrier();
    if (lane < 8u) {
        s_reduce_gate[base + lane] = s_reduce_gate[base + lane] + s_reduce_gate[base + lane + 8u];
        s_reduce_up[base + lane] = s_reduce_up[base + lane] + s_reduce_up[base + lane + 8u];
    }
    workgroupBarrier();
    if (lane < 4u) {
        s_reduce_gate[base + lane] = s_reduce_gate[base + lane] + s_reduce_gate[base + lane + 4u];
        s_reduce_up[base + lane] = s_reduce_up[base + lane] + s_reduce_up[base + lane + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        s_reduce_gate[base + lane] = s_reduce_gate[base + lane] + s_reduce_gate[base + lane + 2u];
        s_reduce_up[base + lane] = s_reduce_up[base + lane] + s_reduce_up[base + lane + 2u];
    }
    workgroupBarrier();
    if (lane < 1u) {
        s_reduce_gate[base + lane] = s_reduce_gate[base + lane] + s_reduce_gate[base + lane + 1u];
        s_reduce_up[base + lane] = s_reduce_up[base + lane] + s_reduce_up[base + lane + 1u];
    }
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        let g_val = s_reduce_gate[base];
        let u_val = s_reduce_up[base];
        Y[row] = gelu_tanh(g_val) * u_val;
    }
}
