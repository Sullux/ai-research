enable subgroups;

struct PushConstants {
    M: u32,
    K: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let local_row = lid.x >> 5u;
    let row = wgid.x * 4u + local_row;

    var acc: f32 = 1.0;
    let sum = subgroupAdd(acc);

    if (lane == 0u && row < pc.M) {
        Y[row] = sum;
    }
}
