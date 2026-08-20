struct PushConstants {
    dim: u32,
};

@group(0) @binding(0) var<storage, read> Gate: array<f32>;
@group(0) @binding(1) var<storage, read> Up: array<f32>;
@group(0) @binding(2) var<storage, read_write> Act: array<f32>;
var<push_constant> pc: PushConstants;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= pc.dim) {
        return;
    }
    let g = Gate[idx];
    let u = Up[idx];
    let swish = g / (1.0 + exp(-g));
    Act[idx] = swish * u;
}
