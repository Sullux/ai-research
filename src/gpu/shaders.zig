const std = @import("std");

pub const bf16 = @import("shaders_bf16.zig");
pub const q8 = @import("shaders_q8.zig");
pub const q4 = @import("shaders_q4.zig");
pub const fused = @import("shaders_fused.zig");
pub const fused_q4 = @import("shaders_fused_q4.zig");
pub const add_norm = @import("shaders_add_rmsnorm.zig");
pub const attn = @import("shaders_attn.zig");

pub const VEC_ADD_SPIRV = fused.VEC_ADD_SPIRV;
pub const FUSED_SWIGLU_SPIRV = fused.FUSED_SWIGLU_SPIRV;
pub const FUSED_GATE_UP_SWIGLU_Q4_SPIRV = fused_q4.FUSED_GATE_UP_SWIGLU_Q4_SPIRV;
pub const FUSED_ADD_RMSNORM_SPIRV = add_norm.FUSED_ADD_RMSNORM_SPIRV;
pub const DECODE_ATTENTION_SPIRV = attn.DECODE_ATTENTION_SPIRV;
pub const GEMV_BF16_SPIRV = bf16.GEMV_BF16_SPIRV;
pub const GEMV_Q8_SPIRV = q8.GEMV_Q8_SPIRV;
pub const GEMV_Q4_SPIRV = q4.GEMV_Q4_SPIRV;
