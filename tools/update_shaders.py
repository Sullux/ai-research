import sys
from gen_spirv import generate_gemv_vec4_spirv

words_vec4 = generate_gemv_vec4_spirv()

with open("src/gpu/shaders.zig", "r") as f:
    orig = f.read()

# Let's format the array
lines = []
for i in range(0, len(words_vec4), 8):
    chunk = words_vec4[i:i+8]
    hex_strs = [f"0x{w:08X}" for w in chunk]
    lines.append("    " + ", ".join(hex_strs) + ",")
array_str = "\n".join(lines)

new_block = f"""/// 128-bit Vectorized BFloat16 Matrix-Vector Product: y = W * x
/// W: M rows x K cols, stored as packed uvec4 (4 uints = 8 bf16s = 128-bit burst load)
/// x: K floats, y: M floats
/// Push constants: {{ uint M, uint K }}
/// Workgroup size: (64, 1, 1)
pub const GEMV_BF16_SPIRV = [_]u32{{
{array_str}
}};"""

# Replace GEMV_BF16_SPIRV block
import re
pattern = r"/// BFloat16 Matrix-Vector Product: y = W \* x[\s\S]*?pub const GEMV_BF16_SPIRV = \[_\]u32\{[\s\S]*?\n\};"
res = re.sub(pattern, new_block, orig)

with open("src/gpu/shaders.zig", "w") as f:
    f.write(res)

print("Updated src/gpu/shaders.zig successfully")
