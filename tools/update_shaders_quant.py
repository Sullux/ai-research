import sys
from gen_spirv import generate_gemv_q8_spirv
from gen_spirv_q4 import generate_gemv_q4_spirv

words_q8 = generate_gemv_q8_spirv()
words_q4 = generate_gemv_q4_spirv()

def format_array(words, name, doc):
    lines = []
    for i in range(0, len(words), 8):
        chunk = words[i:i+8]
        hex_strs = [f"0x{w:08X}" for w in chunk]
        lines.append("    " + ", ".join(hex_strs) + ",")
    array_str = "\n".join(lines)
    return f"""/// {doc}
pub const {name} = [_]u32{{
{array_str}
}};"""

block_q8 = format_array(words_q8, "GEMV_Q8_SPIRV", "Q8_0 Block Quantized GEMV: y = W * x\n/// Block size 32 (9 u32 words per block: 1 f32 scale + 8 u32 of 32 i8 values)\n/// Push constants: { uint M, uint K }\n/// Workgroup size: (64, 1, 1)")
block_q4 = format_array(words_q4, "GEMV_Q4_SPIRV", "Q4_0 Block Quantized GEMV: y = W * x\n/// Block size 32 (5 u32 words per block: 1 f32 scale + 4 u32 of 32 nibbles)\n/// Push constants: { uint M, uint K }\n/// Workgroup size: (64, 1, 1)")

with open("src/gpu/shaders.zig", "r") as f:
    orig = f.read()

# Check if already present or append
if "GEMV_Q8_SPIRV" in orig:
    import re
    orig = re.sub(r"/// Q8_0 Block Quantized GEMV[\s\S]*?pub const GEMV_Q8_SPIRV = \[_\]u32\{[\s\S]*?\n\};", block_q8, orig)
    orig = re.sub(r"/// Q4_0 Block Quantized GEMV[\s\S]*?pub const GEMV_Q4_SPIRV = \[_\]u32\{[\s\S]*?\n\};", block_q4, orig)
    res = orig
else:
    res = orig.strip() + "\n\n" + block_q8 + "\n\n" + block_q4 + "\n"

with open("src/gpu/shaders.zig", "w") as f:
    f.write(res)

print("Updated src/gpu/shaders.zig with Q8 and Q4 shaders successfully")
