from gen_spirv import generate_gemv_vec4_spirv, generate_gemv_q8_spirv
from gen_spirv_q4 import generate_gemv_q4_spirv

def format_zig_file(words, name, doc):
    lines = [
        "const std = @import(\"std\");",
        "",
        f"/// {doc}",
        f"pub const {name} = [_]u32{{",
    ]
    for i in range(0, len(words), 16):
        chunk = words[i:i+16]
        hex_strs = [f"0x{w:08X}" for w in chunk]
        lines.append("    " + ", ".join(hex_strs) + ",")
    lines.append("};")
    lines.append("")
    return "\n".join(lines)

# BF16
words_bf16 = generate_gemv_vec4_spirv()
with open("src/gpu/shaders_bf16.zig", "w") as f:
    f.write(format_zig_file(words_bf16, "GEMV_BF16_SPIRV", "128-bit Vectorized BFloat16 GEMV"))

# Q8
words_q8 = generate_gemv_q8_spirv()
with open("src/gpu/shaders_q8.zig", "w") as f:
    f.write(format_zig_file(words_q8, "GEMV_Q8_SPIRV", "Q8_0 Block Quantized GEMV"))

# Q4
words_q4 = generate_gemv_q4_spirv()
with open("src/gpu/shaders_q4.zig", "w") as f:
    f.write(format_zig_file(words_q4, "GEMV_Q4_SPIRV", "Q4_0 Block Quantized GEMV"))
