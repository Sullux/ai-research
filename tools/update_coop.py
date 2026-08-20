#!/usr/bin/env python3
from gen_spirv_coop import generate_coop_q4

def main():
    q4_code = generate_coop_q4()
    with open("src/gpu/shaders_q4.zig", "w") as f:
        f.write('pub const GEMV_Q4_SPIRV = [_]u32{\n')
        for i, word in enumerate(q4_code):
            f.write(f' 0x{word:08x},')
            if (i + 1) % 8 == 0:
                f.write('\n')
        if len(q4_code) % 8 != 0:
            f.write('\n')
        f.write('};\n')
    print("Updated src/gpu/shaders_q4.zig with wave-cooperative GEMV Q4")

if __name__ == '__main__':
    main()
