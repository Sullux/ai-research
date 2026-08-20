import struct

def generate_gemv_vec4_spirv():
    instructions = []
    
    def emit(op, *args):
        words = []
        for a in args:
            if isinstance(a, str):
                b = a.encode('utf-8') + b'\x00'
                while len(b) % 4 != 0:
                    b += b'\x00'
                words.extend(struct.unpack(f'<{len(b)//4}I', b))
            elif isinstance(a, float):
                words.append(struct.unpack('<I', struct.pack('<f', a))[0])
            else:
                words.append(int(a) & 0xFFFFFFFF)
        length = len(words) + 1
        instructions.append((op | (length << 16)))
        instructions.extend(words)

    id_counter = 1
    def get_id():
        nonlocal id_counter
        tid = id_counter
        id_counter += 1
        return tid

    void_t = get_id()
    func_t = get_id()
    float_t = get_id()
    uint_t = get_id()
    bool_t = get_id()
    uvec3_t = get_id()
    uvec4_t = get_id()
    
    ptr_input_uvec3_t = get_id()
    var_gid = get_id()
    
    arr_uvec4_t = get_id()
    struct_w_t = get_id()
    ptr_sb_struct_w_t = get_id()
    var_w = get_id()
    
    arr_float_t = get_id()
    struct_x_t = get_id()
    ptr_sb_struct_x_t = get_id()
    var_x = get_id()
    
    struct_y_t = get_id()
    ptr_sb_struct_y_t = get_id()
    var_y = get_id()
    
    struct_pc_t = get_id()
    ptr_pc_struct_t = get_id()
    var_pc = get_id()
    
    ptr_sb_uvec4_t = get_id()
    ptr_sb_float_t = get_id()
    ptr_pc_uint_t = get_id()
    
    c_uint_0 = get_id()
    c_uint_1 = get_id()
    c_uint_2 = get_id()
    c_uint_3 = get_id()
    c_uint_4 = get_id()
    c_uint_5 = get_id()
    c_uint_6 = get_id()
    c_uint_7 = get_id()
    c_uint_8 = get_id()
    c_uint_16 = get_id()
    c_uint_mask_lo = get_id()
    c_uint_mask_hi = get_id()
    c_float_0 = get_id()
    
    entry_point = get_id()
    lbl_entry = get_id()
    lbl_in_bounds = get_id()
    lbl_out_of_bounds = get_id()
    lbl_loop_header = get_id()
    lbl_loop_body = get_id()
    lbl_loop_continue = get_id()
    lbl_loop_exit = get_id()
    
    emit(17, 1) # OpCapability Shader
    emit(14, 0, 1) # OpMemoryModel Logical GLSL450
    emit(15, 5, entry_point, "main", var_gid) # OpEntryPoint GLCompute
    emit(16, entry_point, 17, 64, 1, 1) # OpExecutionMode LocalSize 64 1 1
    
    # Decorations
    emit(71, var_gid, 11, 28) # BuiltIn GlobalInvocationId
    
    emit(71, var_w, 33, 0) # Binding 0
    emit(71, var_w, 34, 0) # DescriptorSet 0
    emit(71, struct_w_t, 2) # Block
    emit(72, struct_w_t, 0, 35, 0) # Offset 0
    
    emit(71, var_x, 33, 1) # Binding 1
    emit(71, var_x, 34, 0) # DescriptorSet 0
    emit(71, struct_x_t, 2) # Block
    emit(72, struct_x_t, 0, 35, 0) # Offset 0
    
    emit(71, var_y, 33, 2) # Binding 2
    emit(71, var_y, 34, 0) # DescriptorSet 0
    emit(71, struct_y_t, 2) # Block
    emit(72, struct_y_t, 0, 35, 0) # Offset 0
    
    emit(71, struct_pc_t, 2) # Block
    emit(72, struct_pc_t, 0, 35, 0) # M Offset 0
    emit(72, struct_pc_t, 1, 35, 4) # K Offset 4
    
    # Types
    emit(19, void_t)
    emit(33, func_t, void_t)
    emit(22, float_t, 32)
    emit(21, uint_t, 32, 0)
    emit(20, bool_t)
    emit(23, uvec3_t, uint_t, 3)
    emit(23, uvec4_t, uint_t, 4)
    
    emit(32, ptr_input_uvec3_t, 1, uvec3_t)
    emit(59, ptr_input_uvec3_t, var_gid, 1)
    
    emit(29, arr_uvec4_t, uvec4_t)
    emit(30, struct_w_t, arr_uvec4_t)
    emit(32, ptr_sb_struct_w_t, 12, struct_w_t)
    emit(59, ptr_sb_struct_w_t, var_w, 12)
    
    emit(29, arr_float_t, float_t)
    emit(30, struct_x_t, arr_float_t)
    emit(32, ptr_sb_struct_x_t, 12, struct_x_t)
    emit(59, ptr_sb_struct_x_t, var_x, 12)
    
    emit(30, struct_y_t, arr_float_t)
    emit(32, ptr_sb_struct_y_t, 12, struct_y_t)
    emit(59, ptr_sb_struct_y_t, var_y, 12)
    
    emit(30, struct_pc_t, uint_t, uint_t)
    emit(32, ptr_pc_struct_t, 9, struct_pc_t)
    emit(59, ptr_pc_struct_t, var_pc, 9)
    
    emit(32, ptr_sb_uvec4_t, 12, uvec4_t)
    emit(32, ptr_sb_float_t, 12, float_t)
    emit(32, ptr_pc_uint_t, 9, uint_t)
    
    # Constants
    emit(43, uint_t, c_uint_0, 0)
    emit(43, uint_t, c_uint_1, 1)
    emit(43, uint_t, c_uint_2, 2)
    emit(43, uint_t, c_uint_3, 3)
    emit(43, uint_t, c_uint_4, 4)
    emit(43, uint_t, c_uint_5, 5)
    emit(43, uint_t, c_uint_6, 6)
    emit(43, uint_t, c_uint_7, 7)
    emit(43, uint_t, c_uint_8, 8)
    emit(43, uint_t, c_uint_16, 16)
    emit(43, uint_t, c_uint_mask_lo, 0x0000FFFF)
    emit(43, uint_t, c_uint_mask_hi, 0xFFFF0000)
    emit(43, float_t, c_float_0, 0.0)
    
    # Function main
    emit(54, void_t, entry_point, 0, func_t)
    emit(248, lbl_entry)
    
    v_gid = get_id()
    emit(61, uvec3_t, v_gid, var_gid)
    v_row = get_id()
    emit(81, uint_t, v_row, v_gid, 0)
    
    ptr_m = get_id()
    emit(65, ptr_pc_uint_t, ptr_m, var_pc, c_uint_0)
    v_m = get_id()
    emit(61, uint_t, v_m, ptr_m)
    
    v_cond_m = get_id()
    emit(170, bool_t, v_cond_m, v_row, v_m)
    emit(247, lbl_out_of_bounds, 0)
    emit(250, v_cond_m, lbl_in_bounds, lbl_out_of_bounds)
    
    # In-bounds
    emit(248, lbl_in_bounds)
    ptr_k = get_id()
    emit(65, ptr_pc_uint_t, ptr_k, var_pc, c_uint_1)
    v_k = get_id()
    emit(61, uint_t, v_k, ptr_k)
    
    v_octs = get_id()
    emit(195, uint_t, v_octs, v_k, c_uint_3) # K >> 3 (K / 8)
    
    v_row_offset = get_id()
    emit(132, uint_t, v_row_offset, v_row, v_octs) # row * octs
    emit(249, lbl_loop_header)
    
    # Loop header
    emit(248, lbl_loop_header)
    var_i = get_id()
    emit(245, uint_t, var_i, c_uint_0, lbl_in_bounds)
    var_acc = get_id()
    emit(245, float_t, var_acc, c_float_0, lbl_in_bounds)
    
    v_cond_loop = get_id()
    emit(170, bool_t, v_cond_loop, var_i, v_octs) # i < octs
    emit(246, lbl_loop_exit, lbl_loop_continue, 0) # OpLoopMerge
    emit(250, v_cond_loop, lbl_loop_body, lbl_loop_exit)
    
    # Loop body
    emit(248, lbl_loop_body)
    
    # Load uvec4 W[row_offset + i]
    v_w_idx = get_id()
    emit(128, uint_t, v_w_idx, v_row_offset, var_i)
    ptr_w_elem = get_id()
    emit(65, ptr_sb_uvec4_t, ptr_w_elem, var_w, c_uint_0, v_w_idx)
    v_w4 = get_id()
    emit(61, uvec4_t, v_w4, ptr_w_elem)
    
    # Extract 4 uint pairs
    v_p0 = get_id()
    emit(81, uint_t, v_p0, v_w4, 0)
    v_p1 = get_id()
    emit(81, uint_t, v_p1, v_w4, 1)
    v_p2 = get_id()
    emit(81, uint_t, v_p2, v_w4, 2)
    v_p3 = get_id()
    emit(81, uint_t, v_p3, v_w4, 3)
    
    # Base index for X: i * 8 (i << 3)
    v_x_base = get_id()
    emit(196, uint_t, v_x_base, var_i, c_uint_3)
    
    curr_acc = var_acc
    pairs = [v_p0, v_p1, v_p2, v_p3]
    for p_idx, pair in enumerate(pairs):
        # v0 = uintBitsToFloat((pair & 0xFFFF) << 16)
        and0 = get_id()
        emit(197, uint_t, and0, pair, c_uint_mask_lo)
        shl0 = get_id()
        emit(196, uint_t, shl0, and0, c_uint_16)
        bf0 = get_id()
        emit(124, float_t, bf0, shl0)
        
        # v1 = uintBitsToFloat(pair & 0xFFFF0000)
        and1 = get_id()
        emit(197, uint_t, and1, pair, c_uint_mask_hi)
        bf1 = get_id()
        emit(124, float_t, bf1, and1)
        
        # Load X[x_base + 2*p_idx]
        c_off0 = [c_uint_0, c_uint_2, c_uint_4, c_uint_6][p_idx]
        idx0 = get_id()
        emit(128, uint_t, idx0, v_x_base, c_off0)
        ptr_x0 = get_id()
        emit(65, ptr_sb_float_t, ptr_x0, var_x, c_uint_0, idx0)
        val_x0 = get_id()
        emit(61, float_t, val_x0, ptr_x0)
        
        # Load X[x_base + 2*p_idx + 1]
        c_off1 = [c_uint_1, c_uint_3, c_uint_5, c_uint_7][p_idx]
        idx1 = get_id()
        emit(128, uint_t, idx1, v_x_base, c_off1)
        ptr_x1 = get_id()
        emit(65, ptr_sb_float_t, ptr_x1, var_x, c_uint_0, idx1)
        val_x1 = get_id()
        emit(61, float_t, val_x1, ptr_x1)
        
        # curr_acc += bf0 * val_x0 + bf1 * val_x1
        mul0 = get_id()
        emit(133, float_t, mul0, bf0, val_x0)
        add0 = get_id()
        emit(129, float_t, add0, curr_acc, mul0)
        
        mul1 = get_id()
        emit(133, float_t, mul1, bf1, val_x1)
        add1 = get_id()
        emit(129, float_t, add1, add0, mul1)
        curr_acc = add1

    emit(249, lbl_loop_continue)
    
    # Loop continue
    emit(248, lbl_loop_continue)
    next_i = get_id()
    emit(128, uint_t, next_i, var_i, c_uint_1)
    emit(249, lbl_loop_header)
    
    # Loop exit
    emit(248, lbl_loop_exit)
    ptr_y = get_id()
    emit(65, ptr_sb_float_t, ptr_y, var_y, c_uint_0, v_row)
    emit(62, ptr_y, curr_acc)
    emit(249, lbl_out_of_bounds)
    
    # Out of bounds
    emit(248, lbl_out_of_bounds)
    emit(253) # OpReturn
    emit(56) # OpFunctionEnd
    
    # Update OpPhi instructions in loop header with backedge values
    # var_i (offset of var_i instruction):
    # OpPhi takes (type, result_id, val1, parent1, val2, parent2) -> len=6
    # Let's patch OpPhi for var_i and var_acc:
    for idx in range(len(instructions)):
        w = instructions[idx]
        op = w & 0xFFFF
        if op == 245: # OpPhi
            res_id = instructions[idx+2]
            if res_id == var_i:
                # Replace with 6-word OpPhi
                instructions[idx] = 245 | (6 << 16)
                instructions[idx+1:idx+5] = [uint_t, var_i, c_uint_0, lbl_in_bounds, next_i, lbl_loop_continue]
            elif res_id == var_acc:
                instructions[idx] = 245 | (6 << 16)
                instructions[idx+1:idx+5] = [float_t, var_acc, c_float_0, lbl_in_bounds, curr_acc, lbl_loop_continue]

    header = [0x07230203, 0x00010300, 0x00000000, id_counter, 0x00000000]
    return header + instructions

words = generate_gemv_vec4_spirv()
print(f"Generated {len(words)} SPIR-V words, max id = {words[3]}")
with open("gemv_vec4.spirv", "wb") as f:
    f.write(struct.pack(f"<{len(words)}I", *words))

lines = []
for i in range(0, len(words), 8):
    chunk = words[i:i+8]
    hex_strs = [f"0x{w:08X}" for w in chunk]
    lines.append("    " + ", ".join(hex_strs) + ",")
print("\n".join(lines))
