import struct

def make_spirv_header(id_bound, instructions):
    header = [0x07230203, 0x00010300, 0x00000000, id_bound, 0x00000000]
    return header + instructions

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
    emit(15, 5, entry_point, "main", var_gid)
    emit(16, entry_point, 17, 64, 1, 1)
    
    emit(71, var_gid, 11, 28)
    emit(71, var_w, 33, 0)
    emit(71, var_w, 34, 0)
    emit(71, struct_w_t, 2)
    emit(72, struct_w_t, 0, 35, 0)
    
    emit(71, var_x, 33, 1)
    emit(71, var_x, 34, 0)
    emit(71, struct_x_t, 2)
    emit(72, struct_x_t, 0, 35, 0)
    
    emit(71, var_y, 33, 2)
    emit(71, var_y, 34, 0)
    emit(71, struct_y_t, 2)
    emit(72, struct_y_t, 0, 35, 0)
    
    emit(71, struct_pc_t, 2)
    emit(72, struct_pc_t, 0, 35, 0)
    emit(72, struct_pc_t, 1, 35, 4)
    
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
    
    emit(248, lbl_in_bounds)
    ptr_k = get_id()
    emit(65, ptr_pc_uint_t, ptr_k, var_pc, c_uint_1)
    v_k = get_id()
    emit(61, uint_t, v_k, ptr_k)
    
    v_octs = get_id()
    emit(195, uint_t, v_octs, v_k, c_uint_3)
    
    v_row_offset = get_id()
    emit(132, uint_t, v_row_offset, v_row, v_octs)
    emit(249, lbl_loop_header)
    
    emit(248, lbl_loop_header)
    var_i = get_id()
    emit(245, uint_t, var_i, c_uint_0, lbl_in_bounds)
    var_acc = get_id()
    emit(245, float_t, var_acc, c_float_0, lbl_in_bounds)
    
    v_cond_loop = get_id()
    emit(170, bool_t, v_cond_loop, var_i, v_octs)
    emit(246, lbl_loop_exit, lbl_loop_continue, 0)
    emit(250, v_cond_loop, lbl_loop_body, lbl_loop_exit)
    
    emit(248, lbl_loop_body)
    
    v_w_idx = get_id()
    emit(128, uint_t, v_w_idx, v_row_offset, var_i)
    ptr_w_elem = get_id()
    emit(65, ptr_sb_uvec4_t, ptr_w_elem, var_w, c_uint_0, v_w_idx)
    v_w4 = get_id()
    emit(61, uvec4_t, v_w4, ptr_w_elem)
    
    v_p0 = get_id()
    emit(81, uint_t, v_p0, v_w4, 0)
    v_p1 = get_id()
    emit(81, uint_t, v_p1, v_w4, 1)
    v_p2 = get_id()
    emit(81, uint_t, v_p2, v_w4, 2)
    v_p3 = get_id()
    emit(81, uint_t, v_p3, v_w4, 3)
    
    v_x_base = get_id()
    emit(196, uint_t, v_x_base, var_i, c_uint_3)
    
    curr_acc = var_acc
    pairs = [v_p0, v_p1, v_p2, v_p3]
    for p_idx, pair in enumerate(pairs):
        and0 = get_id()
        emit(197, uint_t, and0, pair, c_uint_mask_lo)
        shl0 = get_id()
        emit(196, uint_t, shl0, and0, c_uint_16)
        bf0 = get_id()
        emit(124, float_t, bf0, shl0)
        
        and1 = get_id()
        emit(197, uint_t, and1, pair, c_uint_mask_hi)
        bf1 = get_id()
        emit(124, float_t, bf1, and1)
        
        c_off0 = [c_uint_0, c_uint_2, c_uint_4, c_uint_6][p_idx]
        idx0 = get_id()
        emit(128, uint_t, idx0, v_x_base, c_off0)
        ptr_x0 = get_id()
        emit(65, ptr_sb_float_t, ptr_x0, var_x, c_uint_0, idx0)
        val_x0 = get_id()
        emit(61, float_t, val_x0, ptr_x0)
        
        c_off1 = [c_uint_1, c_uint_3, c_uint_5, c_uint_7][p_idx]
        idx1 = get_id()
        emit(128, uint_t, idx1, v_x_base, c_off1)
        ptr_x1 = get_id()
        emit(65, ptr_sb_float_t, ptr_x1, var_x, c_uint_0, idx1)
        val_x1 = get_id()
        emit(61, float_t, val_x1, ptr_x1)
        
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
    
    emit(248, lbl_loop_continue)
    next_i = get_id()
    emit(128, uint_t, next_i, var_i, c_uint_1)
    emit(249, lbl_loop_header)
    
    emit(248, lbl_loop_exit)
    ptr_y = get_id()
    emit(65, ptr_sb_float_t, ptr_y, var_y, c_uint_0, v_row)
    emit(62, ptr_y, curr_acc)
    emit(249, lbl_out_of_bounds)
    
    emit(248, lbl_out_of_bounds)
    emit(253)
    emit(56)
    
    for idx in range(len(instructions)):
        w = instructions[idx]
        op = w & 0xFFFF
        if op == 245:
            res_id = instructions[idx+2]
            if res_id == var_i:
                instructions[idx] = 245 | (6 << 16)
                instructions[idx+1:idx+5] = [uint_t, var_i, c_uint_0, lbl_in_bounds, next_i, lbl_loop_continue]
            elif res_id == var_acc:
                instructions[idx] = 245 | (6 << 16)
                instructions[idx+1:idx+5] = [float_t, var_acc, c_float_0, lbl_in_bounds, curr_acc, lbl_loop_continue]

    return make_spirv_header(id_counter, instructions)

def generate_gemv_q8_spirv():
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
    int_t = get_id()
    bool_t = get_id()
    uvec3_t = get_id()
    
    ptr_input_uvec3_t = get_id()
    var_gid = get_id()
    
    arr_uint_t = get_id()
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
    
    ptr_sb_uint_t = get_id()
    ptr_sb_float_t = get_id()
    ptr_pc_uint_t = get_id()
    
    c_uint_0 = get_id()
    c_uint_1 = get_id()
    c_uint_5 = get_id()
    c_uint_8 = get_id()
    c_uint_9 = get_id()
    c_uint_16 = get_id()
    c_uint_24 = get_id()
    c_float_0 = get_id()
    
    entry_point = get_id()
    lbl_entry = get_id()
    lbl_in_bounds = get_id()
    lbl_out_of_bounds = get_id()
    lbl_b_header = get_id()
    lbl_b_body = get_id()
    lbl_b_continue = get_id()
    lbl_b_exit = get_id()
    
    emit(17, 1)
    emit(14, 0, 1)
    emit(15, 5, entry_point, "main", var_gid)
    emit(16, entry_point, 17, 64, 1, 1)
    
    emit(71, var_gid, 11, 28)
    emit(71, var_w, 33, 0)
    emit(71, var_w, 34, 0)
    emit(71, struct_w_t, 2)
    emit(72, struct_w_t, 0, 35, 0)
    
    emit(71, var_x, 33, 1)
    emit(71, var_x, 34, 0)
    emit(71, struct_x_t, 2)
    emit(72, struct_x_t, 0, 35, 0)
    
    emit(71, var_y, 33, 2)
    emit(71, var_y, 34, 0)
    emit(71, struct_y_t, 2)
    emit(72, struct_y_t, 0, 35, 0)
    
    emit(71, struct_pc_t, 2)
    emit(72, struct_pc_t, 0, 35, 0)
    emit(72, struct_pc_t, 1, 35, 4)
    
    emit(19, void_t)
    emit(33, func_t, void_t)
    emit(22, float_t, 32)
    emit(21, uint_t, 32, 0)
    emit(21, int_t, 32, 1)
    emit(20, bool_t)
    emit(23, uvec3_t, uint_t, 3)
    
    emit(32, ptr_input_uvec3_t, 1, uvec3_t)
    emit(59, ptr_input_uvec3_t, var_gid, 1)
    
    emit(29, arr_uint_t, uint_t)
    emit(30, struct_w_t, arr_uint_t)
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
    
    emit(32, ptr_sb_uint_t, 12, uint_t)
    emit(32, ptr_sb_float_t, 12, float_t)
    emit(32, ptr_pc_uint_t, 9, uint_t)
    
    emit(43, uint_t, c_uint_0, 0)
    emit(43, uint_t, c_uint_1, 1)
    emit(43, uint_t, c_uint_5, 5)
    emit(43, uint_t, c_uint_8, 8)
    emit(43, uint_t, c_uint_9, 9)
    emit(43, uint_t, c_uint_16, 16)
    emit(43, uint_t, c_uint_24, 24)
    emit(43, float_t, c_float_0, 0.0)
    
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
    
    emit(248, lbl_in_bounds)
    ptr_k = get_id()
    emit(65, ptr_pc_uint_t, ptr_k, var_pc, c_uint_1)
    v_k = get_id()
    emit(61, uint_t, v_k, ptr_k)
    
    v_num_blocks = get_id()
    emit(195, uint_t, v_num_blocks, v_k, c_uint_5)
    
    v_row_block_base = get_id()
    emit(132, uint_t, v_row_block_base, v_row, v_num_blocks)
    emit(249, lbl_b_header)
    
    emit(248, lbl_b_header)
    var_b = get_id()
    emit(245, uint_t, var_b, c_uint_0, lbl_in_bounds)
    var_total_acc = get_id()
    emit(245, float_t, var_total_acc, c_float_0, lbl_in_bounds)
    
    v_cond_b = get_id()
    emit(170, bool_t, v_cond_b, var_b, v_num_blocks)
    emit(246, lbl_b_exit, lbl_b_continue, 0)
    emit(250, v_cond_b, lbl_b_body, lbl_b_exit)
    
    emit(248, lbl_b_body)
    
    v_curr_blk = get_id()
    emit(128, uint_t, v_curr_blk, v_row_block_base, var_b)
    v_blk_u32_base = get_id()
    emit(132, uint_t, v_blk_u32_base, v_curr_blk, c_uint_9)
    
    ptr_d_u32 = get_id()
    emit(65, ptr_sb_uint_t, ptr_d_u32, var_w, c_uint_0, v_blk_u32_base)
    val_d_u32 = get_id()
    emit(61, uint_t, val_d_u32, ptr_d_u32)
    val_d = get_id()
    emit(124, float_t, val_d, val_d_u32)
    
    v_x_base = get_id()
    emit(196, uint_t, v_x_base, var_b, c_uint_5)
    
    curr_blk_acc = c_float_0
    for w in range(8):
        c_w_off = get_id()
        emit(43, uint_t, c_w_off, 1 + w)
        v_w_idx = get_id()
        emit(128, uint_t, v_w_idx, v_blk_u32_base, c_w_off)
        
        ptr_w = get_id()
        emit(65, ptr_sb_uint_t, ptr_w, var_w, c_uint_0, v_w_idx)
        val_w = get_id()
        emit(61, uint_t, val_w, ptr_w)
        
        shifts_left = [c_uint_24, c_uint_16, c_uint_8, None]
        for byte_idx in range(4):
            shl_val = val_w
            if shifts_left[byte_idx] is not None:
                tmp_shl = get_id()
                emit(196, uint_t, tmp_shl, val_w, shifts_left[byte_idx])
                shl_val = tmp_shl
            
            cast_i = get_id()
            emit(124, int_t, cast_i, shl_val)
            s_val = get_id()
            emit(194, int_t, s_val, cast_i, c_uint_24)
            q_float = get_id()
            emit(111, float_t, q_float, s_val)
            
            c_elem_off = get_id()
            emit(43, uint_t, c_elem_off, w * 4 + byte_idx)
            v_x_idx = get_id()
            emit(128, uint_t, v_x_idx, v_x_base, c_elem_off)
            
            ptr_x = get_id()
            emit(65, ptr_sb_float_t, ptr_x, var_x, c_uint_0, v_x_idx)
            val_x = get_id()
            emit(61, float_t, val_x, ptr_x)
            
            mul_val = get_id()
            emit(133, float_t, mul_val, q_float, val_x)
            next_acc = get_id()
            emit(129, float_t, next_acc, curr_blk_acc, mul_val)
            curr_blk_acc = next_acc

    scaled_blk_acc = get_id()
    emit(133, float_t, scaled_blk_acc, curr_blk_acc, val_d)
    
    next_total_acc = get_id()
    emit(129, float_t, next_total_acc, var_total_acc, scaled_blk_acc)
    
    emit(249, lbl_b_continue)
    
    emit(248, lbl_b_continue)
    next_b = get_id()
    emit(128, uint_t, next_b, var_b, c_uint_1)
    emit(249, lbl_b_header)
    
    emit(248, lbl_b_exit)
    ptr_y = get_id()
    emit(65, ptr_sb_float_t, ptr_y, var_y, c_uint_0, v_row)
    emit(62, ptr_y, var_total_acc)
    emit(249, lbl_out_of_bounds)
    
    emit(248, lbl_out_of_bounds)
    emit(253)
    emit(56)
    
    for idx in range(len(instructions)):
        w = instructions[idx]
        op = w & 0xFFFF
        if op == 245:
            res_id = instructions[idx+2]
            if res_id == var_b:
                instructions[idx] = 245 | (6 << 16)
                instructions[idx+1:idx+5] = [uint_t, var_b, c_uint_0, lbl_in_bounds, next_b, lbl_b_continue]
            elif res_id == var_total_acc:
                instructions[idx] = 245 | (6 << 16)
                instructions[idx+1:idx+5] = [float_t, var_total_acc, c_float_0, lbl_in_bounds, next_total_acc, lbl_b_continue]

    return make_spirv_header(id_counter, instructions)
