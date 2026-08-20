#!/usr/bin/env python3
import struct

def make_spirv_header(id_bound, instructions):
    return [0x07230203, 0x00010300, 0x00000000, id_bound, 0x00000000] + instructions

def generate_coop_q4_fast():
    """
    Optimized Wave-Cooperative Q4_0 GEMV:
    LocalSize = (32, 1, 1)
    Grid = (M, 1, 1)
    Uses OpBitFieldUExtract (Opcode 201) to extract 4-bit nibbles in a single instruction.
    Shared memory tree reduction across 32 threads.
    """
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
    var_gid = get_id() # WorkGroupID
    var_lid = get_id() # LocalInvocationID
    
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
    
    c_uint_32 = get_id()
    arr_float_32_t = get_id()
    ptr_workgroup_arr_float_t = get_id()
    var_sdata = get_id()
    ptr_workgroup_float_t = get_id()
    
    ptr_sb_uint_t = get_id()
    ptr_sb_float_t = get_id()
    ptr_pc_uint_t = get_id()
    
    c_uint_0 = get_id()
    c_uint_1 = get_id()
    c_uint_2 = get_id()
    c_uint_3 = get_id()
    c_uint_4 = get_id()
    c_uint_5 = get_id()
    c_uint_8 = get_id()
    c_uint_16 = get_id()
    c_float_0 = get_id()
    c_float_8 = get_id()
    
    c_uint_barrier_scope = get_id()
    c_uint_barrier_semantics = get_id()
    
    main_func = get_id()
    
    emit(17, 1) # OpCapability Shader
    emit(14, 0, 1) # OpMemoryModel Logical GLSL450
    emit(15, 5, main_func, "main", var_gid, var_lid)
    emit(16, main_func, 1, 32, 1, 1) # LocalSize 32 1 1
    
    # Decorations
    emit(71, struct_w_t, 2)
    emit(72, struct_w_t, 0, 35, 0)
    emit(71, arr_uint_t, 28, 4)
    emit(71, var_w, 33, 0)
    emit(71, var_w, 34, 0)
    
    emit(71, struct_x_t, 2)
    emit(72, struct_x_t, 0, 35, 0)
    emit(71, arr_float_t, 28, 4)
    emit(71, var_x, 33, 0)
    emit(71, var_x, 34, 1)
    
    emit(71, struct_y_t, 2)
    emit(72, struct_y_t, 0, 35, 0)
    emit(71, var_y, 33, 0)
    emit(71, var_y, 34, 2)
    
    emit(71, struct_pc_t, 3)
    emit(72, struct_pc_t, 0, 35, 0)
    emit(72, struct_pc_t, 1, 35, 4)
    
    emit(71, var_gid, 11, 26)
    emit(71, var_lid, 11, 28)
    
    # Types
    emit(19, void_t)
    emit(33, func_t, void_t)
    emit(21, float_t, 32)
    emit(21, uint_t, 32, 0)
    emit(21, int_t, 32, 1)
    emit(20, bool_t)
    emit(23, uvec3_t, uint_t, 3)
    
    emit(32, ptr_input_uvec3_t, 1, uvec3_t)
    emit(28, arr_uint_t, uint_t)
    emit(30, struct_w_t, arr_uint_t)
    emit(32, ptr_sb_struct_w_t, 12, struct_w_t)
    
    emit(28, arr_float_t, float_t)
    emit(30, struct_x_t, arr_float_t)
    emit(32, ptr_sb_struct_x_t, 12, struct_x_t)
    
    emit(30, struct_y_t, arr_float_t)
    emit(32, ptr_sb_struct_y_t, 12, struct_y_t)
    
    emit(30, struct_pc_t, uint_t, uint_t)
    emit(32, ptr_pc_struct_t, 9, struct_pc_t)
    
    emit(43, uint_t, c_uint_32, 32)
    emit(29, arr_float_32_t, float_t, c_uint_32)
    emit(32, ptr_workgroup_arr_float_t, 4, arr_float_32_t)
    emit(32, ptr_workgroup_float_t, 4, float_t)
    
    emit(32, ptr_sb_uint_t, 12, uint_t)
    emit(32, ptr_sb_float_t, 12, float_t)
    emit(32, ptr_pc_uint_t, 9, uint_t)
    
    # Constants
    emit(43, uint_t, c_uint_0, 0)
    emit(43, uint_t, c_uint_1, 1)
    emit(43, uint_t, c_uint_2, 2)
    emit(43, uint_t, c_uint_3, 3)
    emit(43, uint_t, c_uint_4, 4)
    emit(43, uint_t, c_uint_5, 5)
    emit(43, uint_t, c_uint_8, 8)
    emit(43, uint_t, c_uint_16, 16)
    emit(43, float_t, c_float_0, 0.0)
    emit(43, float_t, c_float_8, 8.0)
    emit(43, uint_t, c_uint_barrier_scope, 2)
    emit(43, uint_t, c_uint_barrier_semantics, 0x108)
    
    # Pre-declare offset constants for OpBitFieldUExtract
    c_offsets = [get_id() for _ in range(8)]
    for i, cid in enumerate(c_offsets):
        emit(43, uint_t, cid, i * 4)
        
    c_count_4 = get_id()
    emit(43, uint_t, c_count_4, 4)
    
    # Global Variables
    emit(59, ptr_input_uvec3_t, var_gid, 1)
    emit(59, ptr_input_uvec3_t, var_lid, 1)
    emit(59, ptr_sb_struct_w_t, var_w, 12)
    emit(59, ptr_sb_struct_x_t, var_x, 12)
    emit(59, ptr_sb_struct_y_t, var_y, 12)
    emit(59, ptr_pc_struct_t, var_pc, 9)
    emit(59, ptr_workgroup_arr_float_t, var_sdata, 4)
    
    # Function Body
    emit(54, void_t, main_func, 0, func_t)
    label_entry = get_id()
    emit(248, label_entry)
    
    gid_x_ptr = get_id()
    emit(65, ptr_input_uvec3_t, gid_x_ptr, var_gid, c_uint_0)
    row = get_id()
    emit(61, uint_t, row, gid_x_ptr)
    
    lid_x_ptr = get_id()
    emit(65, ptr_input_uvec3_t, lid_x_ptr, var_lid, c_uint_0)
    lane = get_id()
    emit(61, uint_t, lane, lid_x_ptr)
    
    m_ptr = get_id()
    emit(65, ptr_pc_uint_t, m_ptr, var_pc, c_uint_0)
    val_m = get_id()
    emit(61, uint_t, val_m, m_ptr)
    
    k_ptr = get_id()
    emit(65, ptr_pc_uint_t, k_ptr, var_pc, c_uint_1)
    val_k = get_id()
    emit(61, uint_t, val_k, k_ptr)
    
    cond_row = get_id()
    emit(170, bool_t, cond_row, row, val_m)
    label_body = get_id()
    label_return = get_id()
    emit(245, cond_row, label_body, label_return)
    
    emit(248, label_body)
    c_uint_5_shift = get_id()
    emit(43, uint_t, c_uint_5_shift, 5)
    num_blocks = get_id()
    emit(138, uint_t, num_blocks, val_k, c_uint_5_shift)
    
    row_stride = get_id()
    emit(132, uint_t, row_stride, num_blocks, c_uint_5)
    
    row_base = get_id()
    emit(132, uint_t, row_base, row, row_stride)
    
    label_loop_head = get_id()
    label_loop_body = get_id()
    label_loop_continue = get_id()
    label_loop_merge = get_id()
    
    emit(249, label_loop_head)
    emit(248, label_loop_head)
    
    phi_blk = get_id()
    phi_sum = get_id()
    emit(244, uint_t, phi_blk, lane, label_body, get_id(), label_loop_continue)
    blk_next_id = instructions[-3]
    emit(244, float_t, phi_sum, c_float_0, label_body, get_id(), label_loop_continue)
    sum_next_id = instructions[-3]
    
    emit(247, label_loop_merge, label_loop_continue, 0)
    cond_loop = get_id()
    emit(170, bool_t, cond_loop, phi_blk, num_blocks)
    emit(245, cond_loop, label_loop_body, label_loop_merge)
    
    emit(248, label_loop_body)
    
    blk_x5 = get_id()
    emit(132, uint_t, blk_x5, phi_blk, c_uint_5)
    blk_offset = get_id()
    emit(128, uint_t, blk_offset, row_base, blk_x5)
    
    d_ptr = get_id()
    emit(65, ptr_sb_uint_t, d_ptr, var_w, c_uint_0, blk_offset)
    d_u32 = get_id()
    emit(61, uint_t, d_u32, d_ptr)
    d_float = get_id()
    emit(124, float_t, d_float, d_u32)
    
    k_base = get_id()
    emit(137, uint_t, k_base, phi_blk, c_uint_5_shift)
    
    current_acc = get_id()
    emit(43, float_t, current_acc, 0.0)
    
    for word_idx in range(4):
        w_offset = get_id()
        c_w_idx = get_id()
        emit(43, uint_t, c_w_idx, word_idx + 1)
        emit(128, uint_t, w_offset, blk_offset, c_w_idx)
        
        q_ptr = get_id()
        emit(65, ptr_sb_uint_t, q_ptr, var_w, c_uint_0, w_offset)
        q_val = get_id()
        emit(61, uint_t, q_val, q_ptr)
        
        for nib_idx in range(8):
            # OpBitFieldUExtract: Opcode 201 (extract 4 bits directly)
            nib = get_id()
            emit(201, uint_t, nib, q_val, c_offsets[nib_idx], c_count_4)
            
            nib_f = get_id()
            emit(111, float_t, nib_f, nib)
            nib_unbiased = get_id()
            emit(131, float_t, nib_unbiased, nib_f, c_float_8)
            
            k_elem = get_id()
            c_elem_offset = get_id()
            emit(43, uint_t, c_elem_offset, word_idx * 8 + nib_idx)
            emit(128, uint_t, k_elem, k_base, c_elem_offset)
            
            x_ptr = get_id()
            emit(65, ptr_sb_float_t, x_ptr, var_x, c_uint_0, k_elem)
            x_val = get_id()
            emit(61, float_t, x_val, x_ptr)
            
            prod = get_id()
            emit(133, float_t, prod, nib_unbiased, x_val)
            
            new_acc = get_id()
            emit(129, float_t, new_acc, current_acc, prod)
            current_acc = new_acc

    blk_sum = get_id()
    emit(133, float_t, blk_sum, current_acc, d_float)
    
    new_phi_sum = get_id()
    emit(129, float_t, new_phi_sum, phi_sum, blk_sum)
    
    new_phi_blk = get_id()
    emit(128, uint_t, new_phi_blk, phi_blk, c_uint_32)
    
    emit(249, label_loop_continue)
    emit(248, label_loop_continue)
    emit(249, label_loop_head)
    
    instructions[instructions.index(blk_next_id)] = new_phi_blk
    instructions[instructions.index(sum_next_id)] = new_phi_sum
    
    emit(248, label_loop_merge)
    
    sdata_lane_ptr = get_id()
    emit(65, ptr_workgroup_float_t, sdata_lane_ptr, var_sdata, lane)
    emit(62, sdata_lane_ptr, phi_sum)
    emit(224, c_uint_barrier_scope, c_uint_barrier_scope, c_uint_barrier_semantics)
    
    for stride in [16, 8, 4, 2, 1]:
        c_s = get_id()
        emit(43, uint_t, c_s, stride)
        cond_s = get_id()
        emit(170, bool_t, cond_s, lane, c_s)
        
        lbl_red_then = get_id()
        lbl_red_merge = get_id()
        emit(245, cond_s, lbl_red_then, lbl_red_merge)
        
        emit(248, lbl_red_then)
        other_lane = get_id()
        emit(128, uint_t, other_lane, lane, c_s)
        other_ptr = get_id()
        emit(65, ptr_workgroup_float_t, other_ptr, var_sdata, other_lane)
        other_val = get_id()
        emit(61, float_t, other_val, other_ptr)
        
        my_ptr = get_id()
        emit(65, ptr_workgroup_float_t, my_ptr, var_sdata, lane)
        my_val = get_id()
        emit(61, float_t, my_val, my_ptr)
        
        reduced = get_id()
        emit(129, float_t, reduced, my_val, other_val)
        emit(62, my_ptr, reduced)
        emit(249, lbl_red_merge)
        
        emit(248, lbl_red_merge)
        emit(224, c_uint_barrier_scope, c_uint_barrier_scope, c_uint_barrier_semantics)
        
    cond_lane0 = get_id()
    emit(170, bool_t, cond_lane0, lane, c_uint_1)
    lbl_write = get_id()
    lbl_write_merge = get_id()
    emit(245, cond_lane0, lbl_write, lbl_write_merge)
    
    emit(248, lbl_write)
    sdata0_ptr = get_id()
    emit(65, ptr_workgroup_float_t, sdata0_ptr, var_sdata, c_uint_0)
    final_row_sum = get_id()
    emit(61, float_t, final_row_sum, sdata0_ptr)
    
    y_ptr = get_id()
    emit(65, ptr_sb_float_t, y_ptr, var_y, c_uint_0, row)
    emit(62, y_ptr, final_row_sum)
    emit(249, lbl_write_merge)
    
    emit(248, lbl_write_merge)
    emit(249, label_return)
    
    emit(248, label_return)
    emit(253)
    emit(56)
    
    return make_spirv_header(id_counter, instructions)

if __name__ == '__main__':
    q4_code = generate_coop_q4_fast()
    with open("src/gpu/shaders_q4.zig", "w") as f:
        f.write('pub const GEMV_Q4_SPIRV = [_]u32{\n')
        for i, word in enumerate(q4_code):
            f.write(f' 0x{word:08x},')
            if (i + 1) % 16 == 0:
                f.write('\n')
        if len(q4_code) % 16 != 0:
            f.write('\n')
        f.write('};\n')
    print(f"Generated and wrote fast Q4 SPIR-V: {len(q4_code)} words ({len(q4_code)*4} bytes)")
