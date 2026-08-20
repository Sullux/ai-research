#!/usr/bin/env python3
import struct

def make_spirv_header(id_bound, instructions):
    return [0x07230203, 0x00010300, 0x00000000, id_bound, 0x00000000] + instructions

def generate_fused_gate_up_swiglu_q4():
    """
    Fused Gate+Up+SwiGLU Q4_0 GEMV:
    Bindings:
      0: gate_w (StorageBuffer)
      1: up_w   (StorageBuffer)
      2: x      (StorageBuffer)
      3: out    (StorageBuffer)
    Push constants: { uint M, uint K }
    LocalSize: (32, 1, 1) -> 1 Wave32 computes 1 row 'row = gl_WorkGroupID.x'.
    Loads X once into LDS, computes both gate_dot and up_dot concurrently,
    applies SwiGLU: out[row] = (gate_dot / (1 + exp(-gate_dot))) * up_dot.
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
    var_gid = get_id()
    var_lid = get_id()
    
    arr_uint_t = get_id()
    struct_w_t = get_id()
    ptr_sb_struct_w_t = get_id()
    var_gate_w = get_id()
    var_up_w = get_id()
    
    arr_float_t = get_id()
    struct_x_t = get_id()
    ptr_sb_struct_x_t = get_id()
    var_x = get_id()
    
    struct_out_t = get_id()
    ptr_sb_struct_out_t = get_id()
    var_out = get_id()
    
    struct_pc_t = get_id()
    ptr_pc_struct_t = get_id()
    var_pc = get_id()
    
    c_uint_32 = get_id()
    c_uint_160 = get_id()
    c_uint_1024 = get_id()
    
    arr_float_1024_t = get_id()
    ptr_workgroup_arr_float_1024_t = get_id()
    var_sx = get_id()
    ptr_workgroup_float_t = get_id()
    
    arr_uint_160_t = get_id()
    ptr_workgroup_arr_uint_160_t = get_id()
    var_sw_gate = get_id()
    var_sw_up = get_id()
    ptr_workgroup_uint_t = get_id()
    
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
    c_uint_64 = get_id()
    c_uint_96 = get_id()
    c_uint_128 = get_id()
    
    c_float_0 = get_id()
    c_float_1 = get_id()
    c_float_8 = get_id()
    
    c_uint_subgroup_scope = get_id()
    c_uint_barrier_scope = get_id()
    c_uint_barrier_semantics = get_id()
    
    glsl_ext = get_id()
    main_func = get_id()
    
    emit(17, 1)
    emit(17, 61)
    emit(17, 63)
    emit(11, glsl_ext, "GLSL.std.450")
    emit(14, 0, 1)
    emit(15, 5, main_func, "main", var_gid, var_lid)
    emit(16, main_func, 1, 32, 1, 1)
    
    # Decorations
    emit(71, struct_w_t, 2)
    emit(72, struct_w_t, 0, 35, 0)
    emit(71, arr_uint_t, 28, 4)
    emit(71, var_gate_w, 33, 0)
    emit(71, var_gate_w, 34, 0) # Binding 0
    
    emit(71, var_up_w, 33, 0)
    emit(71, var_up_w, 34, 1) # Binding 1
    
    emit(71, struct_x_t, 2)
    emit(72, struct_x_t, 0, 35, 0)
    emit(71, arr_float_t, 28, 4)
    emit(71, var_x, 33, 0)
    emit(71, var_x, 34, 2) # Binding 2
    
    emit(71, struct_out_t, 2)
    emit(72, struct_out_t, 0, 35, 0)
    emit(71, var_out, 33, 0)
    emit(71, var_out, 34, 3) # Binding 3
    
    emit(71, struct_pc_t, 3)
    emit(72, struct_pc_t, 0, 35, 0)
    emit(72, struct_pc_t, 1, 35, 4)
    
    emit(71, var_gid, 11, 26)
    emit(71, var_lid, 11, 28)
    
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
    
    emit(30, struct_out_t, arr_float_t)
    emit(32, ptr_sb_struct_out_t, 12, struct_out_t)
    
    emit(30, struct_pc_t, uint_t, uint_t)
    emit(32, ptr_pc_struct_t, 9, struct_pc_t)
    
    emit(43, uint_t, c_uint_32, 32)
    emit(43, uint_t, c_uint_160, 160)
    emit(43, uint_t, c_uint_1024, 1024)
    
    emit(29, arr_float_1024_t, float_t, c_uint_1024)
    emit(32, ptr_workgroup_arr_float_1024_t, 4, arr_float_1024_t)
    emit(32, ptr_workgroup_float_t, 4, float_t)
    
    emit(29, arr_uint_160_t, uint_t, c_uint_160)
    emit(32, ptr_workgroup_arr_uint_160_t, 4, arr_uint_160_t)
    emit(32, ptr_workgroup_uint_t, 4, uint_t)
    
    emit(32, ptr_sb_uint_t, 12, uint_t)
    emit(32, ptr_sb_float_t, 12, float_t)
    emit(32, ptr_pc_uint_t, 9, uint_t)
    
    emit(43, uint_t, c_uint_0, 0)
    emit(43, uint_t, c_uint_1, 1)
    emit(43, uint_t, c_uint_2, 2)
    emit(43, uint_t, c_uint_3, 3)
    emit(43, uint_t, c_uint_4, 4)
    emit(43, uint_t, c_uint_5, 5)
    emit(43, uint_t, c_uint_8, 8)
    emit(43, uint_t, c_uint_16, 16)
    emit(43, uint_t, c_uint_64, 64)
    emit(43, uint_t, c_uint_96, 96)
    emit(43, uint_t, c_uint_128, 128)
    emit(43, float_t, c_float_0, 0.0)
    emit(43, float_t, c_float_1, 1.0)
    emit(43, float_t, c_float_8, 8.0)
    emit(43, uint_t, c_uint_subgroup_scope, 3)
    emit(43, uint_t, c_uint_barrier_scope, 2)
    emit(43, uint_t, c_uint_barrier_semantics, 0x108)
    
    c_offsets = [get_id() for _ in range(8)]
    for i, cid in enumerate(c_offsets):
        emit(43, uint_t, cid, i * 4)
        
    c_count_4 = get_id()
    emit(43, uint_t, c_count_4, 4)
    
    emit(59, ptr_input_uvec3_t, var_gid, 1)
    emit(59, ptr_input_uvec3_t, var_lid, 1)
    emit(59, ptr_sb_struct_w_t, var_gate_w, 12)
    emit(59, ptr_sb_struct_w_t, var_up_w, 12)
    emit(59, ptr_sb_struct_x_t, var_x, 12)
    emit(59, ptr_sb_struct_out_t, var_out, 12)
    emit(59, ptr_pc_struct_t, var_pc, 9)
    emit(59, ptr_workgroup_arr_float_1024_t, var_sx, 4)
    emit(59, ptr_workgroup_arr_uint_160_t, var_sw_gate, 4)
    emit(59, ptr_workgroup_arr_uint_160_t, var_sw_up, 4)
    
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
    
    cond_valid_row = get_id()
    emit(170, bool_t, cond_valid_row, row, val_m)
    
    safe_row = get_id()
    emit(169, uint_t, safe_row, cond_valid_row, row, c_uint_0)
    
    c_uint_5_shift = get_id()
    emit(43, uint_t, c_uint_5_shift, 5)
    num_blocks = get_id()
    emit(138, uint_t, num_blocks, val_k, c_uint_5_shift)
    
    row_stride = get_id()
    emit(132, uint_t, row_stride, num_blocks, c_uint_5)
    
    row_base = get_id()
    emit(132, uint_t, row_base, safe_row, row_stride)
    
    lane_sw_base = get_id()
    emit(132, uint_t, lane_sw_base, lane, c_uint_5)
    
    sx_blk_base = get_id()
    emit(137, uint_t, sx_blk_base, lane, c_uint_5_shift)
    
    label_loop_head = get_id()
    label_loop_body = get_id()
    label_loop_continue = get_id()
    label_loop_merge = get_id()
    
    emit(249, label_loop_head)
    emit(248, label_loop_head)
    
    phi_chunk = get_id()
    phi_gate_sum = get_id()
    phi_up_sum = get_id()
    emit(244, uint_t, phi_chunk, c_uint_0, label_entry, get_id(), label_loop_continue)
    chunk_next_id = instructions[-3]
    emit(244, float_t, phi_gate_sum, c_float_0, label_entry, get_id(), label_loop_continue)
    gate_sum_next_id = instructions[-3]
    emit(244, float_t, phi_up_sum, c_float_0, label_entry, get_id(), label_loop_continue)
    up_sum_next_id = instructions[-3]
    
    emit(247, label_loop_merge, label_loop_continue, 0)
    cond_loop = get_id()
    emit(170, bool_t, cond_loop, phi_chunk, num_blocks)
    emit(245, cond_loop, label_loop_body, label_loop_merge)
    
    emit(248, label_loop_body)
    
    # 1. Load X into LDS sx_data (32 bursts)
    chunk_k_base = get_id()
    emit(137, uint_t, chunk_k_base, phi_chunk, c_uint_5_shift)
    
    for step in range(32):
        c_step_offset = get_id()
        emit(43, uint_t, c_step_offset, step * 32)
        sx_idx = get_id()
        emit(128, uint_t, sx_idx, lane, c_step_offset)
        glob_x_idx = get_id()
        emit(128, uint_t, glob_x_idx, chunk_k_base, sx_idx)
        gx_ptr = get_id()
        emit(65, ptr_sb_float_t, gx_ptr, var_x, c_uint_0, glob_x_idx)
        gx_val = get_id()
        emit(61, float_t, gx_val, gx_ptr)
        sx_ptr = get_id()
        emit(65, ptr_workgroup_float_t, sx_ptr, var_sx, sx_idx)
        emit(62, sx_ptr, gx_val)
        
    # 2. Load gate_w and up_w into LDS sw_data (5 contiguous 128-byte bursts each)
    chunk_x5 = get_id()
    emit(132, uint_t, chunk_x5, phi_chunk, c_uint_5)
    wave_w_base = get_id()
    emit(128, uint_t, wave_w_base, row_base, chunk_x5)
    
    for step, c_step_offset in enumerate([c_uint_0, c_uint_32, c_uint_64, c_uint_96, c_uint_128]):
        sw_store_idx = get_id()
        emit(128, uint_t, sw_store_idx, lane, c_step_offset)
        glob_w_idx = get_id()
        emit(128, uint_t, glob_w_idx, wave_w_base, sw_store_idx)
        
        # Load gate_w
        gate_ptr = get_id()
        emit(65, ptr_sb_uint_t, gate_ptr, var_gate_w, c_uint_0, glob_w_idx)
        gate_w_val = get_id()
        emit(61, uint_t, gate_w_val, gate_ptr)
        sw_gate_ptr = get_id()
        emit(65, ptr_workgroup_uint_t, sw_gate_ptr, var_sw_gate, sw_store_idx)
        emit(62, sw_gate_ptr, gate_w_val)
        
        # Load up_w
        up_ptr = get_id()
        emit(65, ptr_sb_uint_t, up_ptr, var_up_w, c_uint_0, glob_w_idx)
        up_w_val = get_id()
        emit(61, uint_t, up_w_val, up_ptr)
        sw_up_ptr = get_id()
        emit(65, ptr_workgroup_uint_t, sw_up_ptr, var_sw_up, sw_store_idx)
        emit(62, sw_up_ptr, up_w_val)
        
    emit(224, c_uint_barrier_scope, c_uint_barrier_scope, c_uint_barrier_semantics)
    
    # 3. Compute both gate and up dots concurrently
    gate_d_ptr = get_id()
    emit(65, ptr_workgroup_uint_t, gate_d_ptr, var_sw_gate, lane_sw_base)
    gate_d_u32 = get_id()
    emit(61, uint_t, gate_d_u32, gate_d_ptr)
    gate_d_f = get_id()
    emit(124, float_t, gate_d_f, gate_d_u32)
    
    up_d_ptr = get_id()
    emit(65, ptr_workgroup_uint_t, up_d_ptr, var_sw_up, lane_sw_base)
    up_d_u32 = get_id()
    emit(61, uint_t, up_d_u32, up_d_ptr)
    up_d_f = get_id()
    emit(124, float_t, up_d_f, up_d_u32)
    
    gate_acc_id = get_id()
    emit(43, float_t, gate_acc_id, 0.0)
    up_acc_id = get_id()
    emit(43, float_t, up_acc_id, 0.0)
    
    for word_idx in range(4):
        sw_word_idx = get_id()
        c_w_idx = get_id()
        emit(43, uint_t, c_w_idx, word_idx + 1)
        emit(128, uint_t, sw_word_idx, lane_sw_base, c_w_idx)
        
        sw_gate_q_ptr = get_id()
        emit(65, ptr_workgroup_uint_t, sw_gate_q_ptr, var_sw_gate, sw_word_idx)
        gate_q_val = get_id()
        emit(61, uint_t, gate_q_val, sw_gate_q_ptr)
        
        sw_up_q_ptr = get_id()
        emit(65, ptr_workgroup_uint_t, sw_up_q_ptr, var_sw_up, sw_word_idx)
        up_q_val = get_id()
        emit(61, uint_t, up_q_val, sw_up_q_ptr)
        
        for nib_idx in range(8):
            # Gate nibble
            g_nib = get_id()
            emit(201, uint_t, g_nib, gate_q_val, c_offsets[nib_idx], c_count_4)
            g_nib_f = get_id()
            emit(111, float_t, g_nib_f, g_nib)
            g_unbiased = get_id()
            emit(131, float_t, g_unbiased, g_nib_f, c_float_8)
            
            # Up nibble
            u_nib = get_id()
            emit(201, uint_t, u_nib, up_q_val, c_offsets[nib_idx], c_count_4)
            u_nib_f = get_id()
            emit(111, float_t, u_nib_f, u_nib)
            u_unbiased = get_id()
            emit(131, float_t, u_unbiased, u_nib_f, c_float_8)
            
            # Load X once from shared memory
            sx_elem_offset = get_id()
            c_elem_offset = get_id()
            emit(43, uint_t, c_elem_offset, word_idx * 8 + nib_idx)
            emit(128, uint_t, sx_elem_offset, sx_blk_base, c_elem_offset)
            
            sx_read_ptr = get_id()
            emit(65, ptr_workgroup_float_t, sx_read_ptr, var_sx, sx_elem_offset)
            x_val = get_id()
            emit(61, float_t, x_val, sx_read_ptr)
            
            # FMA gate
            g_prod = get_id()
            emit(133, float_t, g_prod, g_unbiased, x_val)
            new_g_acc = get_id()
            emit(129, float_t, new_g_acc, gate_acc_id, g_prod)
            gate_acc_id = new_g_acc
            
            # FMA up
            u_prod = get_id()
            emit(133, float_t, u_prod, u_unbiased, x_val)
            new_u_acc = get_id()
            emit(129, float_t, new_u_acc, up_acc_id, u_prod)
            up_acc_id = new_u_acc
            
    gate_blk_sum = get_id()
    emit(133, float_t, gate_blk_sum, gate_acc_id, gate_d_f)
    
    up_blk_sum = get_id()
    emit(133, float_t, up_blk_sum, up_acc_id, up_d_f)
    
    new_phi_gate = get_id()
    emit(129, float_t, new_phi_gate, phi_gate_sum, gate_blk_sum)
    
    new_phi_up = get_id()
    emit(129, float_t, new_phi_up, phi_up_sum, up_blk_sum)
    
    new_phi_chunk = get_id()
    emit(128, uint_t, new_phi_chunk, phi_chunk, c_uint_32)
    
    emit(224, c_uint_barrier_scope, c_uint_barrier_scope, c_uint_barrier_semantics)
    
    emit(249, label_loop_continue)
    emit(248, label_loop_continue)
    emit(249, label_loop_head)
    
    instructions[instructions.index(chunk_next_id)] = new_phi_chunk
    instructions[instructions.index(gate_sum_next_id)] = new_phi_gate
    instructions[instructions.index(up_sum_next_id)] = new_phi_up
    
    emit(248, label_loop_merge)
    
    # Subgroup Reductions
    final_gate = get_id()
    emit(334, float_t, final_gate, c_uint_subgroup_scope, c_uint_0, phi_gate_sum)
    
    final_up = get_id()
    emit(334, float_t, final_up, c_uint_subgroup_scope, c_uint_0, phi_up_sum)
    
    # SwiGLU: (gate * sigmoid(gate)) * up = (gate / (1 + exp(-gate))) * up
    neg_gate = get_id()
    emit(127, float_t, neg_gate, final_gate) # OpFNegate
    
    exp_neg_gate = get_id()
    emit(12, float_t, exp_neg_gate, glsl_ext, 19, neg_gate) # OpExtInst Exp
    
    denom = get_id()
    emit(129, float_t, denom, c_float_1, exp_neg_gate)
    
    sig_gate = get_id()
    emit(136, float_t, sig_gate, c_float_1, denom) # OpFDiv 1.0 / (1.0 + exp(-gate))
    
    silu_gate = get_id()
    emit(133, float_t, silu_gate, final_gate, sig_gate)
    
    swiglu_out = get_id()
    emit(133, float_t, swiglu_out, silu_gate, final_up)
    
    # Write out[row]
    cond_lane0 = get_id()
    emit(170, bool_t, cond_lane0, lane, c_uint_1)
    
    cond_write = get_id()
    emit(164, bool_t, cond_write, cond_lane0, cond_valid_row)
    
    lbl_write_out = get_id()
    lbl_end = get_id()
    emit(245, cond_write, lbl_write_out, lbl_end)
    
    emit(248, lbl_write_out)
    out_ptr = get_id()
    emit(65, ptr_sb_float_t, out_ptr, var_out, c_uint_0, row)
    emit(62, out_ptr, swiglu_out)
    emit(249, lbl_end)
    
    emit(248, lbl_end)
    emit(253)
    emit(56)
    
    return make_spirv_header(id_counter, instructions)

if __name__ == '__main__':
    code = generate_fused_gate_up_swiglu_q4()
    with open("src/gpu/shaders_fused_q4.zig", "w") as f:
        f.write('pub const FUSED_GATE_UP_SWIGLU_Q4_SPIRV = [_]u32{\n')
        for i, word in enumerate(code):
            f.write(f' 0x{word:08x},')
            if (i + 1) % 24 == 0:
                f.write('\n')
        if len(code) % 24 != 0:
            f.write('\n')
        f.write('};\n')
    print(f"Generated fused gate+up+swiglu Q4 SPIR-V: {len(code)} words ({len(code)*4} bytes)")
