#!/usr/bin/env python3
import struct

def make_spirv_header(id_bound, instructions):
    return [0x07230203, 0x00010300, 0x00000000, id_bound, 0x00000000] + instructions

def generate_decode_attention():
    """
    Ultra-Fast Single-Wave GQA Decode Attention on RDNA 3.5:
    Workgroup: (32, 1, 1) -> 1 Wave32 per Query Head.
    Workgroups: (16, 1, 1) -> 16 Query Heads (GQA 16:8 = 2 query heads per KV head).
    Head Dim: 256 (32 threads * 8 elements per thread).
    LDS: s_scores[512] (2 KB).
    Coalesced vector loads, OpGroupNonUniformFAdd 1-cycle wave reductions.
    
    Bindings:
      0: Q (StorageBuffer, [16 * 256] floats)
      1: K (StorageBuffer, [MaxSlots * 8 * 256] floats)
      2: V (StorageBuffer, [MaxSlots * 8 * 256] floats)
      3: ActiveSlots (StorageBuffer, [N] uints)
      4: Out (StorageBuffer, [16 * 256] floats)
    Push constants: { uint N_active, uint kv_stride_elements }
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
    
    arr_float_t = get_id()
    struct_buf_f32_t = get_id()
    ptr_sb_f32_buf_t = get_id()
    var_q = get_id()
    var_k = get_id()
    var_v = get_id()
    
    arr_uint_t = get_id()
    struct_buf_u32_t = get_id()
    ptr_sb_u32_buf_t = get_id()
    var_active_slots = get_id()
    
    var_out = get_id()
    
    struct_pc_t = get_id()
    ptr_pc_struct_t = get_id()
    var_pc = get_id()
    
    c_uint_512 = get_id()
    arr_float_512_t = get_id()
    ptr_workgroup_arr_float_512_t = get_id()
    var_scores = get_id()
    ptr_workgroup_float_t = get_id()
    
    ptr_sb_uint_t = get_id()
    ptr_sb_float_t = get_id()
    ptr_pc_uint_t = get_id()
    
    c_uint_0 = get_id()
    c_uint_1 = get_id()
    c_uint_2 = get_id()
    c_uint_3 = get_id()
    c_uint_4 = get_id()
    c_uint_8 = get_id()
    c_uint_256 = get_id()
    
    c_float_0 = get_id()
    c_float_1 = get_id()
    c_float_neg_inf = get_id()
    
    c_uint_barrier_scope = get_id()
    c_uint_barrier_semantics = get_id()
    c_uint_subgroup_scope = get_id()
    
    glsl_ext = get_id()
    main_func = get_id()

    # Capabilities & Extensions
    emit(17, 0)
    emit(17, 61)
    emit(17, 63)
    emit(11, glsl_ext, "GLSL.std.450")
    emit(14, 0, 1)
    emit(15, 5, main_func, "main", var_gid, var_lid)
    emit(16, main_func, 1, 32, 1, 1)
    
    # Decorations
    for var_id, binding in [(var_q, 0), (var_k, 1), (var_v, 2), (var_active_slots, 3), (var_out, 4)]:
        emit(71, var_id, 33, 0)
        emit(71, var_id, 34, binding)
        
    emit(71, struct_buf_f32_t, 2)
    emit(72, struct_buf_f32_t, 0, 35, 0)
    emit(71, arr_float_t, 28, 4)
    
    emit(71, struct_buf_u32_t, 2)
    emit(72, struct_buf_u32_t, 0, 35, 0)
    emit(71, arr_uint_t, 28, 4)
    
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
    emit(28, arr_float_t, float_t)
    emit(30, struct_buf_f32_t, arr_float_t)
    emit(32, ptr_sb_f32_buf_t, 12, struct_buf_f32_t)
    
    emit(28, arr_uint_t, uint_t)
    emit(30, struct_buf_u32_t, arr_uint_t)
    emit(32, ptr_sb_u32_buf_t, 12, struct_buf_u32_t)
    
    emit(30, struct_pc_t, uint_t, uint_t)
    emit(32, ptr_pc_struct_t, 9, struct_pc_t)
    
    emit(43, uint_t, c_uint_512, 512)
    emit(29, arr_float_512_t, float_t, c_uint_512)
    emit(32, ptr_workgroup_arr_float_512_t, 4, arr_float_512_t)
    emit(32, ptr_workgroup_float_t, 4, float_t)
    
    emit(32, ptr_sb_uint_t, 12, uint_t)
    emit(32, ptr_sb_float_t, 12, float_t)
    emit(32, ptr_pc_uint_t, 9, uint_t)
    
    emit(43, uint_t, c_uint_0, 0)
    emit(43, uint_t, c_uint_1, 1)
    emit(43, uint_t, c_uint_2, 2)
    emit(43, uint_t, c_uint_3, 3)
    emit(43, uint_t, c_uint_4, 4)
    emit(43, uint_t, c_uint_8, 8)
    emit(43, uint_t, c_uint_256, 256)
    
    emit(43, float_t, c_float_0, 0.0)
    emit(43, float_t, c_float_1, 1.0)
    emit(43, float_t, c_float_neg_inf, -100000.0)
    
    emit(43, uint_t, c_uint_barrier_scope, 2)
    emit(43, uint_t, c_uint_barrier_semantics, 0x100 | 0x40 | 0x80)
    emit(43, uint_t, c_uint_subgroup_scope, 3)
    
    emit(59, ptr_input_uvec3_t, var_gid, 1)
    emit(59, ptr_input_uvec3_t, var_lid, 1)
    emit(59, ptr_sb_f32_buf_t, var_q, 12)
    emit(59, ptr_sb_f32_buf_t, var_k, 12)
    emit(59, ptr_sb_f32_buf_t, var_v, 12)
    emit(59, ptr_sb_u32_buf_t, var_active_slots, 12)
    emit(59, ptr_sb_f32_buf_t, var_out, 12)
    emit(59, ptr_pc_struct_t, var_pc, 9)
    emit(59, ptr_workgroup_arr_float_512_t, var_scores, 4)
    
    emit(54, void_t, main_func, 0, func_t)
    label_entry = get_id()
    emit(248, label_entry)
    
    # head_id = gid.x
    gid_x_ptr = get_id()
    emit(65, ptr_input_uvec3_t, gid_x_ptr, var_gid, c_uint_0)
    head_id = get_id()
    emit(61, uint_t, head_id, gid_x_ptr)
    
    # lane = lid.x
    lid_x_ptr = get_id()
    emit(65, ptr_input_uvec3_t, lid_x_ptr, var_lid, c_uint_0)
    lane = get_id()
    emit(61, uint_t, lane, lid_x_ptr)
    
    # N_active from PC
    n_ptr = get_id()
    emit(65, ptr_pc_uint_t, n_ptr, var_pc, c_uint_0)
    n_active = get_id()
    emit(61, uint_t, n_active, n_ptr)
    
    # kv_stride from PC
    stride_ptr = get_id()
    emit(65, ptr_pc_uint_t, stride_ptr, var_pc, c_uint_1)
    kv_stride = get_id()
    emit(61, uint_t, kv_stride, stride_ptr)
    
    # kv_head = head_id / 2 (GQA group size 2)
    c_uint_1_shift = get_id()
    emit(43, uint_t, c_uint_1_shift, 1)
    kv_head = get_id()
    emit(138, uint_t, kv_head, head_id, c_uint_1_shift)
    
    # Q head base offset = head_id * 256
    c_uint_8_shift = get_id()
    emit(43, uint_t, c_uint_8_shift, 8)
    q_head_base = get_id()
    emit(137, uint_t, q_head_base, head_id, c_uint_8_shift)
    
    # lane elem offset = lane * 8
    c_uint_3_shift = get_id()
    emit(43, uint_t, c_uint_3_shift, 3)
    lane_elem_off = get_id()
    emit(137, uint_t, lane_elem_off, lane, c_uint_3_shift)
    
    # Load 8 elements of Q into registers
    reg_q_vals = []
    for j in range(8):
        c_j = get_id()
        emit(43, uint_t, c_j, j)
        q_idx = get_id()
        emit(128, uint_t, q_idx, q_head_base, lane_elem_off)
        q_idx_j = get_id()
        emit(128, uint_t, q_idx_j, q_idx, c_j)
        q_ptr = get_id()
        emit(65, ptr_sb_float_t, q_ptr, var_q, c_uint_0, q_idx_j)
        q_val = get_id()
        emit(61, float_t, q_val, q_ptr)
        reg_q_vals.append(q_val)
        
    # KV head elem base = kv_head * 256 + lane * 8
    kv_head_elem_base = get_id()
    emit(137, uint_t, kv_head_elem_base, kv_head, c_uint_8_shift)
    kv_lane_off = get_id()
    emit(128, uint_t, kv_lane_off, kv_head_elem_base, lane_elem_off)
    
    # 1. Compute Dot Products across N_active slots
    label_dot_head = get_id()
    label_dot_body = get_id()
    label_dot_continue = get_id()
    label_dot_merge = get_id()
    
    emit(249, label_dot_head)
    emit(248, label_dot_head)
    
    phi_s = get_id()
    emit(244, uint_t, phi_s, c_uint_0, label_entry, get_id(), label_dot_continue)
    s_next_id = instructions[-3]
    
    emit(247, label_dot_merge, label_dot_continue, 0)
    cond_dot = get_id()
    emit(170, bool_t, cond_dot, phi_s, n_active)
    emit(245, cond_dot, label_dot_body, label_dot_merge)
    
    emit(248, label_dot_body)
    
    slot_ptr = get_id()
    emit(65, ptr_sb_uint_t, slot_ptr, var_active_slots, c_uint_0, phi_s)
    slot_idx = get_id()
    emit(61, uint_t, slot_idx, slot_ptr)
    
    slot_k_base = get_id()
    emit(132, uint_t, slot_k_base, slot_idx, kv_stride)
    k_elem_base = get_id()
    emit(128, uint_t, k_elem_base, slot_k_base, kv_lane_off)
    
    dot_acc = c_float_0
    for j in range(8):
        c_j = get_id()
        emit(43, uint_t, c_j, j)
        k_idx_j = get_id()
        emit(128, uint_t, k_idx_j, k_elem_base, c_j)
        k_ptr = get_id()
        emit(65, ptr_sb_float_t, k_ptr, var_k, c_uint_0, k_idx_j)
        k_val = get_id()
        emit(61, float_t, k_val, k_ptr)
        
        prod = get_id()
        emit(133, float_t, prod, reg_q_vals[j], k_val)
        new_dot = get_id()
        emit(129, float_t, new_dot, dot_acc, prod)
        dot_acc = new_dot
        
    wave_dot = get_id()
    emit(334, float_t, wave_dot, c_uint_subgroup_scope, c_uint_0, dot_acc)
    
    cond_lane0 = get_id()
    emit(170, bool_t, cond_lane0, lane, c_uint_1)
    lbl_w_score = get_id()
    lbl_skip_score = get_id()
    emit(245, cond_lane0, lbl_w_score, lbl_skip_score)
    emit(248, lbl_w_score)
    score_ptr = get_id()
    emit(65, ptr_workgroup_float_t, score_ptr, var_scores, phi_s)
    emit(62, score_ptr, wave_dot)
    emit(249, lbl_skip_score)
    
    emit(248, lbl_skip_score)
    new_phi_s = get_id()
    emit(128, uint_t, new_phi_s, phi_s, c_uint_1)
    
    emit(249, label_dot_continue)
    emit(248, label_dot_continue)
    emit(249, label_dot_head)
    
    instructions[instructions.index(s_next_id)] = new_phi_s
    emit(248, label_dot_merge)
    
    emit(224, c_uint_barrier_scope, c_uint_barrier_scope, c_uint_barrier_semantics)
    
    # 2. Softmax in LDS by Lane 0
    lbl_softmax = get_id()
    lbl_after_softmax = get_id()
    emit(245, cond_lane0, lbl_softmax, lbl_after_softmax)
    emit(248, lbl_softmax)
    
    # Find max score
    label_sm_max_head = get_id()
    label_sm_max_body = get_id()
    label_sm_max_cont = get_id()
    label_sm_max_merge = get_id()
    
    emit(249, label_sm_max_head)
    emit(248, label_sm_max_head)
    phi_max_i = get_id()
    phi_max_v = get_id()
    emit(244, uint_t, phi_max_i, c_uint_0, lbl_softmax, get_id(), label_sm_max_cont)
    max_i_next = instructions[-3]
    emit(244, float_t, phi_max_v, c_float_neg_inf, lbl_softmax, get_id(), label_sm_max_cont)
    max_v_next = instructions[-3]
    
    emit(247, label_sm_max_merge, label_sm_max_cont, 0)
    cond_max = get_id()
    emit(170, bool_t, cond_max, phi_max_i, n_active)
    emit(245, cond_max, label_sm_max_body, label_sm_max_merge)
    emit(248, label_sm_max_body)
    
    cur_sc_ptr = get_id()
    emit(65, ptr_workgroup_float_t, cur_sc_ptr, var_scores, phi_max_i)
    cur_sc = get_id()
    emit(61, float_t, cur_sc, cur_sc_ptr)
    
    new_max = get_id()
    emit(12, float_t, new_max, glsl_ext, 40, phi_max_v, cur_sc) # FMax
    
    new_max_i = get_id()
    emit(128, uint_t, new_max_i, phi_max_i, c_uint_1)
    
    emit(249, label_sm_max_cont)
    emit(248, label_sm_max_cont)
    emit(249, label_sm_max_head)
    instructions[instructions.index(max_i_next)] = new_max_i
    instructions[instructions.index(max_v_next)] = new_max
    
    emit(248, label_sm_max_merge)
    
    # Compute exp and sum
    label_exp_head = get_id()
    label_exp_body = get_id()
    label_exp_cont = get_id()
    label_exp_merge = get_id()
    
    emit(249, label_exp_head)
    emit(248, label_exp_head)
    phi_exp_i = get_id()
    phi_exp_sum = get_id()
    emit(244, uint_t, phi_exp_i, c_uint_0, label_sm_max_merge, get_id(), label_exp_cont)
    exp_i_next = instructions[-3]
    emit(244, float_t, phi_exp_sum, c_float_0, label_sm_max_merge, get_id(), label_exp_cont)
    exp_sum_next = instructions[-3]
    
    emit(247, label_exp_merge, label_exp_cont, 0)
    cond_exp = get_id()
    emit(170, bool_t, cond_exp, phi_exp_i, n_active)
    emit(245, cond_exp, label_exp_body, label_exp_merge)
    emit(248, label_exp_body)
    
    sc_ptr_e = get_id()
    emit(65, ptr_workgroup_float_t, sc_ptr_e, var_scores, phi_exp_i)
    sc_val_e = get_id()
    emit(61, float_t, sc_val_e, sc_ptr_e)
    
    sc_diff = get_id()
    emit(131, float_t, sc_diff, sc_val_e, phi_max_v)
    sc_exp = get_id()
    emit(12, float_t, sc_exp, glsl_ext, 19, sc_diff) # exp
    emit(62, sc_ptr_e, sc_exp)
    
    new_exp_sum = get_id()
    emit(129, float_t, new_exp_sum, phi_exp_sum, sc_exp)
    new_exp_i = get_id()
    emit(128, uint_t, new_exp_i, phi_exp_i, c_uint_1)
    
    emit(249, label_exp_cont)
    emit(248, label_exp_cont)
    emit(249, label_exp_head)
    instructions[instructions.index(exp_i_next)] = new_exp_i
    instructions[instructions.index(exp_sum_next)] = new_exp_sum
    
    emit(248, label_exp_merge)
    
    inv_sum = get_id()
    emit(136, float_t, inv_sum, c_float_1, phi_exp_sum)
    
    # Normalize exp scores
    label_norm_head = get_id()
    label_norm_body = get_id()
    label_norm_cont = get_id()
    label_norm_merge = get_id()
    
    emit(249, label_norm_head)
    emit(248, label_norm_head)
    phi_norm_i = get_id()
    emit(244, uint_t, phi_norm_i, c_uint_0, label_exp_merge, get_id(), label_norm_cont)
    norm_i_next = instructions[-3]
    
    emit(247, label_norm_merge, label_norm_cont, 0)
    cond_norm = get_id()
    emit(170, bool_t, cond_norm, phi_norm_i, n_active)
    emit(245, cond_norm, label_norm_body, label_norm_merge)
    emit(248, label_norm_body)
    
    sc_ptr_n = get_id()
    emit(65, ptr_workgroup_float_t, sc_ptr_n, var_scores, phi_norm_i)
    sc_val_n = get_id()
    emit(61, float_t, sc_val_n, sc_ptr_n)
    
    sc_normed = get_id()
    emit(133, float_t, sc_normed, sc_val_n, inv_sum)
    emit(62, sc_ptr_n, sc_normed)
    
    new_norm_i = get_id()
    emit(128, uint_t, new_norm_i, phi_norm_i, c_uint_1)
    
    emit(249, label_norm_cont)
    emit(248, label_norm_cont)
    emit(249, label_norm_head)
    instructions[instructions.index(norm_i_next)] = new_norm_i
    
    emit(248, label_norm_merge)
    emit(249, lbl_after_softmax)
    emit(248, lbl_after_softmax)
    
    emit(224, c_uint_barrier_scope, c_uint_barrier_scope, c_uint_barrier_semantics)
    
    # 3. Weighted Value Accumulation across N_active slots
    label_v_head = get_id()
    label_v_body = get_id()
    label_v_cont = get_id()
    label_v_merge = get_id()
    
    emit(249, label_v_head)
    emit(248, label_v_head)
    
    phi_v_s = get_id()
    phi_v_acc = [get_id() for _ in range(8)]
    emit(244, uint_t, phi_v_s, c_uint_0, lbl_after_softmax, get_id(), label_v_cont)
    vs_next_id = instructions[-3]
    
    v_acc_next = []
    for j in range(8):
        emit(244, float_t, phi_v_acc[j], c_float_0, lbl_after_softmax, get_id(), label_v_cont)
        v_acc_next.append(instructions[-3])
        
    emit(247, label_v_merge, label_v_cont, 0)
    cond_v_loop = get_id()
    emit(170, bool_t, cond_v_loop, phi_v_s, n_active)
    emit(245, cond_v_loop, label_v_body, label_v_merge)
    emit(248, label_v_body)
    
    v_slot_ptr = get_id()
    emit(65, ptr_sb_uint_t, v_slot_ptr, var_active_slots, c_uint_0, phi_v_s)
    v_slot_idx = get_id()
    emit(61, uint_t, v_slot_idx, v_slot_ptr)
    
    v_score_ptr = get_id()
    emit(65, ptr_workgroup_float_t, v_score_ptr, var_scores, phi_v_s)
    v_weight = get_id()
    emit(61, float_t, v_weight, v_score_ptr)
    
    slot_v_base = get_id()
    emit(132, uint_t, slot_v_base, v_slot_idx, kv_stride)
    v_elem_base = get_id()
    emit(128, uint_t, v_elem_base, slot_v_base, kv_lane_off)
    
    new_v_accs = []
    for j in range(8):
        c_j = get_id()
        emit(43, uint_t, c_j, j)
        v_idx_j = get_id()
        emit(128, uint_t, v_idx_j, v_elem_base, c_j)
        v_ptr = get_id()
        emit(65, ptr_sb_float_t, v_ptr, var_v, c_uint_0, v_idx_j)
        v_val = get_id()
        emit(61, float_t, v_val, v_ptr)
        
        v_prod = get_id()
        emit(133, float_t, v_prod, v_weight, v_val)
        new_acc = get_id()
        emit(129, float_t, new_acc, phi_v_acc[j], v_prod)
        new_v_accs.append(new_acc)
        
    new_vs_i = get_id()
    emit(128, uint_t, new_vs_i, phi_v_s, c_uint_1)
    
    emit(249, label_v_cont)
    emit(248, label_v_cont)
    emit(249, label_v_head)
    
    instructions[instructions.index(vs_next_id)] = new_vs_i
    for j in range(8):
        instructions[instructions.index(v_acc_next[j])] = new_v_accs[j]
        
    emit(248, label_v_merge)
    
    # Write 8 output values to buf_out
    for j in range(8):
        c_j = get_id()
        emit(43, uint_t, c_j, j)
        out_idx = get_id()
        emit(128, uint_t, out_idx, q_head_base, lane_elem_off)
        out_idx_j = get_id()
        emit(128, uint_t, out_idx_j, out_idx, c_j)
        out_ptr = get_id()
        emit(65, ptr_sb_float_t, out_ptr, var_out, c_uint_0, out_idx_j)
        emit(62, out_ptr, phi_v_acc[j])
        
    emit(253)
    emit(56)
    
    return make_spirv_header(id_counter, instructions)

if __name__ == '__main__':
    code = generate_decode_attention()
    with open("src/gpu/shaders_attn.zig", "w") as f:
        f.write('pub const DECODE_ATTENTION_SPIRV = [_]u32{\n')
        for i, word in enumerate(code):
            f.write(f' 0x{word:08x},')
            if (i + 1) % 32 == 0:
                f.write('\n')
        if len(code) % 32 != 0:
            f.write('\n')
        f.write('};\n')
    print(f"Generated and wrote Fast Decode Attention SPIR-V: {len(code)} words ({len(code)*4} bytes)")
