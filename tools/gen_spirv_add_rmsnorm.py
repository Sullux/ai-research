#!/usr/bin/env python3
import struct

def make_spirv_header(id_bound, instructions):
    return [0x07230203, 0x00010300, 0x00000000, id_bound, 0x00000000] + instructions

def generate_add_rmsnorm():
    """
    Fused Add + RMSNorm Compute Shader:
    x[i] += delta[i]
    normed_x[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
    
    Bindings:
      0: x        (StorageBuffer, in/out)
      1: delta    (StorageBuffer, in)
      2: weight   (StorageBuffer, in)
      3: normed_x (StorageBuffer, out)
    Push constants: { uint H, float eps }
    LocalSize: (64, 1, 1)
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
    var_lid = get_id()
    
    arr_float_t = get_id()
    struct_buf_t = get_id()
    ptr_sb_struct_buf_t = get_id()
    var_x = get_id()
    var_delta = get_id()
    var_weight = get_id()
    var_normed_x = get_id()
    
    struct_pc_t = get_id()
    ptr_pc_struct_t = get_id()
    var_pc = get_id()
    
    c_uint_64 = get_id()
    arr_float_64_t = get_id()
    ptr_workgroup_arr_float_64_t = get_id()
    var_sdata = get_id()
    ptr_workgroup_float_t = get_id()
    
    ptr_sb_float_t = get_id()
    ptr_pc_uint_t = get_id()
    ptr_pc_float_t = get_id()
    
    c_uint_0 = get_id()
    c_uint_1 = get_id()
    c_uint_2 = get_id()
    c_uint_3 = get_id()
    c_uint_4 = get_id()
    c_uint_8 = get_id()
    c_uint_16 = get_id()
    c_uint_32 = get_id()
    
    c_float_0 = get_id()
    c_float_1 = get_id()
    
    c_uint_barrier_scope = get_id()
    c_uint_barrier_semantics = get_id()
    
    glsl_ext = get_id()
    main_func = get_id()
    
    emit(17, 1)
    emit(11, glsl_ext, "GLSL.std.450")
    emit(14, 0, 1)
    emit(15, 5, main_func, "main", var_lid)
    emit(16, main_func, 1, 64, 1, 1)
    
    emit(71, struct_buf_t, 2)
    emit(72, struct_buf_t, 0, 35, 0)
    emit(71, arr_float_t, 28, 4)
    emit(71, var_x, 33, 0)
    emit(71, var_x, 34, 0)
    
    emit(71, var_delta, 33, 0)
    emit(71, var_delta, 34, 1)
    
    emit(71, var_weight, 33, 0)
    emit(71, var_weight, 34, 2)
    
    emit(71, var_normed_x, 33, 0)
    emit(71, var_normed_x, 34, 3)
    
    emit(71, struct_pc_t, 3)
    emit(72, struct_pc_t, 0, 35, 0)
    emit(72, struct_pc_t, 1, 35, 4)
    
    emit(71, var_lid, 11, 28)
    
    emit(19, void_t)
    emit(33, func_t, void_t)
    emit(21, float_t, 32)
    emit(21, uint_t, 32, 0)
    emit(21, int_t, 32, 1)
    emit(20, bool_t)
    emit(23, uvec3_t, uint_t, 3)
    
    emit(32, ptr_input_uvec3_t, 1, uvec3_t)
    emit(28, arr_float_t, float_t)
    emit(30, struct_buf_t, arr_float_t)
    emit(32, ptr_sb_struct_buf_t, 12, struct_buf_t)
    
    emit(30, struct_pc_t, uint_t, float_t)
    emit(32, ptr_pc_struct_t, 9, struct_pc_t)
    
    emit(43, uint_t, c_uint_64, 64)
    emit(29, arr_float_64_t, float_t, c_uint_64)
    emit(32, ptr_workgroup_arr_float_64_t, 4, arr_float_64_t)
    emit(32, ptr_workgroup_float_t, 4, float_t)
    
    emit(32, ptr_sb_float_t, 12, float_t)
    emit(32, ptr_pc_uint_t, 9, uint_t)
    emit(32, ptr_pc_float_t, 9, float_t)
    
    emit(43, uint_t, c_uint_0, 0)
    emit(43, uint_t, c_uint_1, 1)
    emit(43, uint_t, c_uint_2, 2)
    emit(43, uint_t, c_uint_3, 3)
    emit(43, uint_t, c_uint_4, 4)
    emit(43, uint_t, c_uint_8, 8)
    emit(43, uint_t, c_uint_16, 16)
    emit(43, uint_t, c_uint_32, 32)
    emit(43, float_t, c_float_0, 0.0)
    emit(43, float_t, c_float_1, 1.0)
    emit(43, uint_t, c_uint_barrier_scope, 2)
    emit(43, uint_t, c_uint_barrier_semantics, 0x108)
    
    emit(59, ptr_input_uvec3_t, var_lid, 1)
    emit(59, ptr_sb_struct_buf_t, var_x, 12)
    emit(59, ptr_sb_struct_buf_t, var_delta, 12)
    emit(59, ptr_sb_struct_buf_t, var_weight, 12)
    emit(59, ptr_sb_struct_buf_t, var_normed_x, 12)
    emit(59, ptr_pc_struct_t, var_pc, 9)
    emit(59, ptr_workgroup_arr_float_64_t, var_sdata, 4)
    
    emit(54, void_t, main_func, 0, func_t)
    label_entry = get_id()
    emit(248, label_entry)
    
    lid_x_ptr = get_id()
    emit(65, ptr_input_uvec3_t, lid_x_ptr, var_lid, c_uint_0)
    tid = get_id()
    emit(61, uint_t, tid, lid_x_ptr)
    
    h_ptr = get_id()
    emit(65, ptr_pc_uint_t, h_ptr, var_pc, c_uint_0)
    val_h = get_id()
    emit(61, uint_t, val_h, h_ptr)
    
    eps_ptr = get_id()
    emit(65, ptr_pc_float_t, eps_ptr, var_pc, c_uint_1)
    val_eps = get_id()
    emit(61, float_t, val_eps, eps_ptr)
    
    # Pass 1: x[i] += delta[i], accumulate sum_sq
    label_p1_head = get_id()
    label_p1_body = get_id()
    label_p1_cont = get_id()
    label_p1_merge = get_id()
    
    emit(249, label_p1_head)
    emit(248, label_p1_head)
    
    phi_p1_i = get_id()
    phi_p1_sum = get_id()
    emit(244, uint_t, phi_p1_i, tid, label_entry, get_id(), label_p1_cont)
    p1_i_next = instructions[-3]
    emit(244, float_t, phi_p1_sum, c_float_0, label_entry, get_id(), label_p1_cont)
    p1_sum_next = instructions[-3]
    
    emit(247, label_p1_merge, label_p1_cont, 0)
    cond_p1 = get_id()
    emit(170, bool_t, cond_p1, phi_p1_i, val_h)
    emit(245, cond_p1, label_p1_body, label_p1_merge)
    
    emit(248, label_p1_body)
    x_ptr = get_id()
    emit(65, ptr_sb_float_t, x_ptr, var_x, c_uint_0, phi_p1_i)
    old_x = get_id()
    emit(61, float_t, old_x, x_ptr)
    
    delta_ptr = get_id()
    emit(65, ptr_sb_float_t, delta_ptr, var_delta, c_uint_0, phi_p1_i)
    delta_val = get_id()
    emit(61, float_t, delta_val, delta_ptr)
    
    new_x = get_id()
    emit(129, float_t, new_x, old_x, delta_val)
    emit(62, x_ptr, new_x) # Write back to x
    
    x_sq = get_id()
    emit(133, float_t, x_sq, new_x, new_x)
    
    new_p1_sum = get_id()
    emit(129, float_t, new_p1_sum, phi_p1_sum, x_sq)
    
    new_p1_i = get_id()
    emit(128, uint_t, new_p1_i, phi_p1_i, c_uint_64)
    
    emit(249, label_p1_cont)
    emit(248, label_p1_cont)
    emit(249, label_p1_head)
    
    instructions[instructions.index(p1_i_next)] = new_p1_i
    instructions[instructions.index(p1_sum_next)] = new_p1_sum
    
    emit(248, label_p1_merge)
    
    # Store to shared memory sdata[tid] = phi_p1_sum
    sdata_ptr = get_id()
    emit(65, ptr_workgroup_float_t, sdata_ptr, var_sdata, tid)
    emit(62, sdata_ptr, phi_p1_sum)
    emit(224, c_uint_barrier_scope, c_uint_barrier_scope, c_uint_barrier_semantics)
    
    # 6-step reduction across 64 threads
    for stride in [32, 16, 8, 4, 2, 1]:
        c_s = get_id()
        emit(43, uint_t, c_s, stride)
        cond_s = get_id()
        emit(170, bool_t, cond_s, tid, c_s)
        
        lbl_r_then = get_id()
        lbl_r_merge = get_id()
        emit(245, cond_s, lbl_r_then, lbl_r_merge)
        
        emit(248, lbl_r_then)
        other_tid = get_id()
        emit(128, uint_t, other_tid, tid, c_s)
        o_ptr = get_id()
        emit(65, ptr_workgroup_float_t, o_ptr, var_sdata, other_tid)
        o_val = get_id()
        emit(61, float_t, o_val, o_ptr)
        
        my_p = get_id()
        emit(65, ptr_workgroup_float_t, my_p, var_sdata, tid)
        my_v = get_id()
        emit(61, float_t, my_v, my_p)
        
        red = get_id()
        emit(129, float_t, red, my_v, o_val)
        emit(62, my_p, red)
        emit(249, lbl_r_merge)
        
        emit(248, lbl_r_merge)
        emit(224, c_uint_barrier_scope, c_uint_barrier_scope, c_uint_barrier_semantics)
        
    # Read total sum_sq from sdata[0]
    s0_ptr = get_id()
    emit(65, ptr_workgroup_float_t, s0_ptr, var_sdata, c_uint_0)
    tot_sum_sq = get_id()
    emit(61, float_t, tot_sum_sq, s0_ptr)
    
    h_float = get_id()
    emit(111, float_t, h_float, val_h)
    mean_sq = get_id()
    emit(136, float_t, mean_sq, tot_sum_sq, h_float)
    var_plus_eps = get_id()
    emit(129, float_t, var_plus_eps, mean_sq, val_eps)
    rsqrt_val = get_id()
    emit(12, float_t, rsqrt_val, glsl_ext, 32, var_plus_eps) # OpExtInst InverseSqrt
    
    # Pass 2: normed_x[i] = x[i] * rsqrt * weight[i]
    label_p2_head = get_id()
    label_p2_body = get_id()
    label_p2_cont = get_id()
    label_p2_merge = get_id()
    
    emit(249, label_p2_head)
    emit(248, label_p2_head)
    
    phi_p2_i = get_id()
    emit(244, uint_t, phi_p2_i, tid, label_p1_merge, get_id(), label_p2_cont)
    p2_i_next = instructions[-3]
    
    emit(247, label_p2_merge, label_p2_cont, 0)
    cond_p2 = get_id()
    emit(170, bool_t, cond_p2, phi_p2_i, val_h)
    emit(245, cond_p2, label_p2_body, label_p2_merge)
    
    emit(248, label_p2_body)
    p2_x_ptr = get_id()
    emit(65, ptr_sb_float_t, p2_x_ptr, var_x, c_uint_0, phi_p2_i)
    cur_x = get_id()
    emit(61, float_t, cur_x, p2_x_ptr)
    
    w_ptr = get_id()
    emit(65, ptr_sb_float_t, w_ptr, var_weight, c_uint_0, phi_p2_i)
    w_val = get_id()
    emit(61, float_t, w_val, w_ptr)
    
    scaled_x = get_id()
    emit(133, float_t, scaled_x, cur_x, rsqrt_val)
    normed_val = get_id()
    emit(133, float_t, normed_val, scaled_x, w_val)
    
    normed_ptr = get_id()
    emit(65, ptr_sb_float_t, normed_ptr, var_normed_x, c_uint_0, phi_p2_i)
    emit(62, normed_ptr, normed_val)
    
    new_p2_i = get_id()
    emit(128, uint_t, new_p2_i, phi_p2_i, c_uint_64)
    
    emit(249, label_p2_cont)
    emit(248, label_p2_cont)
    emit(249, label_p2_head)
    
    instructions[instructions.index(p2_i_next)] = new_p2_i
    
    emit(248, label_p2_merge)
    emit(253)
    emit(56)
    
    return make_spirv_header(id_counter, instructions)

if __name__ == '__main__':
    code = generate_add_rmsnorm()
    with open("src/gpu/shaders_add_rmsnorm.zig", "w") as f:
        f.write('pub const FUSED_ADD_RMSNORM_SPIRV = [_]u32{\n')
        for i, word in enumerate(code):
            f.write(f' 0x{word:08x},')
            if (i + 1) % 16 == 0:
                f.write('\n')
        if len(code) % 16 != 0:
            f.write('\n')
        f.write('};\n')
    print(f"Generated fused add+rmsnorm SPIR-V: {len(code)} words ({len(code)*4} bytes)")
