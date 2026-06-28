import os

def f10(val, is_int=False, is_str=False):
    if is_int:
        return f"{int(val):>10}"
    elif is_str:
        return f"{str(val):>10}"
    else:
        if val == 0.0:
            return "        0."
        return f"{float(val):>10.3E}"

def create_radioss_deck(run_dir, run_name, test_type, target_strain, strain_rate, prf_data, end_time):
    os.makedirs(run_dir, exist_ok=True)
    starter_file = os.path.join(run_dir, f"{run_name}_0000.rad")
    engine_file = os.path.join(run_dir, f"{run_name}_0001.rad")
    
    lines = []
    lines.append("#RADIOSS STARTER")
    lines.append("/BEGIN")
    lines.append(f"{run_name}")
    lines.append(f10(2024, is_int=True) + f10(0, is_int=True))
    
    # Unit
    lines.append("/UNIT/1")
    lines.append("unit for mat")
    lines.append(f10("Mg", is_str=True) + f10("mm", is_str=True) + f10("s", is_str=True))
    
    # Node
    lines.append("/NODE")
    # node_ID, X, Y, Z
    nodes = [
        [1, 0., 0., 0.],
        [2, 1., 0., 0.],
        [3, 1., 1., 0.],
        [4, 0., 1., 0.],
        [5, 0., 0., 1.],
        [6, 1., 0., 1.],
        [7, 1., 1., 1.],
        [8, 0., 1., 1.],
    ]
    for n in nodes:
        lines.append(f10(n[0], is_int=True) + f10(n[1]) + f10(n[2]) + f10(n[3]))
        
    # Brick
    lines.append("/BRICK")
    # element_ID, part_ID, n1, n2, n3, n4, n5, n6, n7, n8
    lines.append(f10(1, is_int=True) + f10(1, is_int=True) + 
                 f10(1, is_int=True) + f10(2, is_int=True) + f10(3, is_int=True) + f10(4, is_int=True) +
                 f10(5, is_int=True) + f10(6, is_int=True) + f10(7, is_int=True) + f10(8, is_int=True))
                 
    # Part
    lines.append("/PART/1")
    lines.append("1-Element_Block")
    lines.append(f10(1, is_int=True) + f10(1, is_int=True))
    
    # Prop
    lines.append("/PROP/SOLID/1")
    lines.append("Solid_Property")
    lines.append(f10(1, is_int=True) + f10(0, is_int=True))
    
    # Grnod
    groups = {
        1: [1, 2, 3, 4], # Bottom
        2: [5, 6, 7, 8], # Top
        3: [1, 4, 5, 8], # X_face (left)
        4: [1, 2, 5, 6], # Y_face (front)
    }
    for gid, nodes_list in groups.items():
        lines.append(f"/GRNOD/NODE/{gid}")
        lines.append(f"Group_{gid}")
        line_nodes = "".join([f10(n, is_int=True) for n in nodes_list])
        lines.append(line_nodes)
        
    # BCS
    lines.append("/BCS/1")
    lines.append("Bottom_Face_Fixed_Z")
    # Tra rot skew_ID grnod_ID
    lines.append("       001       000" + f10(0, is_int=True) + f10(1, is_int=True))
    
    lines.append("/BCS/2")
    lines.append("Left_Face_Fixed_X")
    lines.append("       100       000" + f10(0, is_int=True) + f10(3, is_int=True))
    
    lines.append("/BCS/3")
    lines.append("Front_Face_Fixed_Y")
    lines.append("       010       000" + f10(0, is_int=True) + f10(4, is_int=True))
    
    # IMPVEL
    lines.append("/IMPVEL/1")
    lines.append("Top_Face_Velocity")
    # funct_IDT Dir skew_ID sensor_ID grnod_ID
    lines.append(f10(1, is_int=True) + f10("Z", is_str=True) + f10(0, is_int=True) + f10(0, is_int=True) + f10(2, is_int=True))
    
    # FUNCT
    lines.append("/FUNCT/1")
    lines.append("Velocity_Curve")
    if test_type == 'tension':
        v = strain_rate
        lines.append(f10(0.0) + f10(v))
        lines.append(f10(end_time) + f10(v))
    else: # relaxation
        ramp_time = 0.1
        v = target_strain / ramp_time
        lines.append(f10(0.0) + f10(v))
        lines.append(f10(ramp_time) + f10(v))
        lines.append(f10(ramp_time+1e-5) + f10(0.0))
        lines.append(f10(end_time) + f10(0.0))
        
    # Material
    lines.append("/MAT/LAW100/1/1")
    lines.append("PRF_MNF_Equivalent")
    lines.append(" 1.0000000000000E-09")
    lines.append(f10(3, is_int=True) + f10(1, is_int=True))
    C10 = prf_data["hyperelastic_params"][0]
    lines.append(f10(C10) + f10(0.0) + f10(0.0) + f10(0.0) + f10(0.0))
    lines.append(f10(0.0) + f10(0.0) + f10(0.0) + f10(0.0))
    lines.append(f10(0.0))
    
    networks = prf_data["networks"]
    for i, net in enumerate(networks):
        s = net["stiffness_ratio"]
        A = net["creep_params"][0]
        n = net["creep_params"][1]
        m = net["creep_params"][2]
        lines.append(f"NETWORK{i+1:d}" + f10(3, is_int=True) + f"{s:>10.5f}")
        lines.append(f10(A) + f10(n) + f10(m))

    lines.append("/END")
    
    starter_content = "\n".join(lines) + "\n"
    with open(starter_file, 'w', encoding='utf-8') as f:
        f.write(starter_content)

    # 2. ENGINE FILE
    engine_content = f"""#RADIOSS ENGINE
/RUN/{run_name}/1
{f10(end_time)}
/TFILE
{f10(end_time/1000.0)}
/ANIM/DT
{f10(end_time/20.0)}
/PRINT/-1000
/END
"""
    with open(engine_file, 'w', encoding='utf-8') as f:
        f.write(engine_content)
        
    return starter_file, engine_file
