import os
import shutil
import subprocess

def f10(val, is_int=False, is_str=False):
    if is_int:
        return f"{int(val):>10}"
    elif is_str:
        return f"{str(val):>10}"
    else:
        if val == 0.0:
            return "        0."
        return f"{float(val):>10.3E}"

def build_and_run_cylinder(run_dir, prf_data):
    base_dir = r"D:\PythonCodeStudy\WHT_PRF\doc\openradioss_exam\rd-e-5200_creep\cylinder_relaxation_beta_1"
    
    os.makedirs(run_dir, exist_ok=True)
    
    # 1. Read base 0000.rad
    base_rad0 = os.path.join(base_dir, "foam_relax_0000.rad")
    with open(base_rad0, 'r') as f:
        content = f.read()
        
    # Update /BEGIN version to 2024
    content = content.replace("       140         0", "      2024         0")
    
    # Find the MAT block and replace it
    start_idx = content.find("/MAT/KELVINMAX/3")
    end_idx = content.find("/NODE", start_idx)
    
    C10 = prf_data["hyperelastic_params"][0]
    
    mat_block = f"""/MAT/LAW100/3
PRF_MNF_Equivalent
#              RHO_I        
 2.0000000000000E-09
#N_NETWORK   FLAG_HE   FLAG_Cr
         3         1          
#                C10                 C01                 C20                 C11                 C02
{f10(C10)}{f10(0.0)}{f10(0.0)}{f10(0.0)}{f10(0.0)}
#                C30                 C21                 C12                 C03 
{f10(0.0)}{f10(0.0)}{f10(0.0)}{f10(0.0)}
#                 D1                  D2                  D3     
{f10(0.0)}
#---1----|----2----|----3----|----4----|----5----|----6----|----7----|----8----|----9----|---10----|
"""
    networks = prf_data["networks"]
    for i, net in enumerate(networks):
        s = net["stiffness_ratio"]
        A = net["creep_params"][0]
        n = net["creep_params"][1]
        m = net["creep_params"][2]
        mat_block += f"""#   KEYNET FLAG_VISC          SCALESTIFF 
NETWORK{i+1}           3          {s:10.5f}
#                 A3                EXPN                EXPM       
{f10(A)}{f10(n)}{f10(m)}             
"""
    mat_block += "#---1----|----2----|----3----|----4----|----5----|----6----|----7----|----8----|----9----|---10----|\n"
    mat_block += "#-  3. NODES:\n"
    mat_block += "#---1----|----2----|----3----|----4----|----5----|----6----|----7----|----8----|----9----|---10----|\n"
    
    new_content = content[:start_idx] + mat_block + content[end_idx:]
    
    out_rad0 = os.path.join(run_dir, "foam_relax_0000.rad")
    with open(out_rad0, 'w') as f:
        f.write(new_content)
        
    # 2. Copy 0001.rad
    base_rad1 = os.path.join(base_dir, "foam_relax_0001.rad")
    target_rad1 = os.path.join(run_dir, "foam_relax_0001.rad")
    if os.path.exists(target_rad1):
        try:
            os.chmod(target_rad1, 0o777)
            os.remove(target_rad1)
        except:
            pass
    shutil.copyfile(base_rad1, target_rad1)
    
    # 3. Run OpenRadioss
    env = os.environ.copy()
    radioss_dir = r"D:\OpenRadioss_win64\OpenRadioss"
    env["RAD_CFG_PATH"] = os.path.join(radioss_dir, "hm_cfg_files")
    extlib_dir = os.path.join(radioss_dir, "extlib", "hm_reader", "win64")
    env["PATH"] = extlib_dir + os.pathsep + env.get("PATH", "")
    env["KMP_STACKSIZE"] = "400m"
    
    print("Running Starter...")
    starter_exe = os.path.join(radioss_dir, "exec", "starter_win64.exe")
    result = subprocess.run([starter_exe, "-i", "foam_relax_0000.rad"], cwd=run_dir, env=env, capture_output=True, text=True)
    if result.returncode != 0 or "ERROR TERMINATION" in result.stdout:
        print("Starter failed!")
        print(result.stdout)
        return False
        
    print("Running Engine...")
    engine_exe = os.path.join(radioss_dir, "exec", "engine_win64.exe")
    result = subprocess.run([engine_exe, "-i", "foam_relax_0001.rad"], cwd=run_dir, env=env, capture_output=True, text=True)
    if result.returncode != 0 or "ERROR TERMINATION" in result.stdout:
        print("Engine failed!")
        print(result.stdout)
        return False
        
    print("Radioss run successful.")
    return True

if __name__ == "__main__":
    prf_data = {
        "hyperelastic_params": [0.2019],
        "networks": [
            {"stiffness_ratio": 0.6, "creep_params": [1.0, 5.0, 2.0]},
            {"stiffness_ratio": 0.1, "creep_params": [1.0, 5.0, 2.0]},
            {"stiffness_ratio": 0.3, "creep_params": [1.0, 5.0, 2.0]}
        ]
    }
    
    run_dir = os.path.join("dev_log", "radioss_runs", "cylinder_relax_law100")
    build_and_run_cylinder(run_dir, prf_data)
