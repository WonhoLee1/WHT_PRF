# WHTOOLs MATCALIB 2026
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp

try:
    import koreanize_matplotlib
except ImportError:
    pass

from wht_prf.io_manager import parse_abaqus_prf
from wht_prf.visco_prony import simulate_prony_test

# Abaqus PRONY Material Parameters
ABAQUS_MATERIAL = """*Material, name=Prony_model
*Hyperelastic,  NeoHooke,  Moduli=instantaneous 
549.6, 1e-4 
*Viscoelastic, time=PRONY
0.2, 0.0, 0.5
0.1, 0.0, 2.0
0.1, 0.0, 10.0"""

def generate_inp(job_name, times, strains):
    amp_lines = ["*AMPLITUDE, NAME=AMP_STRAIN"]
    pairs = []
    for t, e in zip(times, strains):
        u = e  # Apply engineering strain directly as displacement since X=1
        pairs.extend([t, u])
    
    for i in range(0, len(pairs), 8):
        line = ", ".join(f"{x:.6e}" for x in pairs[i:i+8])
        amp_lines.append(line)
        
    amp_block = "\n".join(amp_lines)
    max_time = times[-1]
    
    inp_content = f"""*HEADING
Single element test for {job_name}
*NODE
1, 0., 0., 0.
2, 1., 0., 0.
3, 1., 1., 0.
4, 0., 1., 0.
5, 0., 0., 1.
6, 1., 0., 1.
7, 1., 1., 1.
8, 0., 1., 1.
*ELEMENT, TYPE=C3D8H, ELSET=EA
1, 1, 2, 3, 4, 5, 6, 7, 8
*SOLID SECTION, ELSET=EA, MATERIAL=Prony_model
1.
{ABAQUS_MATERIAL}
{amp_block}
*STEP, INC=100000, NLGEOM=YES
*VISCO, CETOL=0.05
{max_time/1000.0}, {max_time}, 1e-8, {max_time/10.0}
*BOUNDARY
1, 1, 3
4, 1, 1
4, 3, 3
8, 1, 1
5, 1, 2
2, 2, 3
6, 2, 2
2, 3, 3
3, 3, 3
*BOUNDARY, AMPLITUDE=AMP_STRAIN
2, 1, 1, 1.0
3, 1, 1, 1.0
6, 1, 1, 1.0
7, 1, 1, 1.0
*OUTPUT, FIELD, FREQUENCY=1
*ELEMENT OUTPUT
S, LE
*OUTPUT, HISTORY, FREQUENCY=1
*ELEMENT OUTPUT, ELSET=EA
S11, LE11
*END STEP
"""
    with open(f"{job_name}.inp", "w") as f:
        f.write(inp_content)

def run_abaqus(job_name):
    print(f"Running Abaqus job {job_name}...")
    if os.path.exists(f"{job_name}.dat"):
        os.remove(f"{job_name}.dat")
    subprocess.run(f"abaqus job={job_name} interactive ask_delete=OFF", shell=True, check=True)

def extract_odb_python(job_name):
    py_content = f"""from odbAccess import openOdb
import sys

try:
    odb = openOdb('{job_name}.odb')
    step = odb.steps.values()[0]

    s11_data = []
    le11_data = []

    for r_name, region in step.historyRegions.items():
        if 'Int Point 1' in r_name:
            if 'S11' in region.historyOutputs:
                s11_data = region.historyOutputs['S11'].data
            if 'LE11' in region.historyOutputs:
                le11_data = region.historyOutputs['LE11'].data
            break

    with open('{job_name}_extracted.txt', 'w') as f:
        for s, e in zip(s11_data, le11_data):
            f.write(str(s[0]) + ',' + str(s[1]) + ',' + str(e[1]) + '\\n')
    odb.close()
except Exception as e:
    with open('{job_name}_error.txt', 'w') as f:
        f.write(str(e))
"""
    with open("extract.py", "w") as f:
        f.write(py_content)
        
    subprocess.run("abaqus python extract.py", shell=True, check=True)
    
    if os.path.exists(f"{job_name}_extracted.txt"):
        df = pd.read_csv(f"{job_name}_extracted.txt", names=['time', 's11', 'le11'])
        return df['time'].values, df['s11'].values, df['le11'].values
    else:
        raise RuntimeError("ODB Extraction failed.")

def main():
    print("Testing Prony Series Validation...")
    
    # 임의의 램프 변형 + 이완 이력 생성
    # 0 ~ 1초까지 50% 변형률, 이후 10초까지 유지
    times = np.concatenate([np.linspace(0.0, 1.0, 10), np.linspace(1.1, 10.0, 40)])
    strains = np.concatenate([np.linspace(0.0, 0.5, 10), np.ones(40)*0.5])
    
    job_name = "test_prony_val"
    generate_inp(job_name, times, strains)
    
    run_abaqus(job_name)
    abq_time, abq_s11, abq_le11 = extract_odb_python(job_name)
    
    # WHT_PRF JAX 실행
    mat_data = parse_abaqus_prf(ABAQUS_MATERIAL)
    jax_times = jnp.array(times)
    jax_strains = jnp.array(strains)
    
    sim_stress = simulate_prony_test(mat_data, jax_times, jax_strains)
    jax_s11 = sim_stress[:, 0, 0]
    
    # Interpolate JAX results to Abaqus times for comparison
    jax_s11_interp = np.interp(abq_time, times, jax_s11)
    error = np.abs(abq_s11 - jax_s11_interp)
    max_err = np.max(error)
    print(f"Max Error between Abaqus and JAX: {max_err:.5e}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(abq_time, abq_s11, 'k-', linewidth=3, label='Abaqus (Prony)')
    plt.plot(times, jax_s11, 'r--', linewidth=2, label='WHT_PRF (JAX)')
    plt.title('Prony Series Stress Relaxation Validation')
    plt.xlabel('Time (s)')
    plt.ylabel('True Stress S11 (MPa)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dev_log/resources/step12_prony_validation.png')
    
    print("Validation Plot saved to dev_log/resources/step12_prony_validation.png")

if __name__ == "__main__":
    main()
