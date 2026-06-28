import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import matplotlib

# Use koreanize_matplotlib if available
try:
    import koreanize_matplotlib
except ImportError:
    pass

from wht_prf.calibration.step10_final_validation import load_data, simulate_test

# Abaqus PRF Material Parameters provided by the user
ABAQUS_MATERIAL = """*Material, name=PRF_model
*Hyperelastic,  NeoHooke,  Moduli=instantaneous 
549.6, 0.0 
*Viscoelastic, Nonlinear, NetworkId=1, SRatio=0.337060, Law=strain 
4.8828e-7, 3.13398, -0.551660 
*Viscoelastic, Nonlinear, NetworkId=2, SRatio=0.150654, Law=strain 
2.4414e-6, 4.91992, -0.746191 
*Viscoelastic, Nonlinear, NetworkId=3, SRatio=0.372607, Law=strain 
7.81403e-5, 3.801953, -0.616699"""

def generate_inp(job_name, times, strains):
    # strains are True Strains. Displacement u = exp(strain) - 1
    # We will create an amplitude curve
    amp_lines = ["*AMPLITUDE, NAME=AMP_STRAIN"]
    
    pairs = []
    # Abaqus amplitude time must start from 0 strictly.
    for t, e in zip(times, strains):
        u = np.exp(e) - 1.0  # Engineering strain = Displacement for L0=1
        pairs.extend([t, u])
    
    # Chunk into 8 items per line
    for i in range(0, len(pairs), 8):
        line = ", ".join(f"{x:.6e}" for x in pairs[i:i+8])
        amp_lines.append(line)
        
    amp_block = "\n".join(amp_lines)
    max_time = times.iloc[-1] if isinstance(times, pd.Series) else times[-1]
    
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
*SOLID SECTION, ELSET=EA, MATERIAL=PRF_model
1.
{ABAQUS_MATERIAL}
{amp_block}
*STEP, INC=100000, NLGEOM=YES
*VISCO
{max_time/1000.0}, {max_time}, 1e-15, {max_time/20.0}
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
*NODE PRINT, FREQUENCY=0
*EL PRINT, FREQUENCY=1
S11, LE11
*END STEP
"""
    with open(f"{job_name}.inp", "w") as f:
        f.write(inp_content)

def run_abaqus(job_name):
    print(f"Running Abaqus job {job_name}...")
    # remove old dat
    if os.path.exists(f"{job_name}.dat"):
        os.remove(f"{job_name}.dat")
    subprocess.run(f"abaqus job={job_name} interactive ask_delete=OFF", shell=True, check=True)

def parse_dat(job_name):
    dat_file = f"{job_name}.dat"
    time_list = []
    s11_list = []
    le11_list = []
    
    current_time = 0.0
    if not os.path.exists(dat_file):
        return None, None, None
        
    with open(dat_file, "r") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if "STEP TIME COMPLETED" in line:
            parts = line.split()
            try:
                current_time = float(parts[-1])
            except:
                pass
        
        # Look for the data line: usually starts with "    1   1 " followed by S11 and LE11
        if "    1   1 " in line and len(line.split()) >= 4:
            parts = line.split()
            try:
                s11 = float(parts[2])
                le11 = float(parts[3])
                time_list.append(current_time)
                s11_list.append(s11)
                le11_list.append(le11)
            except:
                pass
                
    return np.array(time_list), np.array(s11_list), np.array(le11_list)

def extract_odb_python(job_name):
    py_content = f"""from odbAccess import openOdb
import sys

odb = openOdb('{job_name}.odb')
step = odb.steps.values()[0]
region = step.historyRegions.values()[0]

s11_data = []
le11_data = []

for key in region.historyOutputs.keys():
    if 'S11' in key and 'EL 1' in key:
        s11_data = region.historyOutputs[key].data
    if 'LE11' in key and 'EL 1' in key:
        le11_data = region.historyOutputs[key].data

with open('{job_name}_extracted.txt', 'w') as f:
    for s, e in zip(s11_data, le11_data):
        f.write(str(s[0]) + ',' + str(s[1]) + ',' + str(e[1]) + '\\n')
odb.close()
"""
    with open("extract.py", "w") as f:
        f.write(py_content)
        
    subprocess.run("abaqus python extract.py", shell=True, check=True)
    
    df = pd.read_csv(f"{job_name}_extracted.txt", names=['time', 's11', 'le11'])
    return df['time'].values, df['s11'].values, df['le11'].values

def main():
    print("Loading experimental data...")
    targets = load_data()
    
    params = {
        'A': jnp.array([4.8828e-7, 2.4414e-6, 7.81403e-5]),
        'n': jnp.array([3.13398, 4.91992, 3.801953]),
        'm': jnp.array([-0.551660, -0.746191, -0.616699]),
        's': jnp.array([0.337060, 0.150654, 0.372607]),
        'C10': 549.6
    }
    
    results = {}
    
    for key in ['rate_100', 'relax_0050', 'relax_0075', 'relax_0100', 'relax_0150']:
        data = targets[key]
        job_name = f"step11_{key}"
        
        # generate INP
        generate_inp(job_name, data['time'], data['strain'])
        
        try:
            run_abaqus(job_name)
            abq_time, abq_s11, abq_le11 = extract_odb_python(job_name)
        except Exception as e:
            print(f"Failed to run Abaqus for {key}: {e}")
            continue
            
        dt_arr = jnp.diff(data['time'])
        # Also wait, E_inst = 6 * C10 as per user comment in step10_final_validation.py?
        # Actually in 3D PRF, E_inst = 6 * C10. Let's use 6 * C10 = 3297.6
        sim_stress = simulate_test(params, dt_arr, data['strain'], E_inst=params['C10'] * 6.0)
        sim_time = data['time']
        
        results[key] = {
            'abq_time': abq_time,
            'abq_s11': abq_s11,
            'abq_le11': abq_le11,
            'prf_time': sim_time,
            'prf_stress': sim_stress
        }
        
    plt.figure(figsize=(10, 6), dpi=150)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, key in enumerate(['rate_100', 'relax_0050', 'relax_0075', 'relax_0100', 'relax_0150']):
        if key not in results: continue
        data = targets[key]
        res = results[key]
        
        c = colors[i % len(colors)]
        plt.plot(data['time'], data['stress'], 'o', markersize=4, color=c, alpha=0.5, label=f'Exp ({key})')
        plt.plot(res['prf_time'], res['prf_stress'], '--', color=c, linewidth=2, label=f'WHT_PRF ({key})')
        plt.plot(res['abq_time'], res['abq_s11'], '-', color=c, linewidth=1.5, label=f'Abaqus ({key})')
        
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('True Stress (MPa)')
    plt.title('Step 11: Final Validation - Time vs Stress (Fig 20)')
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('dev_log/resources/step11_fig20_time.png')
    
    plt.figure(figsize=(10, 6), dpi=150)
    for i, key in enumerate(['rate_100', 'relax_0050', 'relax_0075', 'relax_0100', 'relax_0150']):
        if key not in results: continue
        data = targets[key]
        res = results[key]
        
        c = colors[i % len(colors)]
        plt.plot(data['strain'], data['stress'], 'o', markersize=4, color=c, alpha=0.5, label=f'Exp ({key})')
        plt.plot(data['strain'], res['prf_stress'], '--', color=c, linewidth=2, label=f'WHT_PRF ({key})')
        plt.plot(res['abq_le11'], res['abq_s11'], '-', color=c, linewidth=1.5, label=f'Abaqus ({key})')
        
    plt.xlabel('True Strain')
    plt.ylabel('True Stress (MPa)')
    plt.title('Step 11: Final Validation - Strain vs Stress (Fig 21)')
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('dev_log/resources/step11_fig21_strain.png')
    
    print("Done! Saved Step 11 figures.")

if __name__ == "__main__":
    main()
