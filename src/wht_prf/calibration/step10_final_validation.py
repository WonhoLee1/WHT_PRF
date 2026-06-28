"""
WHTOOLs MATCALIB 2026
Final validation using fully calibrated parameters from Abaqus input deck.
"""
import os
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    base_dir = r"G:\Simulia_Video_Contents\Material Calibration\Calibration of a PRF Material Model for Polypropylene\PPC3TF2TestData"
    data_files = {
        'rate_100': 'rate_data_1e+2_partial.txt',
        'relax_0050': 'relax_data_0050.txt',
        'relax_0075': 'relax_data_0075.txt',
        'relax_0100': 'relax_data_0100.txt',
        'relax_0150': 'relax_data_0150.txt'
    }
    
    targets = {}
    for key, filename in data_files.items():
        path = os.path.join(base_dir, filename)
        if 'rate_data' in filename:
            df = pd.read_csv(path, sep=r'\s+', header=None)
        else:
            df = pd.read_csv(path, sep=',', header=1)
            
        time = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        strain = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        stress = pd.to_numeric(df.iloc[:, 2], errors='coerce').values
        
        mask = ~np.isnan(time) & ~np.isnan(strain) & ~np.isnan(stress)
        time = time[mask]
        strain = strain[mask]
        stress = stress[mask]
        
        dt = np.diff(time)
        dt = np.where(dt <= 0, 1e-6, dt)
        
        targets[key] = {
            'time': time,
            'strain': strain,
            'stress': stress,
            'dt': dt
        }
    return targets

def prf_1d_step_substep(state, inputs, params, E_inst):
    e_cr_old = state
    dt_total, strain_tot = inputs
    
    A = params['A']
    n = params['n']
    m = params['m']
    s = params['s']
    E_i = s * E_inst
    
    num_substeps = 100
    dt_sub = dt_total / num_substeps
    
    def substep_fn(i, e_cr_curr):
        sig_i = E_i * (strain_tot - e_cr_curr)
        q = jnp.abs(sig_i)
        e_cr_eff = jnp.abs(e_cr_curr) + 1e-8
        
        base = jnp.clip(A * (q**n) * (((1.0 + m) * e_cr_eff)**m), 0.0, 1e10)
        om1 = 1.0 / (1.0 + m)
        e_dot_mag = jnp.clip(base ** om1, 0.0, 1e5)
        
        e_dot = e_dot_mag * jnp.sign(sig_i)
        return e_cr_curr + e_dot * dt_sub

    e_cr_new = jax.lax.fori_loop(0, num_substeps, substep_fn, e_cr_old)
    
    s_inf = 1.0 - jnp.sum(s)
    sig_inf = s_inf * E_inst * strain_tot
    sig_tot = sig_inf + jnp.sum(E_i * (strain_tot - e_cr_new))
    
    return e_cr_new, sig_tot

def simulate_test(params, dt_arr, strain_arr, E_inst):
    init_e_cr = jnp.zeros(3, dtype=jnp.float64)
    inputs = (dt_arr, strain_arr[1:])
    scan_fn = lambda state, x: prf_1d_step_substep(state, x, params, E_inst)
    _, stress_sim = jax.lax.scan(scan_fn, init_e_cr, inputs)
    sig0 = E_inst * strain_arr[0]
    stress_sim = jnp.concatenate([jnp.array([sig0]), stress_sim])
    return stress_sim

def main():
    targets = load_data()
    
    # Parameters provided by user from Abaqus deck
    # C10 = 549.6, E_inst = 6 * 549.6 = 3297.6 MPa
    E_inst = 3297.6
    
    params = {
        'A': jnp.array([4.8828e-7, 2.4414e-6, 7.81403e-5]),
        'n': jnp.array([3.13398, 4.91992, 3.801953]),
        'm': jnp.array([-0.551660, -0.746191, -0.616699]),
        's': jnp.array([0.337060, 0.150654, 0.372607])
    }
    
    print("Simulating tests with final parameters...")
    sim_results = {}
    for key, data in targets.items():
        sim_stress = simulate_test(params, jnp.array(data['dt']), jnp.array(data['strain']), E_inst)
        sim_results[key] = np.array(sim_stress)
        print(f"Simulated {key}")

    # Figure 20: Stress vs Time
    plt.figure(figsize=(10, 6), dpi=150)
    for key in ['rate_100', 'relax_0050', 'relax_0075', 'relax_0100']:
        data = targets[key]
        sim = sim_results[key]
        plt.plot(data['time'], data['stress'], 's', color='darkred', markersize=3, label='Exp' if key=='rate_100' else "")
        plt.plot(data['time'], sim, '-', color='#0070C0', linewidth=1.5, label='Sim' if key=='rate_100' else "")
        
    plt.xlabel('Time (secs)', fontweight='bold')
    plt.ylabel('Engineering Stress (MPa)', fontweight='bold')
    plt.title('Final PRF model after 3 successive HJ optimizations', fontweight='bold', fontsize=14)
    plt.xlim(-10, 350)
    plt.ylim(0, 25)
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.tight_layout()
    
    out_dir = os.path.dirname(__file__)
    out_path1 = os.path.join(out_dir, '..', '..', '..', 'dev_log', 'resources', 'step10_fig20_time.png')
    plt.savefig(out_path1)
    print(f"Saved {out_path1}")
    
    # Figure 21: Stress vs Strain
    plt.figure(figsize=(10, 6), dpi=150)
    for key in ['rate_100', 'relax_0050', 'relax_0075', 'relax_0100']:
        data = targets[key]
        sim = sim_results[key]
        plt.plot(data['strain'], data['stress'], 's', color='darkred', markersize=3, label='Exp' if key=='rate_100' else "")
        plt.plot(data['strain'], sim, '-', color='#0070C0', linewidth=1.5, label='Sim' if key=='rate_100' else "")
        
    plt.xlabel('Strain', fontweight='bold')
    plt.ylabel('Engineering Stress (MPa)', fontweight='bold')
    plt.title('Final PRF model after 3 successive HJ optimizations', fontweight='bold', fontsize=14)
    plt.xlim(0, 0.012)
    plt.ylim(0, 25)
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.tight_layout()
    
    out_path2 = os.path.join(out_dir, '..', '..', '..', 'dev_log', 'resources', 'step10_fig21_strain.png')
    plt.savefig(out_path2)
    print(f"Saved {out_path2}")

if __name__ == '__main__':
    main()
