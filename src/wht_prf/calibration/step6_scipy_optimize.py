import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_data():
    base_dir = r"G:\Simulia_Video_Contents\Material Calibration\Calibration of a PRF Material Model for Polypropylene\PPC3TF2TestData"
    data_files = {
        'relax_0075': 'relax_data_0075.txt',
        'relax_0100': 'relax_data_0100.txt',
        'relax_0150': 'relax_data_0150.txt'
    }
    
    targets = {}
    for key, filename in data_files.items():
        path = os.path.join(base_dir, filename)
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

def simulate_1d_numpy(params, time_arr, strain_arr, dt_arr, E_inst):
    A = params['A']
    n = params['n']
    m = params['m']
    s = params['s']
    E_i = s * E_inst
    
    N = len(dt_arr)
    stress_sim = np.zeros(N+1)
    stress_sim[0] = E_inst * strain_arr[0]
    
    e_cr = np.zeros(3)
    num_substeps = 10
    
    for i in range(N):
        dt = dt_arr[i]
        dt_sub = dt / num_substeps
        strain_tot = strain_arr[i+1]
        
        for _ in range(num_substeps):
            sig_i = E_i * (strain_tot - e_cr)
            q = np.abs(sig_i)
            e_cr_eff = np.abs(e_cr) + 1e-8
            
            base = np.clip(A * (q**n) * (((1.0 + m) * e_cr_eff)**m), 0.0, 1e10)
            om1 = 1.0 / (1.0 + m)
            e_dot_mag = np.clip(base ** om1, 0.0, 1e5)
            
            e_dot = e_dot_mag * np.sign(sig_i)
            e_cr = e_cr + e_dot * dt_sub
            
        s_inf = 1.0 - np.sum(s)
        sig_inf = s_inf * E_inst * strain_tot
        stress_sim[i+1] = sig_inf + np.sum(E_i * (strain_tot - e_cr))
        
    return stress_sim

def main():
    targets = load_data()
    E_inst = 2316.39
    
    keys = ['relax_0075', 'relax_0100', 'relax_0150']
    
    def objective(x):
        # x = [A1, A2, A3, n1, n2, n3, m1, m2, m3]
        A = np.exp(x[0:3]) * 1e-4
        n = np.exp(x[3:6]) + 1.0
        m = -1.0 / (1.0 + np.exp(-x[6:9])) * 0.99
        s = np.array([0.0397, 0.0800, 0.1537])
        
        params = {'A': A, 'n': n, 'm': m, 's': s}
        
        loss = 0.0
        for key in keys:
            t = targets[key]['time']
            st = targets[key]['strain']
            dt = targets[key]['dt']
            exp_sig = targets[key]['stress']
            
            sim_sig = simulate_1d_numpy(params, t, st, dt, E_inst)
            loss += np.mean((sim_sig - exp_sig)**2)
            
        return loss
        
    # Initial guess
    # A = [3.34e-03, 2.30e-04, 1.39e-05]
    # log(A*1e4) => log([33.4, 2.3, 0.139]) = [3.5, 0.83, -1.97]
    x0 = np.array([3.5, 0.83, -1.97, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0])
    
    print("Starting scipy optimization...")
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'disp': True})
    print(res)
    
    x = res.x
    A = np.exp(x[0:3]) * 1e-4
    n = np.exp(x[3:6]) + 1.0
    m = -1.0 / (1.0 + np.exp(-x[6:9])) * 0.99
    s = np.array([0.0397, 0.0800, 0.1537])
    
    best_params = {'A': A, 'n': n, 'm': m, 's': s}
    print("Best params:", best_params)
    
    plt.figure(figsize=(10, 6), dpi=150)
    colors = ['b', 'g', 'orange']
    labels = ['0.75% Strain', '1.0% Strain', '1.5% Strain']
    
    for j, key in enumerate(keys):
        t = targets[key]['time']
        st = targets[key]['strain']
        dt = targets[key]['dt']
        exp_sig = targets[key]['stress']
        
        sim_sig = simulate_1d_numpy(best_params, t, st, dt, E_inst)
        
        plt.plot(t, exp_sig, 'o', color=colors[j], markersize=3, alpha=0.5, label=f'Exp: {labels[j]}')
        plt.plot(t, sim_sig, '-', color=colors[j], linewidth=2, label=f'Sim: {labels[j]}')
               
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Engineering Stress (MPa)')
    plt.title('Step 6: Optimized PRF Model Response (Sub-stepping Explicit Scipy)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('C:/Users/GOODMAN/.gemini/antigravity/brain/d27d4e34-8ab9-4a61-bbc5-db108318fbaf/step6_implicit_prf.png')
    print("Saved step6_implicit_prf.png")
    
if __name__ == '__main__':
    main()
