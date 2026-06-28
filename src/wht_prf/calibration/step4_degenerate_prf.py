import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import koreanize_matplotlib

from step2_prony_series import fit_prony_series

def convert_prony_to_degenerate_prf(g_i, tau_i, E_inst):
    """
    Convert Prony series parameters to Degenerate PRF parameters.
    For a linear Maxwell element, the creep strain rate is:
        e_dot_cr = sigma / (E_i * tau_i)
    Abaqus PRF (LAW=STRAIN) effective creep rate equation:
        e_dot_cr = ( A * q^n * ((1+m)*e_cr)^m )^(1/(1+m))
    For degenerate PRF (linear):
        m = 0
        n = 1
        A = 1 / (E_i * tau_i)
        SRatio = g_i
    """
    E_i = g_i * E_inst
    A_i = 1.0 / (E_i * tau_i)
    n_i = 1.0
    m_i = 0.0
    return A_i, n_i, m_i

def run_degenerate_prf_1d(data_path, output_dir='dev_log/figures'):
    """
    Step 4 & 5: Convert to degenerate PRF and show response.
    Simulate the degenerate PRF using a 1D implicit/explicit integration and compare to the test data.
    """
    # 1. Get Prony parameters from Step 2
    popt, (g1, tau1, g2, tau2, g3, tau3) = fit_prony_series(data_path, output_dir)
    E_inst = popt[0] + popt[1] + popt[3] + popt[5]  # E_inf + E1 + E2 + E3
    E_inf = popt[0]
    
    # 2. Convert to PRF parameters
    print("\n--- Degenerate PRF Parameters ---")
    networks = [(g1, tau1), (g2, tau2), (g3, tau3)]
    prf_params = []
    for i, (g, tau) in enumerate(networks):
        A, n, m = convert_prony_to_degenerate_prf(g, tau, E_inst)
        q0 = 1.0 / A  # New Power Law parameter
        prf_params.append({'A': A, 'q0': q0, 'n': n, 'm': m, 'SRatio': g})
        print(f"Network {i+1}: SRatio={g:.6f}, (Old) A={A:.6e}, (New) q0={q0:.4f}, n={n:.1f}, m={m:.1f}")
        
    # 3. Simulate Degenerate PRF (1D Small Strain) for 0.5% Relaxation
    df = pd.read_excel(data_path, sheet_name='test relax 0050')
    header_idx = -1
    for idx, row in df.iterrows():
        if 'Time' in [str(v).strip() for v in row.values]:
            header_idx = idx; break
    df.columns = [str(c).strip() for c in df.iloc[header_idx].values]
    df = df.iloc[header_idx+1:].reset_index(drop=True)
    df = df[~df['Time'].astype(str).str.contains('secs', na=False)]
    for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Time', 'Eng. Stress', 'Eng. Strain'])
    
    # Isolate relaxation
    peak_idx = df['Eng. Stress'].idxmax()
    df_relax = df.loc[peak_idx:].copy()
    time = df_relax['Time'].values - df_relax['Time'].values[0]
    stress_exp = df_relax['Eng. Stress'].values
    strain_val = df_relax['Eng. Strain'].mean()
    
    # Simulation array
    dt_arr = np.diff(time)
    dt_arr = np.insert(dt_arr, 0, 0)
    
    # Initial states
    # In a relaxation test, a strain is applied instantaneously (or ramped up).
    # Since we are starting from the peak, the elastic stress is already developed.
    # Total strain is constant.
    e_cr = np.zeros(3)  # creep strain for 3 networks
    stress_sim = np.zeros(len(time))
    
    # Initial stress at t=0
    # elastic stress = E_i * (strain_val - e_cr_i)
    # total stress = E_inf * strain_val + sum(E_i * (strain_val - e_cr_i))
    for step in range(len(time)):
        dt = dt_arr[step]
        if dt > 0:
            # Update creep strains (Explicit Euler for simplicity, or semi-implicit)
            # e_dot_cr = A * (sigma_i)
            # sigma_i = E_i * (strain_val - e_cr)
            for i in range(3):
                E_i = prf_params[i]['SRatio'] * E_inst
                A_i = prf_params[i]['A']
                # Semi-implicit: e_cr_new = e_cr_old + dt * A_i * E_i * (strain_val - e_cr_new)
                # e_cr_new * (1 + dt*A_i*E_i) = e_cr_old + dt*A_i*E_i*strain_val
                term = dt * A_i * E_i
                e_cr[i] = (e_cr[i] + term * strain_val) / (1.0 + term)
                
        # Calculate stress
        sig_tot = E_inf * strain_val
        for i in range(3):
            E_i = prf_params[i]['SRatio'] * E_inst
            sig_i = E_i * (strain_val - e_cr[i])
            sig_tot += sig_i
        stress_sim[step] = sig_tot
        
    plt.figure(figsize=(8, 6))
    plt.plot(time, stress_exp, 'ko', markersize=2, label='Test Data (0.5%)', alpha=0.5)
    plt.plot(time, stress_sim, 'r-', linewidth=2, label='Degenerate PRF Simulation')
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Engineering Stress (MPa)')
    plt.title('Step 4 & 5: Degenerate PRF Model Response')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'step4_degenerate_prf.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved degenerate PRF plot to {out_path}")

if __name__ == "__main__":
    file_path = 'src/wht_prf/data/decimated_test_data.xlsx'
    if os.path.exists(file_path):
        run_degenerate_prf_1d(file_path)
