import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import koreanize_matplotlib

def prony_series_3(t, e_inf, e1, tau1, e2, tau2, e3, tau3):
    """
    3-term Prony series for stress relaxation.
    E(t) = E_inf + E1*exp(-t/tau1) + E2*exp(-t/tau2) + E3*exp(-t/tau3)
    """
    return e_inf + e1 * np.exp(-t / tau1) + e2 * np.exp(-t / tau2) + e3 * np.exp(-t / tau3)

def fit_prony_series(data_path, output_dir='dev_log/figures'):
    """
    Step 2: Calculate a Prony series viscoelasticity from only the lowest stress relaxation data (0.5% strain).
    """
    print(f"Loading 0.5% strain relaxation data from {data_path}...")
    df = pd.read_excel(data_path, sheet_name='test relax 0050')
    
    # Process headers
    header_idx = -1
    for idx, row in df.iterrows():
        if 'Time' in [str(v).strip() for v in row.values]:
            header_idx = idx
            break
            
    if header_idx != -1:
        df.columns = [str(c).strip() for c in df.iloc[header_idx].values]
        df = df.iloc[header_idx+1:].reset_index(drop=True)
    else:
        df.columns = [str(c).strip() for c in df.columns]
        
    df = df[~df['Time'].astype(str).str.contains('secs', na=False)]
    df = df.dropna(subset=['Time', 'Eng. Stress', 'Eng. Strain'])
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    # The data includes a ramp-up phase. We need to isolate the relaxation hold phase.
    # The hold phase starts near the peak stress.
    peak_idx = df['Eng. Stress'].idxmax()
    df_relax = df.loc[peak_idx:].copy()
    
    # Shift time so t=0 at the start of hold
    time = df_relax['Time'].values - df_relax['Time'].values[0]
    stress = df_relax['Eng. Stress'].values
    strain_val = df_relax['Eng. Strain'].mean() # Approximate constant strain during hold
    
    # Calculate relaxation modulus E(t) = sigma(t) / epsilon_0
    modulus = stress / strain_val
    
    # Initial guesses for 3-term Prony
    # tau usually span decades: e.g., 0.1, 10, 1000
    e0_guess = max(modulus[0], 1.0)
    e_inf_guess = max(modulus[-1], 0.1)
    
    if e0_guess <= e_inf_guess:
        e0_guess = e_inf_guess * 2
        
    e_i_guess = (e0_guess - e_inf_guess) / 3
    
    p0 = [e_inf_guess, e_i_guess, 0.1, e_i_guess, 10.0, e_i_guess, 1000.0]
    
    # Bounds to ensure positivity
    lower_bounds = [0, 0, 0.001, 0, 0.1, 0, 10.0]
    upper_bounds = [np.inf, np.inf, 10.0, np.inf, 1000.0, np.inf, 100000.0]
    
    print(f"p0 = {p0}")
    
    print("Fitting 3-term Prony series...")
    popt, pcov = curve_fit(prony_series_3, time, modulus, p0=p0, bounds=(lower_bounds, upper_bounds))
    
    e_inf, e1, tau1, e2, tau2, e3, tau3 = popt
    e_inst = e_inf + e1 + e2 + e3
    
    # Calculate dimensionless g_i = E_i / E_inst
    g1 = e1 / e_inst
    g2 = e2 / e_inst
    g3 = e3 / e_inst
    
    print("\n--- Fitted Prony Series Parameters ---")
    print(f"E_inst: {e_inst:.2f} MPa")
    print(f"E_inf:  {e_inf:.2f} MPa")
    print(f"g1: {g1:.4f}, tau1: {tau1:.4f} s")
    print(f"g2: {g2:.4f}, tau2: {tau2:.4f} s")
    print(f"g3: {g3:.4f}, tau3: {tau3:.4f} s")
    print("--------------------------------------")
    
    # Plotting
    modulus_fit = prony_series_3(time, *popt)
    stress_fit = modulus_fit * strain_val
    
    plt.figure(figsize=(8, 6))
    plt.plot(time, stress, 'ko', markersize=2, label='Test Data (0.5% strain)', alpha=0.5)
    plt.plot(time, stress_fit, 'r-', linewidth=2, label='3-Term Prony Fit')
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Engineering Stress (MPa)')
    plt.title('Step 2: Linear Viscoelastic Fit (Lowest Strain)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'step2_prony_fit.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved fit plot to {out_path}")
    plt.close()
    
    return popt, (g1, tau1, g2, tau2, g3, tau3)

if __name__ == "__main__":
    file_path = 'src/wht_prf/data/decimated_test_data.xlsx'
    if os.path.exists(file_path):
        fit_prony_series(file_path)
