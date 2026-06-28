import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import koreanize_matplotlib

# Import the fitting function from step2
from step2_prony_series import prony_series_3, fit_prony_series

def check_nonlinearity(data_path, output_dir='dev_log/figures'):
    """
    Step 3: Show the Prony series for all relaxation data to demonstrate that nonlinear viscoelasticity is needed.
    """
    # 1. Get the fitted Prony parameters from 0.5% data
    popt, _ = fit_prony_series(data_path, output_dir=output_dir)
    
    # 2. Load all relaxation data
    print("Loading all relaxation data...")
    df_dict = pd.read_excel(data_path, sheet_name=None)
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'orange', 'red']
    strains_label = ['0.5% strain', '0.75% strain', '1.0% strain', '1.5% strain']
    sheets = ['test relax 0050', 'test relax 0075', 'test relax 0100', 'test relax 0150']
    
    for i, (sheet, label) in enumerate(zip(sheets, strains_label)):
        df = df_dict[sheet]
        
        # Parse headers
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
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        # Isolate relaxation phase
        peak_idx = df['Eng. Stress'].idxmax()
        df_relax = df.loc[peak_idx:].copy()
        
        time = df_relax['Time'].values - df_relax['Time'].values[0]
        stress = df_relax['Eng. Stress'].values
        strain_val = df_relax['Eng. Strain'].mean()
        
        # Predict linear viscoelastic stress
        E_t = prony_series_3(time, *popt)
        stress_pred = E_t * strain_val
        
        plt.plot(time, stress, 'o', color=colors[i], markersize=2, label=f'Test: {label}', alpha=0.5)
        plt.plot(time, stress_pred, '-', color=colors[i], linewidth=2, label=f'Linear Pred: {label}')
        
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Engineering Stress (MPa)')
    plt.title('Step 3: Linear Viscoelastic Prediction vs Test Data (Nonlinearity Check)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'step3_nonlinearity_check.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved nonlinearity check plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    file_path = 'src/wht_prf/data/decimated_test_data.xlsx'
    if os.path.exists(file_path):
        check_nonlinearity(file_path)
