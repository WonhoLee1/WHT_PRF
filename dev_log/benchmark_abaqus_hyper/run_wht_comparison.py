# WHTOOLs MATCALIB 2026
import pandas as pd
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import glob
import sys
import os

# Ensure korean matplotlib font if applicable
try:
    import koreanize_matplotlib
except ImportError:
    pass

sys.path.append('../../src')
from wht_prf.equilibrium import compute_cauchy_stress

PARAMS = {
    'n': jnp.array([100.0, 1e-5]),
    'y': jnp.array([100.0, -1.0, 0.01, 1e-5]),
    'a': jnp.array([200.0, 5.0, 1e-5]),
}

MODELS = {
    'n': 'NEO_HOOKEAN',
    'y': 'YEOH',
    'a': 'ARRUDA_BOYCE'
}

csv_files = glob.glob('*.csv')

for csv_file in csv_files:
    basename = csv_file.replace('.csv', '')
    mat_code = basename[2] # e.g. mh n coo3hut -> n
    params = PARAMS[mat_code]
    model_type = MODELS[mat_code]
    
    df = pd.read_csv(csv_file)
    
    wht_s11 = []
    wht_s22 = []
    
    for idx, row in df.iterrows():
        le11, le22, le33, le12 = row['LE11'], row['LE22'], row['LE33'], row['LE12']
        
        # Abaqus returns Logarithmic Strain (LE)
        # Principal stretches: lambda_i = exp(LE_i)
        F11 = np.exp(le11)
        F22 = np.exp(le22)
        F33 = np.exp(le33)
        
        F = jnp.array([
            [F11, 0.0, 0.0],
            [0.0, F22, 0.0],
            [0.0, 0.0, F33]
        ])
        
        # Compute 3D Cauchy stress
        sigma = compute_cauchy_stress(model_type, F, params)
        
        # For these benchmark models (uniaxial, biaxial, planar), the Z direction is free.
        # Thus sigma_33 should be 0. We subtract the calculated sigma_33 from all components 
        # to implicitly solve for the pressure term enforcing sigma_33 = 0.
        s11_true = sigma[0,0] - sigma[2,2]
        s22_true = sigma[1,1] - sigma[2,2]
        
        wht_s11.append(float(s11_true))
        wht_s22.append(float(s22_true))
        
    df['WHT_S11'] = wht_s11
    df['WHT_S22'] = wht_s22
    
    # Calculate Engineering Stresses and Strains
    # True Stress = Eng Stress * lambda => Eng Stress = True Stress / lambda
    df['Eng_S11'] = df['S11'] / np.exp(df['LE11'])
    df['WHT_Eng_S11'] = df['WHT_S11'] / np.exp(df['LE11'])
    
    # Avoid division by zero or very small numbers for lambda_22 if it's compressed
    df['Eng_S22'] = df['S22'] / np.exp(df['LE22'])
    df['WHT_Eng_S22'] = df['WHT_S22'] / np.exp(df['LE22'])
    
    df['E11'] = np.exp(df['LE11']) - 1.0
    df['E22'] = np.exp(df['LE22']) - 1.0

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # (a) True Stress vs True Strain
    ax1.plot(df['LE11'], df['S11'], 'k--', linewidth=2, label='Abaqus S11')
    ax1.plot(df['LE11'], df['WHT_S11'], 'r-', linewidth=1.5, alpha=0.8, label='WHT_PRF S11')
    
    if 'ibt' in basename or 'gsh' in basename:
        ax1.plot(df['LE22'], df['S22'], 'k:', linewidth=2, label='Abaqus S22')
        ax1.plot(df['LE22'], df['WHT_S22'], 'b-', linewidth=1.5, alpha=0.8, label='WHT_PRF S22')
        
    ax1.set_xlabel('True Strain (LE)', fontsize=12)
    ax1.set_ylabel('True Stress (Cauchy Stress)', fontsize=12)
    ax1.set_title(f'True Stress-Strain ({model_type})', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # (b) Eng Stress vs Eng Strain
    ax2.plot(df['E11'], df['Eng_S11'], 'k--', linewidth=2, label='Abaqus S11')
    ax2.plot(df['E11'], df['WHT_Eng_S11'], 'r-', linewidth=1.5, alpha=0.8, label='WHT_PRF S11')
    
    if 'ibt' in basename or 'gsh' in basename:
        ax2.plot(df['E22'], df['Eng_S22'], 'k:', linewidth=2, label='Abaqus S22')
        ax2.plot(df['E22'], df['WHT_Eng_S22'], 'b-', linewidth=1.5, alpha=0.8, label='WHT_PRF S22')
        
    ax2.set_xlabel('Engineering Strain (E)', fontsize=12)
    ax2.set_ylabel('Engineering Stress (Nominal Stress)', fontsize=12)
    ax2.set_title(f'Engineering Stress-Strain ({model_type})', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle(f'Verification: Abaqus vs WHT_PRF - {basename.upper()}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{basename}_comparison.png', dpi=150)
    plt.close()
    
    # Validation Print
    err_s11 = np.max(np.abs(df['S11'] - df['WHT_S11']))
    print(f"[{basename}] Max True S11 Error: {err_s11:.4e}")

print("Validation complete. Plots generated.")
