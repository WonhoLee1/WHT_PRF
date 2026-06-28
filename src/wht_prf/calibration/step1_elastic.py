import pandas as pd
import numpy as np
import os

def determine_elastic_modulus(data_path, strain_limit=0.01):
    """
    Step 1: Determine elastic modulus from the highest strain rate test data.
    E = stress / strain in the linear elastic region.
    """
    df = pd.read_excel(data_path, sheet_name='test rate 100')
    
    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]
    df = df.iloc[1:].reset_index(drop=True)
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna()
    
    # Filter for low strain to find elastic region
    linear_region = df[(df['Eng. Strain'] > 0.001) & (df['Eng. Strain'] <= strain_limit)]
    
    if len(linear_region) == 0:
        print("No data points found in the specified linear region.")
        return None
        
    # Fit a line (y = E*x) through the origin or (y = E*x + b)
    # E = average of (stress / strain)
    strain = linear_region['Eng. Strain'].values
    stress = linear_region['Eng. Stress'].values
    
    # Least squares fit without intercept: stress = E * strain -> E = sum(stress*strain) / sum(strain^2)
    E = np.sum(stress * strain) / np.sum(strain**2)
    
    print(f"Determined Elastic Modulus (E) from 100/s data: {E:.2f} MPa")
    
    # The Abaqus PRF card given in the paper has NeoHooke Moduli C10 = 549.6 MPa.
    # For NeoHooke, E = 6 * C10.
    # E = 6 * 549.6 = 3297.6 MPa. Let's see if it matches.
    print(f"Target E from NeoHooke (C10=549.6): {6 * 549.6:.2f} MPa")
    
    return E

if __name__ == "__main__":
    file_path = 'src/wht_prf/data/decimated_test_data.xlsx'
    if os.path.exists(file_path):
        determine_elastic_modulus(file_path, strain_limit=0.005)
