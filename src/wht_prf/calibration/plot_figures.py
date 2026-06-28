import pandas as pd
import matplotlib.pyplot as plt
import os
import koreanize_matplotlib

def load_data(file_path='src/wht_prf/data/decimated_test_data.xlsx'):
    print(f"Loading data from {file_path}...")
    df_dict = pd.read_excel(file_path, sheet_name=None)
    
    processed_dict = {}
    for sheet, df in df_dict.items():
        # Find the row that contains 'Time'
        header_idx = -1
        if 'Time' in [str(c).strip() for c in df.columns]:
            header_idx = -1 # header is already columns
        else:
            for idx, row in df.iterrows():
                if 'Time' in [str(v).strip() for v in row.values]:
                    header_idx = idx
                    break
        
        if header_idx != -1:
            df.columns = [str(c).strip() for c in df.iloc[header_idx].values]
            df = df.iloc[header_idx+1:].reset_index(drop=True)
        else:
            df.columns = [str(c).strip() for c in df.columns]
            
        # Drop rows that are just units or NaN
        df = df[~df['Time'].astype(str).str.contains('secs', na=False)]
        df = df.dropna(subset=['Time', 'Eng. Stress'])
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        processed_dict[sheet] = df
    return processed_dict

def plot_relaxation_tests(data_dict, output_dir='dev_log/figures'):
    """Reproduces Figure 3 (Stress relaxation tests)"""
    plt.figure(figsize=(8, 6))
    
    relax_sheets = ['test relax 0050', 'test relax 0075', 'test relax 0100', 'test relax 0150']
    labels = ['0.5% strain', '0.75% strain', '1.0% strain', '1.5% strain']
    
    for sheet, label in zip(relax_sheets, labels):
        if sheet in data_dict:
            df = data_dict[sheet]
            # Plot Time vs Eng. Stress
            plt.plot(df['Time'], df['Eng. Stress'], label=label)
            
    plt.xlabel('Time (s)')
    plt.ylabel('Engineering Stress (MPa)')
    plt.title('Figure 3: Stress relaxation test suite responses')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'fig3_stress_relaxation.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

def plot_rate_test(data_dict, output_dir='dev_log/figures'):
    """Reproduces Figure 4 (Simple tension tests at strain rates) - using available data"""
    plt.figure(figsize=(8, 6))
    
    if 'test rate 100' in data_dict:
        df = data_dict['test rate 100']
        # Plot Eng. Strain vs Eng. Stress
        plt.plot(df['Eng. Strain'], df['Eng. Stress'], label='100 /sec rate')
            
    plt.xlabel('Engineering Strain (-)')
    plt.ylabel('Engineering Stress (MPa)')
    plt.title('Figure 4: Tension pull tests (High Strain Rate)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'fig4_tension_rate.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    data_dict = load_data()
    plot_relaxation_tests(data_dict)
    plot_rate_test(data_dict)
