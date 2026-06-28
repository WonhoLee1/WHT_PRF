"""
WHTOOLs MATCALIB 2026
--------------------------------------------------------------------------------
Bulk Plotting Script
Reads Abaqus text extraction files and creates a 3-panel PNG plot for each.

Usage:
    python plot_all_results.py
--------------------------------------------------------------------------------
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def main():
    benchmark_dir = r"d:\PythonCodeStudy\WHT_PRF\benchmark_abaqus"
    os.chdir(benchmark_dir)
    
    txt_files = glob.glob("*_abaqus_results.txt")
    txt_files = [f for f in txt_files if not f.startswith("REP_")]
    
    total = len(txt_files)
    print(f"Found {total} extracted result files.")
    
    for i, txt in enumerate(txt_files):
        job_name = txt.replace("_abaqus_results.txt", "")
        png_name = f"{job_name}_graphs.png"
        
        try:
            data = np.loadtxt(txt, skiprows=1)
            if data.shape[0] == 0 or data.shape[1] < 3:
                print(f"[{i+1}/{total}] Skipping {job_name}: Empty or invalid data.")
                continue
                
            time = data[:, 0]
            stress = data[:, 1]
            strain = data[:, 2]
            
            plt.figure(figsize=(15, 4))
            
            # Time vs Strain
            plt.subplot(1, 3, 1)
            plt.plot(time, strain, 'b-')
            plt.xlabel('Time')
            plt.ylabel('Strain (11)')
            plt.title('Time vs Strain')
            
            # Time vs Stress
            plt.subplot(1, 3, 2)
            plt.plot(time, stress, 'r-')
            plt.xlabel('Time')
            plt.ylabel('Stress (11)')
            plt.title('Time vs Stress')
            
            # Strain vs Stress
            plt.subplot(1, 3, 3)
            plt.plot(strain, stress, 'g-')
            plt.xlabel('Strain (11)')
            plt.ylabel('Stress (11)')
            plt.title('Strain vs Stress')
            
            plt.tight_layout()
            plt.savefig(png_name)
            plt.close()
            
            print(f"[{i+1}/{total}] Saved {png_name}")
        except Exception as e:
            print(f"Error plotting {job_name}: {e}")

if __name__ == "__main__":
    main()
