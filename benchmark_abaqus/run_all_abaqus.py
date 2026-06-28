"""
WHTOOLs MATCALIB 2026
--------------------------------------------------------------------------------
Bulk Abaqus Execution and Extraction Script
Runs Abaqus sequentially for all matching .inp files and extracts results.

Usage:
    python run_all_abaqus.py
--------------------------------------------------------------------------------
"""
import os
import glob
import subprocess

def main():
    benchmark_dir = r"d:\PythonCodeStudy\WHT_PRF\benchmark_abaqus"
    os.chdir(benchmark_dir)
    
    # Collect all INP files that do NOT start with REP_
    inp_files = glob.glob("viscnet_*.inp") + glob.glob("x_viscnet_*.inp")
    inp_files = [f for f in inp_files if not f.startswith("REP_")]
    
    total = len(inp_files)
    print(f"Found {total} files to process.")
    
    for i, inp in enumerate(inp_files):
        job_name = os.path.splitext(inp)[0]
        results_file = f"{job_name}_abaqus_results.txt"
        
        # Skip if already extracted
        if os.path.exists(results_file):
            print(f"[{i+1}/{total}] Skipping {job_name}, results already exist.")
            continue
            
        print(f"[{i+1}/{total}] Running Abaqus for {job_name}...")
        try:
            # Run Abaqus
            subprocess.run(
                f"abaqus job={job_name} interactive",
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Run extraction
            print(f"[{i+1}/{total}] Extracting results for {job_name}...")
            subprocess.run(
                f"abaqus python extract_odb.py {job_name}.odb",
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"Error processing {job_name}: {e}")

if __name__ == "__main__":
    main()
