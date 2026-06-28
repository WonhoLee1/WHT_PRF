"""
WHTOOLs MATCALIB 2026
--------------------------------------------------------------------------------
Bulk JAX PRF Execution Script
Runs all generated JAX benchmark Python scripts in parallel using ProcessPoolExecutor.
Max workers is set to 4 to balance JIT compilation speed and RAM usage.

Usage:
    python run_all_jax_benchmarks.py
--------------------------------------------------------------------------------
"""
import os
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_script(script):
    job_name = os.path.basename(script)
    try:
        # Run the python script
        subprocess.run(
            f"python {job_name}",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return (job_name, True, None)
    except Exception as e:
        return (job_name, False, str(e))

def main():
    benchmark_dir = r"d:\PythonCodeStudy\WHT_PRF\benchmark_abaqus"
    os.chdir(benchmark_dir)
    
    # Exclude the bulk scripts themselves
    all_scripts = glob.glob("run_viscnet_*.py") + glob.glob("run_x_viscnet_*.py")
    all_scripts = [s for s in all_scripts if not s.startswith("run_all_")]
    
    total = len(all_scripts)
    print(f"Found {total} JAX PRF scripts to execute.")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_script, s): s for s in all_scripts}
        
        for i, future in enumerate(as_completed(futures)):
            script_name = futures[future]
            try:
                name, success, error = future.result()
                if success:
                    print(f"[{i+1}/{total}] SUCCESS: {name}")
                    success_count += 1
                else:
                    print(f"[{i+1}/{total}] FAILED: {name} - {error}")
            except Exception as e:
                print(f"[{i+1}/{total}] ERROR processing future for {script_name}: {e}")

    print(f"\nExecution Complete: {success_count}/{total} succeeded.")

if __name__ == "__main__":
    main()
