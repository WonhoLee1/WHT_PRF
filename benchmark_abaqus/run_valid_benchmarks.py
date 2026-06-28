import glob
import json
import subprocess
import concurrent.futures

scripts = glob.glob('run_viscnet_*.py')
valid_scripts = []
for s in scripts:
    if s.startswith('run_x_'): continue
    valid_scripts.append(s)

def run_script(s):
    print(f"Running {s}...")
    subprocess.run(["python", s])
    return s

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        list(executor.map(run_script, valid_scripts))

    # After running, run generate_benchmarks.py to compile the report
    subprocess.run(["python", "generate_benchmarks.py"])
    print("Finished!")
