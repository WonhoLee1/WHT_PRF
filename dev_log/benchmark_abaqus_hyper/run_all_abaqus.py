import os
import subprocess
import glob

inp_files = glob.glob('*.inp')

for inp in inp_files:
    job_name = inp.replace('.inp', '')
    if os.path.exists(job_name + '.csv'):
        continue
    
    print(f'Running Abaqus for {job_name}...')
    subprocess.run(f'abaqus job={job_name} interactive', shell=True, check=True)
    
    print(f'Extracting ODB for {job_name}...')
    subprocess.run(f'abaqus python extract_odb.py {job_name}.odb', shell=True, check=True)

print('All Abaqus jobs completed!')
