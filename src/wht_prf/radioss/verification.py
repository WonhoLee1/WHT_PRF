import os
import sys
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from wht_prf.radioss.deck_builder import create_radioss_deck
from wht_prf.radioss.runner import run_openradioss
from wht_prf.radioss.parser import parse_radioss_t01

def main():
    exec_dir = "D:/OpenRadioss_win64/OpenRadioss/exec"
    run_dir = os.path.abspath("dev_log/radioss_runs/tension_100")
    
    # We use the optimized parameters from Step 6:
    # E_inst = 6.0 * 549.6 = 3297.6
    # Network 1: SRatio=0.2308, A=3.8816e-03, n=1.3447, m=-0.2895
    # Network 2: SRatio=0.0130, A=3.2325e-04, n=2.7448, m=-0.3389
    # Network 3: SRatio=0.6562, A=7.7803e-05, n=1.2020, m=-0.6158
    
    prf_data = {
        "material_name": "MNF_MAT",
        "hyperelastic_type": "YEOH",
        "hyperelastic_params": [549.6, 0.0, 0.0],
        "networks": [
            {
                "stiffness_ratio": 0.2308,
                "creep_params": [3.8816e-03, 1.3447, -0.2895]
            },
            {
                "stiffness_ratio": 0.0130,
                "creep_params": [3.2325e-04, 2.7448, -0.3389]
            },
            {
                "stiffness_ratio": 0.6562,
                "creep_params": [7.7803e-05, 1.2020, -0.6158]
            }
        ]
    }
    
    print("Creating deck...")
    create_radioss_deck(run_dir, "test_tension", "tension", 
                       target_strain=0.015, strain_rate=100.0, 
                       prf_data=prf_data, end_time=0.0002)
                       
    print("Running OpenRadioss...")
    success = run_openradioss(run_dir, "test_tension", exec_dir)
    
    if success:
        print("Success! Parsing T01...")
        t01_file = os.path.join(run_dir, "test_tensionT01")
        if os.path.exists(t01_file):
            data = parse_radioss_t01(t01_file)
            print(f"Parsed {len(data.get('time', []))} steps.")
        else:
            print("T01 file not found.")

if __name__ == "__main__":
    main()
