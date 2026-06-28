import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the JAX simulator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from wht_prf.calibration.step10_final_validation import simulate_test
import jax.numpy as jnp

df = pd.read_csv(r"dev_log\radioss_runs\cylinder_relax_law100\foam_relaxT01.csv", header=None)
time = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
time = np.nan_to_num(time)

sig_xx = pd.to_numeric(df.iloc[:, 47], errors='coerce').values
sig_xx = np.nan_to_num(sig_xx)

disp_scale = pd.to_numeric(df.iloc[:, 46], errors='coerce').values
disp_scale = np.nan_to_num(disp_scale)

# Assume cylinder length is 30mm and imposed disp is -1mm
strain_approx = disp_scale * (-1.0 / 30.0)

# Theoretical Simulation using JAX
# Radioss uses the parameters we just verified
E_inst = 3297.6
params = {
    'A': jnp.array([4.8828e-7, 2.4414e-6, 7.81403e-5]),
    'n': jnp.array([3.13398, 4.91992, 3.801953]),
    'm': jnp.array([-0.551660, -0.746191, -0.616699]),
    's': jnp.array([0.337060, 0.150654, 0.372607])
}

# The simulation time array
# Note: since the Radioss data is sampled at variable dt, we extract dt from time array
dt_arr = np.diff(time)
dt_arr = np.where(dt_arr <= 0, 1e-6, dt_arr)
# We also flip the sign of strain for the 1D model to compare magnitude of stress
sim_stress = simulate_test(params, jnp.array(dt_arr), jnp.array(np.abs(strain_approx)), E_inst)

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

# Time vs Stress
# We flip the sign of Radioss stress to make it positive for comparison (compression)
ax1.plot(time, -sig_xx, 's', color='darkred', markersize=4, label="OpenRadioss (SIGXX)")
ax1.plot(time, np.array(sim_stress), '-', color='#0070C0', lw=2, label="JAX Theoretical PRF")
ax1.set_xlabel("Time (s)", fontweight='bold')
ax1.set_ylabel("Compressive Stress (MPa)", fontweight='bold')
ax1.set_title("Step 8: OpenRadioss Verification against JAX PRF Theory", fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

plt.tight_layout()
plt.savefig("dev_log/radioss_runs/cylinder_relax_law100/time_strain_stress.png")
print("Saved Step 8 comparison plot")
