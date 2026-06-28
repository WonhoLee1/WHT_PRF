import os
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from functools import partial

# Configure JAX for 64-bit precision (important for stiff ODEs/creep)
jax.config.update("jax_enable_x64", True)

def load_and_interpolate_data(data_path, num_steps=200):
    """
    Load the 4 target tests and interpolate them to a fixed number of steps
    for efficient JAX processing.
    """
    targets = {}
    sheets = {
        'relax_0075': 'test relax 0075',
        'relax_0100': 'test relax 0100',
        'relax_0150': 'test relax 0150',
        'rate_100': 'test rate 100'
    }
    
    df_dict = pd.read_excel(data_path, sheet_name=None)
    
    for key, sheet in sheets.items():
        df = df_dict[sheet]
        header_idx = -1
        if 'Time' in [str(c).strip() for c in df.columns]:
            header_idx = -1
            df.columns = [str(c).strip() for c in df.columns]
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
                
        df = df[~df['Time'].astype(str).str.contains('secs', na=False)]
        for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        
        time = df['Time'].values
        stress = df['Eng. Stress'].values
        strain = df['Eng. Strain'].values
        
        if 'relax' in key:
            peak_idx = df['Eng. Stress'].idxmax()
            time = time[peak_idx:] - time[peak_idx]
            stress = stress[peak_idx:]
            strain = strain[peak_idx:]
            
        # Log-spaced interpolation for relaxation (since it spans decades)
        # Linear for rate
        if 'relax' in key:
            t_interp = np.geomspace(max(time[0], 1e-3), time[-1], num_steps)
            t_interp = np.insert(t_interp, 0, 0.0)
        else:
            t_interp = np.linspace(0, time[-1], num_steps + 1)
            
        stress_interp = np.interp(t_interp, time, stress)
        strain_interp = np.interp(t_interp, time, strain)
        
        dt_arr = np.diff(t_interp)
        
        targets[key] = {
            'time': jnp.array(t_interp, dtype=jnp.float64),
            'dt': jnp.array(dt_arr, dtype=jnp.float64),
            'strain': jnp.array(strain_interp, dtype=jnp.float64),
            'stress': jnp.array(stress_interp, dtype=jnp.float64)
        }
        
    return targets

def prf_1d_step(state, inputs, params, E_inst):
    """
    A single time step integration for 1D PRF model using explicit Euler.
    """
    e_cr = state          # Shape (3,)
    dt, strain_tot = inputs
    
    A = params['A']       # Shape (3,)
    n = params['n']       # Shape (3,)
    m = params['m']       # Shape (3,)
    s = params['s']       # Shape (3,)
    
    # Elastic stiffness of each network
    E_i = s * E_inst
    
    # Current stress in each network
    sig_i = E_i * (strain_tot - e_cr)
    
    # PRF LAW=STRAIN effective creep strain rate
    # e_dot = ( A * q^n * ((1+m)*e_cr)^m )^(1/(1+m))
    # Note: ensure positivity for fractional powers
    q = jnp.abs(sig_i)
    e_cr_eff = jnp.abs(e_cr) + 1e-8
    
def prf_1d_step_implicit(state, inputs, params, E_inst):
    e_cr_old = state
    dt, strain_tot = inputs
    
    A = params['A']
    n = params['n']
    m = params['m']
    s = params['s']
    E_i = s * E_inst
    
    def residual(e_cr_new):
        sig_i = E_i * (strain_tot - e_cr_new)
        q = jnp.abs(sig_i)
        e_cr_eff = jnp.abs(e_cr_new) + 1e-8
        
        base = jnp.clip(A * (q**n) * (((1.0 + m) * e_cr_eff)**m), 0.0, 1e10)
        om1 = 1.0 / (1.0 + m)
        e_dot_mag = jnp.clip(base ** om1, 0.0, 1e5)
        
        e_dot = e_dot_mag * jnp.sign(sig_i)
        return e_cr_new - e_cr_old - e_dot * dt

    def newton_step(i, e_cr_curr):
        R = residual(e_cr_curr)
        # Compute exact Jacobian using JAX forward mode auto-diff
        J = jax.jacfwd(residual)(e_cr_curr)
        # Extract diagonal since networks are decoupled
        J_diag = jnp.diag(J)
        # Prevent division by zero
        J_diag = jnp.where(jnp.abs(J_diag) < 1e-8, 1e-8 * jnp.sign(J_diag + 1e-16), J_diag)
        dx = R / J_diag
        # Apply damped update to avoid wild jumps
        return e_cr_curr - 0.5 * dx

    # 10 iterations of Newton-Raphson
    e_cr_new = jax.lax.fori_loop(0, 10, newton_step, e_cr_old)
    
    # Total stress
    s_inf = 1.0 - jnp.sum(s)
    sig_inf = s_inf * E_inst * strain_tot
    sig_tot = sig_inf + jnp.sum(E_i * (strain_tot - e_cr_new))
    
    return e_cr_new, sig_tot

@partial(jax.jit, static_argnames=['E_inst'])
def simulate_test(params, dt_arr, strain_arr, E_inst):
    # Initial state: 0 creep strain for 3 networks
    init_e_cr = jnp.zeros(3, dtype=jnp.float64)
    
    # We need to scan over dt and strain
    # dt_arr is length N, strain_arr is length N+1. 
    # We will pass strain at t_{i+1} to compute stress at t_{i+1}
    inputs = (dt_arr, strain_arr[1:])
    
    scan_fn = lambda state, x: prf_1d_step_implicit(state, x, params, E_inst)
    
    _, stress_sim = jax.lax.scan(scan_fn, init_e_cr, inputs)
    
    # Prepend t=0 stress
    sig0 = E_inst * strain_arr[0]
    stress_sim = jnp.concatenate([jnp.array([sig0]), stress_sim])
    
    return stress_sim

@partial(jax.jit, static_argnames=['E_inst'])
def compute_loss(raw_params, batch_dt, batch_strain, batch_stress_exp, E_inst):
    """
    Compute MSE loss. We use vmap to run the 3 relaxation tests in parallel!
    """
    # Transform raw params to ensure physical bounds
    params = {
        'A': jax.nn.softplus(raw_params['A']) * 1e-4, # Scale A
        'n': jax.nn.softplus(raw_params['n']) + 1.0,  # n >= 1
        'm': -jax.nn.sigmoid(raw_params['m']) * 0.99,      # m in (-0.99, 0)
        's': jax.nn.softmax(raw_params['s']) * 0.9    # sum(s) = 0.9
    }
    
    # VMAP over the 3 relaxation tests
    # batch_dt: (3, N), batch_strain: (3, N+1)
    sim_fn = lambda dt, st: simulate_test(params, dt, st, E_inst)
    batch_stress_sim = jax.vmap(sim_fn)(batch_dt, batch_strain)
    
    loss_relax = jnp.mean((batch_stress_sim - batch_stress_exp)**2)
    
    return loss_relax, params

def main():
    data_path = 'src/wht_prf/data/decimated_test_data.xlsx'
    targets = load_and_interpolate_data(data_path)
    
    # E_inst from NeoHooke C10=549.6
    E_inst = 6.0 * 549.6  # 3297.6
    
    # Prepare batched data for relaxation tests
    batch_dt = jnp.stack([targets['relax_0075']['dt'], targets['relax_0100']['dt'], targets['relax_0150']['dt']])
    batch_strain = jnp.stack([targets['relax_0075']['strain'], targets['relax_0100']['strain'], targets['relax_0150']['strain']])
    batch_stress_exp = jnp.stack([targets['relax_0075']['stress'], targets['relax_0100']['stress'], targets['relax_0150']['stress']])
    
    # Initialize trainable raw parameters based on Step 4 Degenerate PRF results
    # Target: A = [3.34e-3, 2.30e-4, 1.39e-5], n=1.0, m=0.0
    # A_raw (softplus(x)*1e-4) -> x = softplus_inv(A_target / 1e-4)
    # n_raw (softplus(x)+1) -> x = -10 (for n=1.0)
    # m_raw -> x = -10 (for m=0.0)
    # s_raw -> [log(0.04), log(0.08), log(0.15)]
    key = jax.random.PRNGKey(42)
    raw_params = {
        'A': jnp.array([33.4, 2.3, 0.139]), 
        'n': jnp.array([-10.0, -10.0, -10.0]),
        'm': jnp.array([-10.0, -10.0, -10.0]),
        's': jnp.array([jnp.log(0.04), jnp.log(0.08), jnp.log(0.15)])
    }
    
    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(raw_params)
    
    @jax.jit
    def step(raw_params, opt_state):
        (loss, params), grads = jax.value_and_grad(compute_loss, has_aux=True)(raw_params, batch_dt, batch_strain, batch_stress_exp, E_inst)
        updates, opt_state = optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)
        return raw_params, opt_state, loss, params
        
    print("Starting Multi-Objective Optimization (JAX vmap over relaxation tests)...")
    epochs = 2000
    for i in range(epochs):
        raw_params, opt_state, loss, params = step(raw_params, opt_state)
        if i % 200 == 0 or i == epochs - 1:
            print(f"Epoch {i:4d} | Loss: {loss:.4f}")
            
    print("\n--- Optimized PRF Parameters ---")
    for net in range(3):
        print(f"Network {net+1}: SRatio={params['s'][net]:.4f}, A={params['A'][net]:.4e}, n={params['n'][net]:.4f}, m={params['m'][net]:.4f}")
        
    # Plotting Results
    output_dir = 'dev_log/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'orange']
    labels = ['0.75% Strain', '1.0% Strain', '1.5% Strain']
    
    # Run optimized model
    opt_sim_stress = jax.vmap(lambda dt, st: simulate_test(params, dt, st, E_inst))(batch_dt, batch_strain)
    
    keys = ['relax_0075', 'relax_0100', 'relax_0150']
    for j in range(3):
        key = keys[j]
        plt.plot(targets[key]['time'], 
                 batch_stress_exp[j], 'o', color=colors[j], markersize=3, alpha=0.5, label=f'Exp: {labels[j]}')
        plt.plot(targets[key]['time'], 
                 opt_sim_stress[j], '-', color=colors[j], linewidth=2, label=f'Sim: {labels[j]}')
                 
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Engineering Stress (MPa)')
    plt.title('Step 6: Optimized PRF Model Response (Multi-Objective, JAX vmap)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'step6_optimized_prf.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved optimized PRF plot to {out_path}")

if __name__ == "__main__":
    main()
