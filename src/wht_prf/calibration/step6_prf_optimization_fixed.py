"""
WHTOOLs MATCALIB 2026
PRF Parameter Optimization using JAX.
"""
import os
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

def load_data():
    base_dir = r"G:\Simulia_Video_Contents\Material Calibration\Calibration of a PRF Material Model for Polypropylene\PPC3TF2TestData"
    data_files = {
        'relax_0050': 'relax_data_0050.txt',
        'relax_0075': 'relax_data_0075.txt',
        'relax_0100': 'relax_data_0100.txt',
        'relax_0150': 'relax_data_0150.txt'
    }
    
    targets = {}
    for key, filename in data_files.items():
        path = os.path.join(base_dir, filename)
        df = pd.read_csv(path, sep=',', header=1)
        time = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        strain = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        stress = pd.to_numeric(df.iloc[:, 2], errors='coerce').values
        
        mask = ~np.isnan(time) & ~np.isnan(strain) & ~np.isnan(stress)
        time = time[mask]
        strain = strain[mask]
        stress = stress[mask]
        
        dt = np.diff(time)
        dt = np.where(dt <= 0, 1e-6, dt)
        
        targets[key] = {
            'time': time,
            'strain': strain,
            'stress': stress,
            'dt': dt
        }
    return targets

def prf_1d_step_substep(state, inputs, params, E_inst):
    e_cr_old = state
    dt_total, strain_tot = inputs
    
    A = params['A']
    n = params['n']
    m = params['m']
    s = params['s']
    E_i = s * E_inst
    
    num_substeps = 100
    dt_sub = dt_total / num_substeps
    
    def substep_fn(i, e_cr_curr):
        sig_i = E_i * (strain_tot - e_cr_curr)
        q = jnp.abs(sig_i)
        e_cr_eff = jnp.abs(e_cr_curr) + 1e-8
        
        base = jnp.clip(A * (q**n) * (((1.0 + m) * e_cr_eff)**m), 0.0, 1e10)
        om1 = 1.0 / (1.0 + m)
        e_dot_mag = jnp.clip(base ** om1, 0.0, 1e5)
        
        e_dot = e_dot_mag * jnp.sign(sig_i)
        return e_cr_curr + e_dot * dt_sub

    e_cr_new = jax.lax.fori_loop(0, num_substeps, substep_fn, e_cr_old)
    
    s_inf = 1.0 - jnp.sum(s)
    sig_inf = s_inf * E_inst * strain_tot
    sig_tot = sig_inf + jnp.sum(E_i * (strain_tot - e_cr_new))
    
    return e_cr_new, sig_tot

def simulate_test(params, dt_arr, strain_arr, E_inst):
    init_e_cr = jnp.zeros(3, dtype=jnp.float64)
    inputs = (dt_arr, strain_arr[1:])
    scan_fn = lambda state, x: prf_1d_step_substep(state, x, params, E_inst)
    _, stress_sim = jax.lax.scan(scan_fn, init_e_cr, inputs)
    sig0 = E_inst * strain_arr[0]
    stress_sim = jnp.concatenate([jnp.array([sig0]), stress_sim])
    return stress_sim

@partial(jax.jit, static_argnames=['E_inst'])
def compute_loss(raw_params, batch_dt, batch_strain, batch_stress_exp, batch_mask, E_inst):
    params = {
        'A': jax.nn.softplus(raw_params['A']) * 1e-4, 
        'n': jax.nn.softplus(raw_params['n']) + 1.0,  
        'm': -jax.nn.sigmoid(raw_params['m']) * 0.99,      
        's': jax.nn.softmax(raw_params['s']) * 0.9    
    }
    sim_fn = lambda dt, st: simulate_test(params, dt, st, E_inst)
    batch_stress_sim = jax.vmap(sim_fn)(batch_dt, batch_strain)
    
    # Masked MSE
    sq_err = (batch_stress_sim - batch_stress_exp)**2
    loss_relax = jnp.sum(sq_err * batch_mask) / jnp.sum(batch_mask)
    return loss_relax, params

def main():
    targets = load_data()
    E_inst = 2316.39
    keys = ['relax_0050', 'relax_0075', 'relax_0100', 'relax_0150']
    
    max_len = max([len(t['time']) for t in targets.values()])
    
    # Pad sequences to max_len
    for k in keys:
        L = len(targets[k]['time'])
        pad_len = max_len - L
        if pad_len > 0:
            last_strain = targets[k]['strain'][-1]
            last_time = targets[k]['time'][-1]
            
            pad_time = last_time + np.arange(1, pad_len+1) * 0.1
            pad_strain = np.full(pad_len, last_strain)
            pad_stress = np.zeros(pad_len)
            pad_dt = np.full(pad_len, 0.1)
            mask = np.concatenate([np.ones(L), np.zeros(pad_len)])
            
            targets[k]['time'] = np.concatenate([targets[k]['time'], pad_time])
            targets[k]['strain'] = np.concatenate([targets[k]['strain'], pad_strain])
            targets[k]['stress'] = np.concatenate([targets[k]['stress'], pad_stress])
            # dt array is length-1, but simulate_test needs len-1 dt.
            # wait, simulate_test scan takes dt_arr and strain_arr[1:]
            # so dt_arr must be length max_len - 1
            dt_L = len(targets[k]['dt'])
            targets[k]['dt'] = np.concatenate([targets[k]['dt'], np.full(pad_len, 0.1)])
            targets[k]['mask'] = mask
        else:
            targets[k]['mask'] = np.ones(L)
            
    batch_dt = jnp.stack([targets[k]['dt'][:max_len-1] for k in keys])
    batch_strain = jnp.stack([targets[k]['strain'] for k in keys])
    batch_stress_exp = jnp.stack([targets[k]['stress'] for k in keys])
    batch_mask = jnp.stack([targets[k]['mask'] for k in keys])
    
    raw_params = {
        'A': jnp.array([33.4, 2.3, -1.96]),
        'n': jnp.array([-10.0, -10.0, -10.0]), 
        'm': jnp.array([-10.0, -10.0, -10.0]), 
        's': jnp.array([jnp.log(0.04), jnp.log(0.08), jnp.log(0.15)])
    }
    
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(raw_params)
    
    @jax.jit
    def step(raw_params, opt_state):
        (loss, params), grads = jax.value_and_grad(compute_loss, has_aux=True)(raw_params, batch_dt, batch_strain, batch_stress_exp, batch_mask, E_inst)
        updates, opt_state = optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)
        return raw_params, opt_state, loss, params

    print("Starting optimization...")
    best_loss = 1e9
    best_params = None
    
    for epoch in range(1500):
        raw_params, opt_state, loss, params = step(raw_params, opt_state)
        if loss < best_loss:
            best_loss = loss
            best_params = params
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
    print(f"Final Loss: {best_loss:.4f}")
    print(f"Best Params: A: {best_params['A']}, n: {best_params['n']}, m: {best_params['m']}, s: {best_params['s']}")
    
    # Run best model
    sim_fn = lambda dt, st: simulate_test(best_params, dt, st, E_inst)
    opt_sim_stress = jax.vmap(sim_fn)(batch_dt, batch_strain)
    
    plt.figure(figsize=(10, 6), dpi=150)
    colors = ['r', 'b', 'g', 'orange']
    labels = ['0.5% Strain', '0.75% Strain', '1.0% Strain', '1.5% Strain']
    
    for j in range(4):
        key = keys[j]
        L = int(np.sum(targets[key]['mask']))
        plt.plot(targets[key]['time'][:L], targets[key]['stress'][:L], 'o', color=colors[j], markersize=3, alpha=0.5, label=f'Exp: {labels[j]}')
        plt.plot(targets[key]['time'][:L], opt_sim_stress[j][:L], '-', color=colors[j], linewidth=2, label=f'Sim: {labels[j]}')
               
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Engineering Stress (MPa)')
    plt.title('Step 6: Optimized PRF Model Response (JAX Masked Full Time)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_dir = os.path.dirname(__file__)
    plt.savefig(os.path.join(out_dir, '..', '..', '..', 'dev_log', 'resources', 'step6_optimized_prf_fixed.png'))
    print("Saved dev_log/resources/step6_optimized_prf_fixed.png")
    
if __name__ == '__main__':
    main()
