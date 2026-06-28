import jax
import jax.numpy as jnp
from wht_prf.equilibrium import compute_cauchy_stress
from wht_prf.model import compute_deviatoric

def prony_time_integration(mat_data: dict, times: jnp.ndarray, F_history: jnp.ndarray) -> jnp.ndarray:
    """lax.scan을 활용해 전체 시간 단계 동안 Prony Series 점탄성 모델의 Cauchy 응력을 시계열 적분합니다.
    Abaqus의 *VISCOELASTIC, TIME=PRONY 유한 변형률 수식과 동일하게 거동합니다.
    
    Args:
        mat_data (dict): 초탄성 계수 및 prony_series 정보를 담은 PyTree
        times (jnp.ndarray): 시간 배열 (N_steps,)
        F_history (jnp.ndarray): 변형구배 시계열 (N_steps, 3, 3)
        
    Returns:
        jnp.ndarray: 각 시간 단계별 Cauchy 응력 배열 (N_steps, 3, 3)
    """
    he_type = mat_data.get("hyperelastic_type", "YEOH")
    he_params = mat_data.get("hyperelastic_params", jnp.array([]))
    prony_series = mat_data.get("prony_series", [])
    
    num_terms = len(prony_series)
    
    # Prony 계수 텐서 적층
    g_i = jnp.array([term["g_i"] for term in prony_series])
    k_i = jnp.array([term["k_i"] for term in prony_series])
    tau_i = jnp.array([term["tau_i"] for term in prony_series])
    
    # carry 상태: (H_dev 텐서, H_vol 텐서, S0_dev_old, S0_vol_old, t_prev)
    init_H_dev = jnp.zeros((num_terms, 3, 3)) if num_terms > 0 else jnp.zeros((0, 3, 3))
    init_H_vol = jnp.zeros((num_terms, 3, 3)) if num_terms > 0 else jnp.zeros((0, 3, 3))
    init_S0_dev = jnp.zeros((3, 3))
    init_S0_vol = jnp.zeros((3, 3))
    
    init_carry = (init_H_dev, init_H_vol, init_S0_dev, init_S0_vol, 0.0)
    
    def step_fn(carry, input_step):
        H_dev_old, H_vol_old, S0_dev_old, S0_vol_old, t_prev = carry
        F_curr, t_curr = input_step
        
        dt = t_curr - t_prev
        dt_safe = jnp.maximum(dt, 1e-12)
        
        # 1. 순간 초탄성 Cauchy 응력 산출
        sigma0 = compute_cauchy_stress(he_type, F_curr, he_params)
        
        # 2. 역학적 변환 (Abaqus는 객관적 응력률이나 Kirchhoff 기반 연산을 수행)
        # 본 1D 인장/압축 벤치마크에서는 회전이 없으므로 Kirchhoff 응력(tau = J * sigma)의 
        # 편차 및 체적 성분 직접 분리 연산이 Abaqus의 2nd PK 방식과 일치합니다.
        J = jnp.linalg.det(F_curr)
        tau0 = J * sigma0
        tau0_vol = jnp.trace(tau0) / 3.0 * jnp.eye(3)
        tau0_dev = tau0 - tau0_vol
        
        dtau0_dev = tau0_dev - S0_dev_old
        dtau0_vol = tau0_vol - S0_vol_old
        
        new_H_dev_list = []
        new_H_vol_list = []
        
        tau_relax = tau0
        
        for i in range(num_terms):
            x = dt_safe / tau_i[i]
            exp_term = jnp.exp(-x)
            # 수치 안정성을 위해 x가 0에 가까울 때 테일러 전개 근사 적용
            fac = jnp.where(x > 1e-4, (1.0 - exp_term) / x, 1.0 - x/2.0 + x**2/6.0)
            
            # 내부 변수 업데이트 (Q_k: 합성곱 이력 적분 변수)
            # Q_{k,n+1} = exp(-dt/tau_k) * Q_{k,n} + fac * \Delta \tau^0
            Q_dev_new = exp_term * H_dev_old[i] + fac * dtau0_dev
            Q_vol_new = exp_term * H_vol_old[i] + fac * dtau0_vol
            
            # 점탄성 이완 계산
            # tau_relax = tau0 - \sum g_k (tau0_dev - Q_{k, dev}) - \sum k_k (tau0_vol - Q_{k, vol})
            tau_relax = tau_relax - g_i[i] * (tau0_dev - Q_dev_new) - k_i[i] * (tau0_vol - Q_vol_new)
            
            new_H_dev_list.append(Q_dev_new)
            new_H_vol_list.append(Q_vol_new)
            
        new_H_dev = jnp.stack(new_H_dev_list) if num_terms > 0 else init_H_dev
        new_H_vol = jnp.stack(new_H_vol_list) if num_terms > 0 else init_H_vol
        
        # 최종 Cauchy 응력 복원
        sigma_relax = tau_relax / J
        
        # 상태 업데이트
        new_carry = (new_H_dev, new_H_vol, tau0_dev, tau0_vol, t_curr)
        
        return new_carry, sigma_relax
        
    inputs = (F_history, times)
    _, stresses = jax.lax.scan(step_fn, init_carry, inputs)
    return stresses

def compute_prony_step_stress(F_curr: jnp.ndarray, mat_data: dict, dt: float, prev_state: dict) -> jnp.ndarray:
    """1스텝 동안의 Prony Cauchy stress를 계산하는 도우미 함수."""
    he_type = mat_data.get("hyperelastic_type", "YEOH")
    he_params = mat_data.get("hyperelastic_params", jnp.array([]))
    prony_series = mat_data.get("prony_series", [])
    
    num_terms = len(prony_series)
    g_i = jnp.array([term["g_i"] for term in prony_series])
    k_i = jnp.array([term["k_i"] for term in prony_series])
    tau_i = jnp.array([term["tau_i"] for term in prony_series])
    
    H_dev_old = prev_state["H_dev"]
    H_vol_old = prev_state["H_vol"]
    S0_dev_old = prev_state["S0_dev"]
    S0_vol_old = prev_state["S0_vol"]
    
    dt_safe = jnp.maximum(dt, 1e-12)
    
    sigma0 = compute_cauchy_stress(he_type, F_curr, he_params)
    J = jnp.linalg.det(F_curr)
    tau0 = J * sigma0
    tau0_vol = jnp.trace(tau0) / 3.0 * jnp.eye(3)
    tau0_dev = tau0 - tau0_vol
    
    dtau0_dev = tau0_dev - S0_dev_old
    dtau0_vol = tau0_vol - S0_vol_old
    
    tau_relax = tau0
    
    for i in range(num_terms):
        x = dt_safe / tau_i[i]
        exp_term = jnp.exp(-x)
        fac = jnp.where(x > 1e-4, (1.0 - exp_term) / x, 1.0 - x/2.0 + x**2/6.0)
        
        Q_dev_new = exp_term * H_dev_old[i] + fac * dtau0_dev
        Q_vol_new = exp_term * H_vol_old[i] + fac * dtau0_vol
        
        tau_relax = tau_relax - g_i[i] * (tau0_dev - Q_dev_new) - k_i[i] * (tau0_vol - Q_vol_new)
        
    return tau_relax / J

def update_prony_states(F_curr: jnp.ndarray, mat_data: dict, dt: float, prev_state: dict) -> dict:
    he_type = mat_data.get("hyperelastic_type", "YEOH")
    he_params = mat_data.get("hyperelastic_params", jnp.array([]))
    prony_series = mat_data.get("prony_series", [])
    
    num_terms = len(prony_series)
    g_i = jnp.array([term["g_i"] for term in prony_series])
    k_i = jnp.array([term["k_i"] for term in prony_series])
    tau_i = jnp.array([term["tau_i"] for term in prony_series])
    
    H_dev_old = prev_state["H_dev"]
    H_vol_old = prev_state["H_vol"]
    S0_dev_old = prev_state["S0_dev"]
    S0_vol_old = prev_state["S0_vol"]
    
    dt_safe = jnp.maximum(dt, 1e-12)
    
    sigma0 = compute_cauchy_stress(he_type, F_curr, he_params)
    J = jnp.linalg.det(F_curr)
    tau0 = J * sigma0
    tau0_vol = jnp.trace(tau0) / 3.0 * jnp.eye(3)
    tau0_dev = tau0 - tau0_vol
    
    dtau0_dev = tau0_dev - S0_dev_old
    dtau0_vol = tau0_vol - S0_vol_old
    
    new_H_dev_list = []
    new_H_vol_list = []
    
    for i in range(num_terms):
        x = dt_safe / tau_i[i]
        exp_term = jnp.exp(-x)
        fac = jnp.where(x > 1e-4, (1.0 - exp_term) / x, 1.0 - x/2.0 + x**2/6.0)
        
        Q_dev_new = exp_term * prev_state['H_dev'][i] + fac * dtau0_dev
        Q_vol_new = exp_term * prev_state['H_vol'][i] + fac * dtau0_vol
        
        new_H_dev_list.append(Q_dev_new)
        new_H_vol_list.append(Q_vol_new)
        
    new_H_dev = jnp.stack(new_H_dev_list) if num_terms > 0 else jnp.zeros((0, 3, 3))
    new_H_vol = jnp.stack(new_H_vol_list) if num_terms > 0 else jnp.zeros((0, 3, 3))
    
    return {
        "H_dev": new_H_dev,
        "H_vol": new_H_vol,
        "S0_dev": tau0_dev,
        "S0_vol": tau0_vol
    }

def solve_prony_displacement_control(mat_data: dict, F11: float, dt: float, prev_state: dict) -> tuple:
    tol = 1e-6
    max_iter = 25
    init_F22 = 1.0 / jnp.sqrt(F11)
    
    def get_residual(F22_val):
        F_matrix = jnp.diag(jnp.array([F11, F22_val, F22_val]))
        stress = compute_prony_step_stress(F_matrix, mat_data, dt, prev_state)
        return stress[1, 1]
        
    get_jacobian = jax.grad(get_residual)
    init_val = (init_F22, 999.0, 0)
    
    def cond_fun(state):
        F22, err, it = state
        return (err > tol) & (it < max_iter)
        
    def body_fun(state):
        F22, _, it = state
        res = get_residual(F22)
        jac = get_jacobian(F22)
        jac_safe = jnp.where(jnp.abs(jac) < 1e-12, 1e-12, jac)
        F22_new = F22 - res / jac_safe
        return (F22_new, jnp.abs(res), it + 1)
        
    final_F22, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)
    F_final = jnp.diag(jnp.array([F11, final_F22, final_F22]))
    stress_final = compute_prony_step_stress(F_final, mat_data, dt, prev_state)
    
    return F_final, stress_final

def simulate_prony_test(mat_data: dict, times: jnp.ndarray, target_strains: jnp.ndarray) -> jnp.ndarray:
    """1축 변위 제어(인장/압축/완화) 동안의 Cauchy 응력을 시계열 계산합니다."""
    
    num_terms = len(mat_data.get("prony_series", []))
    init_state = {
        "H_dev": jnp.zeros((num_terms, 3, 3)) if num_terms > 0 else jnp.zeros((0, 3, 3)),
        "H_vol": jnp.zeros((num_terms, 3, 3)) if num_terms > 0 else jnp.zeros((0, 3, 3)),
        "S0_dev": jnp.zeros((3, 3)),
        "S0_vol": jnp.zeros((3, 3))
    }
    
    def step_fn(carry, input_step):
        prev_state, t_prev = carry
        t_curr, eng_strain = input_step
        
        dt = t_curr - t_prev
        dt_safe = jnp.maximum(dt, 1e-12)
        F11 = 1.0 + eng_strain
        
        F_final, stress_final = solve_prony_displacement_control(mat_data, F11, dt_safe, prev_state)
        state_new = update_prony_states(F_final, mat_data, dt_safe, prev_state)
        
        new_carry = (state_new, t_curr)
        return new_carry, stress_final
        
    inputs = (times, target_strains)
    init_carry = (init_state, 0.0)
    
    _, stresses = jax.lax.scan(step_fn, init_carry, inputs)
    return stresses
