import jax
import jax.numpy as jnp
from wht_prf.equilibrium import (
    neo_hookean_energy,
    yeoh_energy,
    arruda_boyce_energy,
    compute_cauchy_stress
)
from wht_prf.nonequilibrium import power_law_creep_rate, expm_update_f_creep
from wht_prf.kinematics import compute_det_3x3

def get_energy_fn(he_type: str):
    """지정된 초탄성 에너지 밀도 함수를 반환합니다."""
    he_type_upper = he_type.upper()
    if "YEOH" in he_type_upper:
        return yeoh_energy
    elif "NEO" in he_type_upper or "NEO_HOOKEAN" in he_type_upper:
        return neo_hookean_energy
    elif "ARRUDA" in he_type_upper or "ARRUDA_BOYCE" in he_type_upper:
        return arruda_boyce_energy
    else:
        raise ValueError(f"Unsupported hyperelastic type: {he_type}")

def compute_deviatoric(tensor: jnp.ndarray) -> jnp.ndarray:
    """3x3 텐서의 편차(Deviatoric) 성분 dev(T) = T - 1/3 * tr(T) * I 를 계산합니다."""
    tr = jnp.trace(tensor)
    return tensor - (1.0 / 3.0) * tr * jnp.eye(3)

def compute_von_mises(s_dev: jnp.ndarray) -> float:
    """편차 응력 텐서 s_dev로부터 Von Mises 등가 응력 sqrt(3/2 * s_dev : s_dev)를 계산합니다.
    0 부근에서 JAX Autograd 역전파 미분 폭주(NaN)를 방지하기 위해 제곱근 내부에 미소한 에프실론을 보정합니다.
    """
    contraction = jnp.sum(s_dev * s_dev)
    return jnp.sqrt(1.5 * contraction + 1e-12)

def prf_time_integration(mat_data: dict, times: jnp.ndarray, F_history: jnp.ndarray) -> jnp.ndarray:
    """lax.scan을 활용해 전체 시간 단계 동안 PRF 병렬 네트워크 모델의 Cauchy 응력을 시계열 적분합니다.
    
    Args:
        mat_data (dict): 초탄성/점탄성 계수 및 네트워크 정보를 담은 PyTree
        times (jnp.ndarray): 시간 배열 (N_steps,)
        F_history (jnp.ndarray): 변형구배 시계열 (N_steps, 3, 3)
        
    Returns:
        jnp.ndarray: 각 시간 단계별 Cauchy 응력 배열 (N_steps, 3, 3)
    """
    he_type = mat_data.get("hyperelastic_type", "YEOH")
    he_params = mat_data.get("hyperelastic_params", jnp.array([]))
    networks = mat_data.get("networks", [])
    
    num_nets = len(networks)
    
    # 텐서 형태로 파라미터 적층
    ratios = jnp.array([net.get("stiffness_ratio", 0.0) for net in networks])
    total_viscous_ratio = jnp.sum(ratios)
    eq_ratio = 1.0 - total_viscous_ratio
    
    # 각 네트워크의 크립 파라미터 (A, n, m) 적층 (m 생략 시 0.0 보정)
    creep_list = []
    for net in networks:
        cp = net.get("creep_params", jnp.array([]))
        if cp.shape[0] >= 3:
            cp_3 = cp[:3]
        elif cp.shape[0] == 2:
            cp_3 = jnp.array([cp[0], cp[1], 0.0])
        elif cp.shape[0] == 1:
            cp_3 = jnp.array([cp[0], 0.0, 0.0])
        else:
            cp_3 = jnp.array([0.0, 0.0, 0.0])
        creep_list.append(cp_3)
    
    creep_params_stacked = jnp.stack(creep_list) if num_nets > 0 else jnp.zeros((0, 3))
    
    # carry 상태: (N_networks, 3, 3) F_cr 텐서
    init_F_cr = jnp.stack([jnp.eye(3) for _ in range(num_nets)]) if num_nets > 0 else jnp.zeros((0, 3, 3))
    init_carry = (init_F_cr, 0.0)
    
    def step_fn(carry, input_step):
        F_cr_tensor, t_prev = carry
        F_curr, t_curr = input_step
        
        dt = t_curr - t_prev
        dt_safe = jnp.maximum(dt, 1e-10)
        
        # 1. 평형 네트워크 Cauchy 응력 산출
        sigma_eq = compute_cauchy_stress(he_type, F_curr, he_params)
        sigma_total = eq_ratio * sigma_eq
        
        new_F_cr_list = []
        
        # 2. 각 비평형 네트워크 루프
        for i in range(num_nets):
            ratio = ratios[i]
            creep_params = creep_params_stacked[i]
            F_cr_old = F_cr_tensor[i]
            
            # 내부 서브스태핑(Sub-stepping) 루프를 돌려 과도 크립 오버슈트 방지 및 수치 강건성 확보
            F_cr_temp = F_cr_old
            t_temp = t_prev
            N_sub = 10
            dt_sub = dt_safe / N_sub
            
            for _ in range(N_sub):
                t_temp = t_temp + dt_sub
                F_cr_inv_sub = jnp.diag(1.0 / (jnp.diagonal(F_cr_temp) + 1e-12))
                F_e_sub = jnp.dot(F_curr, F_cr_inv_sub)
                sigma_v_sub = compute_cauchy_stress(he_type, F_e_sub, he_params)
                
                S_dev_sub = compute_deviatoric(sigma_v_sub)
                sigma_eff_sub = compute_von_mises(S_dev_sub)
                
                dep_cr_dt = power_law_creep_rate(sigma_eff_sub, t_temp, creep_params)
                direction = 1.5 * S_dev_sub / jnp.sqrt(sigma_eff_sub**2 + 1e-12)
                D_cr_dt = jnp.clip(dep_cr_dt * direction * dt_sub, -0.05, 0.05)
                
                exp_D_dt = jax.scipy.linalg.expm(D_cr_dt)
                F_cr_temp = jnp.dot(exp_D_dt, F_cr_temp)
                
            F_cr_new = F_cr_temp
            F_cr_inv_final = jnp.diag(1.0 / (jnp.diagonal(F_cr_new) + 1e-12))
            F_e_final = jnp.dot(F_curr, F_cr_inv_final)
            sigma_v_i = compute_cauchy_stress(he_type, F_e_final, he_params)
            sigma_total = sigma_total + ratio * sigma_v_i
            new_F_cr_list.append(F_cr_new)
            
        new_F_cr_tensor = jnp.stack(new_F_cr_list) if num_nets > 0 else jnp.zeros((0, 3, 3))
        new_carry = (new_F_cr_tensor, t_curr)
        return new_carry, sigma_total
        
    # lax.scan 실행을 위해 인풋을 스택 형태로 구성
    inputs = (F_history, times)
    
    # scan 루프 전진
    _, stresses = jax.lax.scan(step_fn, init_carry, inputs)
    return stresses
