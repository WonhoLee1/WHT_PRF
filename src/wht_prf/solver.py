import jax
import jax.numpy as jnp
from wht_prf.model import get_energy_fn, compute_deviatoric, compute_von_mises
from wht_prf.equilibrium import compute_cauchy_stress, compute_hyperelastic_state
from wht_prf.nonequilibrium import compute_creep_rate, expm_update_f_creep
from wht_prf.kinematics import compute_det_3x3

def compute_step_stress(F_curr: jnp.ndarray, mat_data: dict, dt: float, prev_state: dict) -> jnp.ndarray:
    """1스텝 동안의 Cauchy stress를 계산하는 도우미 함수. 
    이전 크립 상태(prev_state)를 기반으로 현재 F_curr 조건에서의 응력을 계산합니다.
    """
    he_type = mat_data.get("hyperelastic_type", "YEOH")
    he_params = mat_data.get("hyperelastic_params", jnp.array([]))
    networks = mat_data.get("networks", [])
    
    num_nets = len(networks)
    ratios = jnp.array([net.get("stiffness_ratio", 0.0) for net in networks])
    total_viscous_ratio = jnp.sum(ratios)
    eq_ratio = 1.0 - total_viscous_ratio
    
    # 1. 평형 성분 응력
    mullins_params = mat_data.get("mullins_params", None)
    U_dev_max = prev_state.get("U_dev_max", 0.0)
    sigma_eq, _ = compute_hyperelastic_state(he_type, F_curr, he_params, mullins_params, U_dev_max)
    sigma_total = eq_ratio * sigma_eq
    
    # 2. 비평형 성분 응력 합산
    F_cr_tensor = prev_state["F_cr"]
    
    for i in range(num_nets):
        ratio = ratios[i]
        F_cr_old = F_cr_tensor[i]
        F_cr_inv = jnp.diag(1.0 / (jnp.diagonal(F_cr_old) + 1e-12))
        F_e = jnp.dot(F_curr, F_cr_inv)
        
        sigma_v_i = compute_cauchy_stress(he_type, F_e, he_params)
        sigma_total = sigma_total + ratio * sigma_v_i
        
    return sigma_total

def solve_displacement_control(mat_data: dict, F11: float, dt: float, prev_state: dict) -> tuple:
    """1D Newton-Raphson: F11이 고정되었을 때 sigma_22 = 0.0을 만족하는 가로 수축 F22를 탐색합니다."""
    tol = 1e-6
    max_iter = 25
    
    # 1축 등방이므로 F22 초기 추측값은 F11^-0.5 (비압축성 기준 부근)
    init_F22 = 1.0 / jnp.sqrt(F11)
    
    def get_residual(F22_val):
        """1D 잔차: sigma_22 = 0"""
        F_matrix = jnp.diag(jnp.array([F11, F22_val, F22_val]))
        stress = compute_step_stress(F_matrix, mat_data, dt, prev_state)
        return stress[1, 1]
        
    # JAX 자동 미분을 이용한 1D 자코비안(도함수) 함수 정의
    get_jacobian = jax.grad(get_residual)
    
    # while_loop 상태 트리: (F22, error, iter_count)
    init_val = (init_F22, 999.0, 0)
    
    def cond_fun(state):
        F22, err, it = state
        return (err > tol) & (it < max_iter)
        
    def body_fun(state):
        F22, _, it = state
        res = get_residual(F22)
        jac = get_jacobian(F22)
        
        # Newton-Raphson 업데이트: X_new = X_old - R / R'
        # 자코비안이 0에 극도로 가깝거나 singularity가 있는 경우 방지
        jac_safe = jnp.where(jnp.abs(jac) < 1e-12, 1e-12, jac)
        F22_new = F22 - res / jac_safe
        
        err_new = jnp.abs(res)
        return (F22_new, err_new, it + 1)
        
    # JIT 호환 while_loop 구동
    final_F22, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)
    
    # 최종 수렴한 3x3 변형구배 및 응력 계산
    F_final = jnp.diag(jnp.array([F11, final_F22, final_F22]))
    stress_final = compute_step_stress(F_final, mat_data, dt, prev_state)
    
    return F_final, stress_final

def solve_load_control(mat_data: dict, F11_guess: float, target_stress: float, dt: float, prev_state: dict) -> tuple:
    """2D Newton-Raphson: 축방향 하중(target_stress)을 만족하는 [F11, F22] 쌍을 공동 탐색합니다.
    잔차 R = [sigma_11 - target_stress, sigma_22] = [0, 0]
    """
    tol = 1e-6
    max_iter = 25
    
    # X = [F11, F22] 초기 추측값 설정
    init_F11 = F11_guess
    init_F22 = 1.0 / jnp.sqrt(init_F11)
    init_X = jnp.array([init_F11, init_F22])
    
    def get_residual_vector(X):
        """2D 잔차 벡터 리턴"""
        F_matrix = jnp.diag(jnp.array([X[0], X[1], X[1]]))
        stress = compute_step_stress(F_matrix, mat_data, dt, prev_state)
        return jnp.array([stress[0, 0] - target_stress, stress[1, 1]])
        
    # JAX 자동 미분을 이용한 2D 자코비안 행렬 함수 정의
    get_jacobian_matrix = jax.jacobian(get_residual_vector)
    
    # carry: (X_vector, error, iter_count)
    init_val = (init_X, 999.0, 0)
    
    def cond_fun(state):
        X, err, it = state
        return (err > tol) & (it < max_iter)
        
    def body_fun(state):
        X, _, it = state
        res = get_residual_vector(X)
        jac = get_jacobian_matrix(X)
        
        # 2D Newton-Raphson: X_new = X_old - J^-1 * R
        # 역행렬 풀이 (solve)
        dX = jnp.linalg.solve(jac, res)
        X_new = X - dX
        
        err_new = jnp.linalg.norm(res)
        return (X_new, err_new, it + 1)
        
    final_X, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)
    
    F_final = jnp.diag(jnp.array([final_X[0], final_X[1], final_X[1]]))
    stress_final = compute_step_stress(F_final, mat_data, dt, prev_state)
    
    return F_final, stress_final

def update_prf_states(mat_data: dict, F_curr: jnp.ndarray, dt: float, total_time: float, prev_state: dict) -> dict:
    """1스텝 해석 완료 후, 다음 스텝 해석을 위해 비평형 네트워크 크립 변형구배(F_cr) 상태를 갱신합니다.
    """
    he_type = mat_data.get("hyperelastic_type", "YEOH")
    he_params = mat_data.get("hyperelastic_params", jnp.array([]))
    networks = mat_data.get("networks", [])
    
    num_nets = len(networks)
    ratios = jnp.array([net.get("stiffness_ratio", 0.0) for net in networks])
    laws = [net.get("LAW", "TIME") for net in networks]
    
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
    
    new_F_cr = []
    F_cr_tensor = prev_state["F_cr"]
    
    for i in range(num_nets):
        creep_params = creep_params_stacked[i]
        F_cr_old = F_cr_tensor[i]
        
        law = laws[i]
        
        # 내부 서브스태핑(Sub-stepping) 루프 적용
        F_cr_temp = F_cr_old
        N_sub = 10
        dt_sub = dt / N_sub
        t_temp = total_time
        
        for _ in range(N_sub):
            t_temp = t_temp + dt_sub
            F_cr_inv_sub = jnp.diag(1.0 / (jnp.diagonal(F_cr_temp) + 1e-12))
            F_e_sub = jnp.dot(F_curr, F_cr_inv_sub)
            sigma_v_sub = compute_cauchy_stress(he_type, F_e_sub, he_params)
            
            S_dev_sub = compute_deviatoric(sigma_v_sub)
            sigma_eff_sub = compute_von_mises(S_dev_sub)
            
            dep_cr_dt = compute_creep_rate(law, sigma_eff_sub, t_temp, F_cr_temp, creep_params)
            direction = 1.5 * S_dev_sub / jnp.sqrt(sigma_eff_sub**2 + 1e-12)
            D_cr_dt = jnp.clip(dep_cr_dt * direction * dt_sub, -0.05, 0.05)
            
            exp_D_dt = jax.scipy.linalg.expm(D_cr_dt)
            F_cr_temp = jnp.dot(exp_D_dt, F_cr_temp)
            
        F_cr_new = F_cr_temp
        new_F_cr.append(F_cr_new)
        
    new_F_cr_tensor = jnp.stack(new_F_cr) if num_nets > 0 else jnp.zeros((0, 3, 3))
    
    # Update U_dev_max
    mullins_params = mat_data.get("mullins_params", None)
    U_dev_max = prev_state.get("U_dev_max", 0.0)
    _, U_dev_max_new = compute_hyperelastic_state(he_type, F_curr, he_params, mullins_params, U_dev_max)
    
    return {"F_cr": new_F_cr_tensor, "U_dev_max": U_dev_max_new}

def solve_uniaxial_step(mat_data: dict, input_val: float, control_flag: float, 
                        target_value: float, dt: float, total_time: float, prev_state: dict) -> tuple:
    """jax.lax.cond를 활용해 변위 제어(flag=0.0)와 하중 제어(flag=1.0)를 
    JIT 브레이크 없이 선택하여 해결하고 갱신된 상태를 반환합니다.
    
    Args:
        mat_data (dict): 물성 PyTree
        input_val (float): F11 주어짐 (변위 제어 시 실제 값, 하중 제어 시 초기 힌트)
        control_flag (float): 0.0 이면 변위 제어, 1.0 이면 하중 제어
        target_value (float): 하중 제어 시의 목표 Cauchy Stress (sigma_11)
        dt (float): 시간 증분
        total_time (float): 스텝 시작 전 누적 시간
        prev_state (dict): 이전 스텝 상태 ("F_cr" 리스트)
        
    Returns:
        tuple: (F_final, stress_final, state_new)
    """
    
    # 1. 제어 플래그에 따른 분기 수행
    # mat_data는 JAX 컴파일에 맞지 않는 텍스트 키가 있으므로 클로저로 직접 캡처합니다.
    # 양쪽 분기 함수의 입력 operand 구조는 완벽히 일치해야 합니다.
    def run_disp_control(operand):
        # operand = (input_val, target_value, dt, prev_state)
        f11, _, time_d, p_st = operand
        return solve_displacement_control(mat_data, f11, time_d, p_st)
        
    def run_load_control(operand):
        # operand = (input_guess, target_stress, dt, prev_state)
        f11_guess, t_stress, time_d, p_st = operand
        return solve_load_control(mat_data, f11_guess, t_stress, time_d, p_st)

    # 1D/2D 분기 실행 (operands에는 오직 JAX 호환 수치 형식만 남김)
    F_final, stress_final = jax.lax.cond(
        control_flag == 0.0,
        run_disp_control,
        run_load_control,
        (input_val, target_value, dt, prev_state)
    )
    
    # 2. 비평형 상태 업데이트 진행
    state_new = update_prf_states(mat_data, F_final, dt, total_time, prev_state)
    
    return F_final, stress_final, state_new
