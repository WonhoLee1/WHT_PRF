import pytest
import jax.numpy as jnp
from wht_prf.solver import solve_uniaxial_step

def test_uniaxial_stress_solver_1d_displacement_control():
    """1D 변위 제어(F11 주어짐) 하에서 가로 방향 수축(F22, F33)을 수렴 계산하여 측면 응력 0.0을 만족하는지 검증"""
    mat_data = {
        "hyperelastic_type": "YEOH",
        "hyperelastic_params": jnp.array([0.5, -0.05, 0.01, 10.0]),
        "networks": [] # 순수 초탄성 네트워크만 테스트
    }
    
    # 변위 제어 입력: F11 = 1.15, control_flag = 0.0 (변위 제어), target_value = 0.0 (미사용)
    # 초기 상태 (t=0): F = eye(3), F_cr = eye(3)
    # 다음 스텝 (t=1): dt = 1.0, F11 = 1.15
    F11_step = 1.15
    control_flag = 0.0 # 0: 변위 제어
    target_value = 0.0
    
    # 이전 상태 PyTree
    prev_state = {
        "F_cr": [jnp.eye(3)]
    }
    
    # 1스텝 해석 진행
    F_new, stress_new, state_new = solve_uniaxial_step(
        mat_data, F11_step, control_flag, target_value, 1.0, prev_state
    )
    
    # F11이 입력한 값과 일치해야 함
    assert jnp.allclose(F_new[0, 0], F11_step)
    # 측면 응력(sigma_22, sigma_33)은 수치 허용 오차 내에서 0이어야 함
    assert jnp.allclose(stress_new[1, 1], 0.0, atol=1e-5)
    assert jnp.allclose(stress_new[2, 2], 0.0, atol=1e-5)
    # 단축 응력(sigma_11)은 팽창했으므로 양수여야 함
    assert stress_new[0, 0] > 0.0

def test_uniaxial_stress_solver_2d_load_control():
    """2D 하중 제어(목표 응력 sigma_11 주어짐) 하에서 축방향 인장(F11)과 가로 수축(F22, F33)을 자동 수렴하여 경계조건을 만족하는지 검증"""
    mat_data = {
        "hyperelastic_type": "YEOH",
        "hyperelastic_params": jnp.array([0.5, -0.05, 0.01, 10.0]),
        "networks": []
    }
    
    # 하중 제어 입력: control_flag = 1.0 (하중 제어), target_value = 0.2 (목표 Cauchy Stress sigma_11)
    control_flag = 1.0
    target_stress = 0.2
    
    # 이전 상태
    prev_state = {
        "F_cr": [jnp.eye(3)]
    }
    
    # F11_guess = 1.05 (뉴턴-랩슨 시작점 힌트)
    F_new, stress_new, state_new = solve_uniaxial_step(
        mat_data, 1.05, control_flag, target_stress, 1.0, prev_state
    )
    
    # 1축 응력(sigma_11)이 목표 값 0.2에 완벽히 피팅되었는지 확인
    assert jnp.allclose(stress_new[0, 0], target_stress, atol=1e-5)
    # 측면 응력은 여전히 0이어야 함
    assert jnp.allclose(stress_new[1, 1], 0.0, atol=1e-5)
    # 축방향으로 인장(F11 > 1.0)되어 수렴하였는지 확인
    assert F_new[0, 0] > 1.0
