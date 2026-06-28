import pytest
import jax.numpy as jnp
from wht_prf.io_manager import ExperimentalDataLoader
from wht_prf.model import prf_time_integration
from wht_prf.optimization import fit_prf_parameters

def test_deterministic_fitting_optax():
    """복합 하중(인장-이완) 스마트 테스팅 시나리오 하에서 Optax 기반 계수 피팅 수렴성 검증"""
    # 1. 가상 타겟 물성 설정 (Ground Truth)
    true_mat_data = {
        "hyperelastic_type": "YEOH",
        "hyperelastic_params": jnp.array([0.6, -0.04, 0.015, 10.0]), # C10, C20, C30, D1
        "networks": [
            {
                "stiffness_ratio": 0.35,
                "creep_params": jnp.array([1.0e-5, 3.2, 0.0]) # A, n, m
            }
        ]
    }
    
    # 2. 복합 하중 이력 정의
    # 0~1초: 1축 인장 (e = 0.0 -> 0.12)
    # 1~3초: 응력 이완 (e = 0.12 유지)
    N_steps = 15
    times = jnp.linspace(0.0, 3.0, N_steps)
    
    strains = jnp.zeros(N_steps)
    strains = strains.at[0:5].set(jnp.linspace(0.0, 0.12, 5))
    strains = strains.at[5:].set(0.12)
    
    # 3. Ground Truth 변형구배 및 응답 응력 생성
    F_history = []
    for eps in strains:
        lam = 1.0 + eps
        F_step = jnp.diag(jnp.array([lam, 1.0/jnp.sqrt(lam), 1.0/jnp.sqrt(lam)]))
        F_history.append(F_step)
    F_history = jnp.stack(F_history)
    
    # 타겟 응력 계산
    target_stresses = prf_time_integration(true_mat_data, times, F_history)
    # 1축 응력 차이 (sigma_11 - sigma_22)를 피팅 타겟으로 사용
    target_diff = target_stresses[:, 0, 0] - target_stresses[:, 1, 1]
    
    # 4. 초기 임의의 물성값에서 최적화 수행
    init_params = {
        "hyperelastic_params": jnp.array([0.4, -0.01, 0.005]), # D1은 10.0 고정
        "stiffness_ratio": 0.2,
        "creep_params": jnp.array([5.0e-5, 2.0]) # m은 0.0 고정
    }
    
    # 피팅 진행 (최대 에포크 100회)
    best_params, loss_history = fit_prf_parameters(
        init_params, times, F_history, target_diff, max_epochs=150
    )
    
    # 최적화 후 오차가 초기 오차보다 대폭 줄어들었는지 검증 (Loss 수렴성)
    assert loss_history[-1] < loss_history[0]
    assert loss_history[-1] < 1e-2 # 낮은 MSE 달성 검증
