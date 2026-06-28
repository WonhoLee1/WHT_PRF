import pytest
import jax
import jax.numpy as jnp
from wht_prf.nonequilibrium import (
    power_law_creep_rate,
    expm_update_f_creep,
    CreepNeuralNetwork
)
from wht_prf.model import prf_time_integration

def test_expm_incompressibility():
    """행렬 지수 expm을 이용한 크립 변형률 업데이트가 체적 비압축성(det(F_cr) == 1.0)을 보존하는지 검증"""
    # Trace가 0인 임의의 3x3 대칭 편차 속도 구배 D_cr
    D_cr = jnp.array([
        [0.05, 0.0, 0.0],
        [0.0, -0.025, 0.0],
        [0.0, 0.0, -0.025]
    ])
    dt = 0.5
    F_cr_old = jnp.eye(3)
    
    # expm 기반 업데이트
    F_cr_new = expm_update_f_creep(F_cr_old, D_cr, dt)
    det_F_cr = jnp.linalg.det(F_cr_new)
    
    # det(F_cr)은 1.0으로 완벽 보존되어야 함
    assert jnp.allclose(det_F_cr, 1.0, atol=1e-6)

def test_neural_network_flow():
    """Flax CreepNeuralNetwork가 NaN 없이 타당한 크립 속도를 출력하는지 검증"""
    # Flax 초기화
    key = jax.random.PRNGKey(42)
    model = CreepNeuralNetwork(features=[8, 1])
    
    # 입력 데이터: [I1, I2, 누적크립변형률] = [3.2, 3.2, 0.05]
    x_test = jnp.array([[3.2, 3.2, 0.05]])
    params = model.init(key, x_test)
    
    # 예측 수행
    y_pred = model.apply(params, x_test)
    
    # 유효한 스칼라 속도 예측값인지 확인
    assert not jnp.isnan(y_pred).any()
    assert y_pred.shape == (1, 1)

def test_prf_time_integration():
    """lax.scan을 이용한 평형+비평형 복합 네트워크의 시계열 적분 루프 검증"""
    # 하이퍼파라미터 정의
    mat_data = {
        "hyperelastic_type": "YEOH",
        "hyperelastic_params": jnp.array([0.5, -0.05, 0.01, 10.0]), # 평형 네트워크 (C10, C20, C30, D1)
        "networks": [
            {
                "stiffness_ratio": 0.4, # 비평형 네트워크 1의 비율
                "creep_params": jnp.array([1.5e-5, 3.0, 0.0]) # Power-Law (A, n, m)
            }
        ]
    }
    
    # 1축 변형구배 시계열 F_history (시간 0초부터 2.0초까지, e_max = 0.1)
    N_steps = 10
    times = jnp.linspace(0.0, 2.0, N_steps)
    strains = jnp.linspace(0.0, 0.1, N_steps)
    
    # 각 스텝별 변형구배 3x3 텐서 생성 (비압축성 가정 하의 1축 인장 입력)
    F_history = []
    for eps in strains:
        lam = 1.0 + eps
        F_step = jnp.diag(jnp.array([lam, 1.0/jnp.sqrt(lam), 1.0/jnp.sqrt(lam)]))
        F_history.append(F_step)
    F_history = jnp.stack(F_history)
    
    # 시계열 전진 해석 실행
    stresses = prf_time_integration(mat_data, times, F_history)
    
    # 시간 스텝수와 응력 텐서 모양 일치 확인
    assert stresses.shape == (N_steps, 3, 3)
    assert jnp.allclose(stresses[0, 0, 0], 0.0, atol=1e-8)  # 초기 응력은 0이어야 함
    assert stresses[-1, 0, 0] > 0.0 # 최종 인장 응력 발생 확인
