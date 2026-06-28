import pytest
import jax.numpy as jnp
from wht_prf.equilibrium import (
    neo_hookean_energy,
    yeoh_energy,
    arruda_boyce_energy,
    compute_cauchy_stress
)

def test_neo_hookean_cauchy_stress():
    """Neo-Hookean 모델의 자동미분 Cauchy 응력이 1축 인장 해석해와 일치하는지 검증"""
    # Neo-Hookean 파라미터 (C10 = 0.5, D1 = 10.0)
    # W = C10 * (I1_bar - 3) + (J-1)^2 / D1
    params = jnp.array([0.5, 10.0])
    
    # 1축 인장 F (비압축성 가정: lambda = 1.2)
    lam = 1.2
    F = jnp.diag(jnp.array([lam, 1.0/jnp.sqrt(lam), 1.0/jnp.sqrt(lam)]))
    
    # Cauchy 응력 계산
    stress = compute_cauchy_stress(neo_hookean_energy, F, params)
    
    # 1축 인장 해석해 (자유 경계조건 하의 sigma_11 - sigma_22):
    # J = 1.0 이므로, sigma_11 - sigma_22 = 2 * C10 * (lam^2 - 1/lam)
    expected_sigma_diff = 2.0 * 0.5 * (lam**2 - 1.0/lam)
    
    # 1축 인장 자유경계 조건에 따른 응력 차이 비교
    assert jnp.allclose(stress[0, 0] - stress[1, 1], expected_sigma_diff, atol=1e-5)
    # 다른 전단 응력 성분은 0이어야 함
    assert jnp.allclose(stress[0, 1], 0.0)

def test_yeoh_stress_update():
    """Yeoh 초탄성 모델 (3차 Polynomial) 에너지 밀도와 응력 계산 검증"""
    # Yeoh 파라미터 (C10 = 0.5, C20 = -0.05, C30 = 0.01, D1 = 20.0)
    # W = C10*(I1_bar-3) + C20*(I1_bar-3)^2 + C30*(I1_bar-3)^3 + (J-1)^2/D1
    params = jnp.array([0.5, -0.05, 0.01, 20.0])
    
    lam = 1.15
    F = jnp.diag(jnp.array([lam, 1.0/jnp.sqrt(lam), 1.0/jnp.sqrt(lam)]))
    
    # Cauchy 응력 계산
    stress = compute_cauchy_stress(yeoh_energy, F, params)
    
    # 계산된 응력이 유한한 값을 가지며 수치적으로 안정한지 확인
    assert not jnp.isnan(stress).any()
    assert stress[0, 0] > 0.0  # 인장 응력 발생 확인

def test_arruda_boyce_stress_update():
    """Arruda-Boyce 초탄성 모델 에너지 밀도와 응력 계산 검증"""
    # Arruda-Boyce 파라미터 (mu = 1.0, lambda_L = 7.0, D1 = 10.0)
    params = jnp.array([1.0, 7.0, 10.0])
    
    lam = 1.1
    F = jnp.diag(jnp.array([lam, 1.0/jnp.sqrt(lam), 1.0/jnp.sqrt(lam)]))
    
    # Cauchy 응력 계산
    stress = compute_cauchy_stress(arruda_boyce_energy, F, params)
    
    assert not jnp.isnan(stress).any()
    assert stress[0, 0] > 0.0
