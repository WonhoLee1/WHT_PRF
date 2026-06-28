import pytest
import jax.numpy as jnp
from wht_prf.kinematics import (
    compute_right_cauchy_green,
    compute_invariants,
    decompose_isochoric
)

def test_right_cauchy_green():
    """변형 구배 F로부터 우변형 텐서 C = F^T F 계산이 올바른지 검증"""
    # 1축 인장 조건의 F (e = 0.1, 체적 보존 단순 모델)
    F = jnp.array([
        [1.1, 0.0, 0.0],
        [0.0, 1.0/jnp.sqrt(1.1), 0.0],
        [0.0, 0.0, 1.0/jnp.sqrt(1.1)]
    ])
    C = compute_right_cauchy_green(F)
    
    # C = F^T * F 이므로 대각 성분은 [1.21, 1/1.1, 1/1.1]
    expected_C = jnp.diag(jnp.array([1.21, 1.0/1.1, 1.0/1.1]))
    assert jnp.allclose(C, expected_C, atol=1e-6)

def test_tensor_invariants():
    """3D 대칭 텐서 C의 불변량 (I1, I2, I3) 계산 검증"""
    # 임의의 대각 텐서 C = diag(2.0, 3.0, 1.5)
    C = jnp.diag(jnp.array([2.0, 3.0, 1.5]))
    I1, I2, I3 = compute_invariants(C)
    
    # I1 = 2.0 + 3.0 + 1.5 = 6.5
    # I2 = 2*3 + 3*1.5 + 1.5*2 = 6.0 + 4.5 + 3.0 = 13.5
    # I3 = 2.0 * 3.0 * 1.5 = 9.0
    assert jnp.allclose(I1, 6.5)
    assert jnp.allclose(I2, 13.5)
    assert jnp.allclose(I3, 9.0)

def test_isochoric_decomposition():
    """등방체적 분해 C_iso = J^(-2/3) * C가 비압축성(det(C_iso) == 1.0)을 완벽하게 만족하는지 검증"""
    # 임의의 체적 팽창 변형 F (J = det(F) = 2.0 * 1.5 * 1.0 = 3.0)
    F = jnp.array([
        [2.0, 0.0, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 1.0]
    ])
    C = compute_right_cauchy_green(F)
    J = jnp.linalg.det(F)
    
    C_iso = decompose_isochoric(C, J)
    det_C_iso = jnp.linalg.det(C_iso)
    
    # det(C_iso)는 1.0이어야 비압축성이 보존됨
    assert jnp.allclose(det_C_iso, 1.0, atol=1e-5)
