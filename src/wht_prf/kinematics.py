import jax
import jax.numpy as jnp

def compute_det_3x3(A: jnp.ndarray) -> float:
    """스칼라 삼중곱 (row0 . (row1 x row2))을 활용하여 3x3 행렬의 Determinant를 JAX 이중 미분에 강건하게 계산합니다."""
    row0 = A[0, :]
    row1 = A[1, :]
    row2 = A[2, :]
    return jnp.dot(row0, jnp.cross(row1, row2))

def compute_right_cauchy_green(F: jnp.ndarray) -> jnp.ndarray:
    """변형 구배 F로부터 우변형 텐서 C = F^T F를 계산합니다.
    
    Args:
        F (jnp.ndarray): 3x3 변형구배 텐서
        
    Returns:
        jnp.ndarray: 3x3 우변형 텐서 C
    """
    return jnp.dot(F.T, F)

def compute_invariants(C: jnp.ndarray) -> tuple:
    """3x3 대칭 텐서 C의 세 가지 주불변량(I1, I2, I3)을 계산합니다.
    
    Args:
        C (jnp.ndarray): 3x3 우변형 텐서
        
    Returns:
        tuple: (I1, I2, I3) 불변량 스칼라 값들
    """
    # I1 = tr(C)
    I1 = jnp.trace(C)
    
    # I2 = 0.5 * ( (tr(C))^2 - tr(C^2) )
    C2 = jnp.dot(C, C)
    I2 = 0.5 * (I1**2 - jnp.trace(C2))
    
    # I3 = det(C)
    I3 = compute_det_3x3(C)
    
    return I1, I2, I3

def decompose_isochoric(C: jnp.ndarray, J: float) -> jnp.ndarray:
    """우변형 텐서 C를 등방체적(Isochoric) 성분 C_iso = J^(-2/3) * C 로 분해합니다.
    
    Args:
        C (jnp.ndarray): 3x3 우변형 텐서
        J (float): 체적 변화율 det(F)
        
    Returns:
        jnp.ndarray: 3x3 등방체적 우변형 텐서 C_iso
    """
    # J^(-2/3) 스케일링 적용
    factor = J ** (-2.0 / 3.0)
    return factor * C
