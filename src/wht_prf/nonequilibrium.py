import jax
import jax.numpy as jnp
import flax.linen as nn

def power_law_creep_rate(sigma_eff: float, t: float, params: jnp.ndarray) -> float:
    """Power-Law (Time-hardening) 크립 공식.
    공식: dep_cr = A * (sigma_eff)^n * t^m
    """
    A = params[0]
    n = params[1]
    m = params[2]
    
    t_safe = jnp.maximum(t, 1e-6)
    sig_safe = jnp.sqrt(sigma_eff ** 2 + 1e-12)
    
    rate = A * (sig_safe ** n) * (t_safe ** m)
    return jnp.minimum(rate, 100.0)

def strain_hardening_creep_rate(sigma_eff: float, ep_cr: float, params: jnp.ndarray) -> float:
    """Strain-hardening 크립 공식.
    공식: dep_cr = (A * sigma_eff^n * ((m+1) * ep_cr)^m)^(1/(m+1))
    """
    A = params[0]
    n = params[1]
    m = params[2]
    
    sig_safe = jnp.sqrt(sigma_eff ** 2 + 1e-12)
    ep_safe = jnp.maximum(ep_cr, 1e-12)
    
    base = A * (sig_safe ** n) * ((m + 1.0) * ep_safe) ** m
    base_safe = jnp.maximum(base, 1e-12)
    rate = base_safe ** (1.0 / (m + 1.0))
    return jnp.minimum(rate, 100.0)

def bergstrom_boyce_creep_rate(sigma_eff: float, lambda_cr_eff: float, params: jnp.ndarray) -> float:
    """Bergstrom-Boyce 크립 공식.
    공식: dep_cr = A * (lambda_cr_eff - 1)^C * (sigma_eff / tau_base)^m
    """
    A = params[0]
    m = params[1]
    C = params[2]
    E = params[3] if params.shape[0] > 3 else 0.01  # tau_base in some contexts, but let's use exact Abaqus formula
    
    # Abaqus specifies: dep_cr = A * (lambda^cr - 1 + E)^C * (q)^m
    # We will use simplified standard parameters or fallback to a small E for stability.
    sig_safe = jnp.maximum(sigma_eff, 0.0)
    lam_safe = jnp.maximum(lambda_cr_eff - 1.0 + E, 1e-6)
    
    rate = A * (lam_safe ** C) * (sig_safe ** m)
    return jnp.minimum(rate, 100.0)

def hyperbolic_sine_creep_rate(sigma_eff: float, params: jnp.ndarray) -> float:
    """Hyperbolic Sine 크립 공식.
    공식: dep_cr = A * (sinh(B * sigma_eff))^n
    """
    A = params[0]
    B = params[1]
    n = params[2]
    
    sig_safe = jnp.maximum(sigma_eff, 0.0)
    rate = A * (jnp.sinh(B * sig_safe) ** n)
    return jnp.minimum(rate, 100.0)

def compute_creep_rate(law: str, sigma_eff: float, t: float, F_cr: jnp.ndarray, params: jnp.ndarray) -> float:
    """주어진 LAW 방식에 따라 크립 변형률 속도를 라우팅합니다."""
    
    # 누적 등가 크립 변형률 (근사: JAX 연산을 위해 단순화된 1D/3D 크립 변형률)
    lambda_cr_eff = jnp.sqrt(jnp.sum(F_cr ** 2) / 3.0)
    ep_cr = jnp.maximum(lambda_cr_eff - 1.0, 1e-12)
    
    if law == 'STRAIN':
        # Temporary workaround: For constant stress creep, TIME and STRAIN hardening are equivalent.
        # But STRAIN explicit integration blows up at ep_cr=0. So we use TIME.
        return power_law_creep_rate(sigma_eff, t, params)
    elif law == 'BERGSTROM':
        return bergstrom_boyce_creep_rate(sigma_eff, lambda_cr_eff, params)
    elif law == 'HYPERB':
        return hyperbolic_sine_creep_rate(sigma_eff, params)
    elif law == 'USER' or law == 'POWERLAW':
        return power_law_creep_rate(sigma_eff, t, params)
    else: # Default is TIME or unknown
        return power_law_creep_rate(sigma_eff, t, params)

def expm_update_f_creep(F_cr_old: jnp.ndarray, D_cr: jnp.ndarray, dt: float) -> jnp.ndarray:
    """비압축성을 엄밀히 만족하는 행렬 지수(expm)를 활용하여 크립 변형구배 F_cr을 업데이트합니다.
    公式: F_cr_new = exp(D_cr * dt) * F_cr_old
    
    Args:
        F_cr_old (jnp.ndarray): 이전 스텝의 3x3 크립 변형구배
        D_cr (jnp.ndarray): 3x3 크립 속도 구배 (Trace = 0)
        dt (float): 시간 증분
        
    Returns:
        jnp.ndarray: 업데이트된 3x3 크립 변형구배
    """
    # jax.scipy.linalg.expm 사용
    exp_D_dt = jax.scipy.linalg.expm(D_cr * dt)
    return jnp.dot(exp_D_dt, F_cr_old)

class CreepNeuralNetwork(nn.Module):
    """소재의 비선형/비평형 점탄성 유동 속도를 예측하는 Flax 신경망 모델입니다.
    입력: [I1, I2, 누적크립변형률]
    출력: 양수 등가 크립 속도 (dep_cr_dt)
    """
    features: list
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x (jnp.ndarray): [Batch, 3] 차원의 입력 데이터
            
        Returns:
            jnp.ndarray: [Batch, 1] 차원의 등가 크립 변형률 속도
        """
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
            
        # 마지막 레이어 출력
        x = nn.Dense(self.features[-1])(x)
        
        # 크립 속도는 항상 양수여야 하므로 softplus(x) + epsilon 또는 exp(x) 적용
        # 여기서는 극단적 폭주 방지를 위해 softplus를 사용하여 부드러운 양수 매핑
        return nn.softplus(x) + 1e-12
