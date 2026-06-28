import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp
from wht_prf.model import prf_time_integration

def prf_bayesian_model(times: jnp.ndarray, F_history: jnp.ndarray, target_diff: jnp.ndarray, 
                       fixed_viscous_params: dict):
    """NumPyro 기반 베이지안 사후 확률 분포 모델을 정의합니다.
    연산 효율성을 위해 평형 초탄성 Yeoh 파라미터(C10, C20, C30)에 대해서만 불확실성을 평가합니다.
    
    Args:
        times (jnp.ndarray): 시간 배열
        F_history (jnp.ndarray): 변형구배 역사 텐서
        target_diff (jnp.ndarray): 타겟 1축 응력 차이
        fixed_viscous_params (dict): 고정된 점성 네트워크 정보 (stiffness_ratio, creep_params)
    """
    # 1. 사전 확률 분포 (Priors) 정의
    # Yeoh 파라미터들
    C10 = numpyro.sample("C10", dist.Uniform(0.05, 3.0))
    C20 = numpyro.sample("C20", dist.Normal(0.0, 0.5))
    C30 = numpyro.sample("C30", dist.Normal(0.0, 0.2))
    D1 = 10.0 # D1은 정적 고정
    
    # 관측 모델의 가우시안 노이즈 표준편차
    sigma_noise = numpyro.sample("sigma_noise", dist.HalfNormal(0.02))
    
    # 2. 고정된 점탄성 파라미터들과 융합하여 PRF 물성 사전 딕셔너리 구축
    mat_data = {
        "hyperelastic_type": "YEOH",
        "hyperelastic_params": jnp.array([C10, C20, C30, D1]),
        "networks": [
            {
                "stiffness_ratio": fixed_viscous_params["stiffness_ratio"],
                "creep_params": jnp.concatenate([fixed_viscous_params["creep_params"], jnp.array([0.0])]) # m=0.0 추가
            }
        ]
    }
    
    # 3. 물리 엔진 시뮬레이션 응답 유도
    pred_stresses = prf_time_integration(mat_data, times, F_history)
    pred_diff = pred_stresses[:, 0, 0] - pred_stresses[:, 1, 1]
    
    # 4. 우도 (Likelihood) 정의
    # 실측치 target_diff는 예측값 pred_diff 주변에 정규분포(노이즈=sigma_noise) 형태로 분포한다고 가정
    numpyro.sample("obs", dist.Normal(pred_diff, sigma_noise), obs=target_diff)


def run_mcmc_inference(times: jnp.ndarray, F_history: jnp.ndarray, target_diff: jnp.ndarray, 
                       fixed_viscous_params: dict, num_samples: int = 200, num_warmup: int = 100) -> dict:
    """NUTS 해밀토니안 몬테카를로(HMC) 방식을 구동하여 초탄성 파라미터의 MCMC 샘플 사후 분포를 추출합니다.
    
    Args:
        times (jnp.ndarray): 시간 배열
        F_history (jnp.ndarray): 변형구배 역사 텐서
        target_diff (jnp.ndarray): 타겟 응력 차이
        fixed_viscous_params (dict): 고정 비선형 점탄성 파라미터 딕셔너리
        num_samples (int): 샘플링 수
        num_warmup (int): 웜업(버림) 수
        
    Returns:
        dict: MCMC 샘플링 결과 (사후 분포 샘플들의 딕셔너리)
    """
    # NUTS 커널 설정
    nuts_kernel = NUTS(prf_bayesian_model)
    
    # MCMC 구동 클래스 초기화
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    
    # 난수 키 획득 및 샘플링 개시
    rng_key = jax.random.PRNGKey(101)
    mcmc.run(rng_key, times, F_history, target_diff, fixed_viscous_params)
    
    # 샘플링 데이터 리턴
    return mcmc.get_samples()
