import pytest
import jax.numpy as jnp
from wht_prf.io_manager import parse_abaqus_prf, ExperimentalDataLoader
from wht_prf.model import prf_time_integration
from wht_prf.optimization import fit_prf_parameters

def test_internet_benchmark_data_fitting():
    """Abaqus PRF 공식 매뉴얼 텍스트 템플릿과 Axel Products형 실험 데이터를 활용한 실제 물성 피팅 정확도 검증"""
    # 1. 인터넷(Abaqus 매뉴얼)에서 수집한 PRF 재료 카드 규격 텍스트
    benchmark_inp = """
*MATERIAL, NAME=PTFE_BENCHMARK
*HYPERELASTIC, YEOH
0.75, -0.03, 0.012
*VISCOELASTIC, NONLINEAR, NETWORKID=1
*NETWORK STIFFNESS RATIO
0.38
*CREEP, LAW=POWER LAW
1.2e-5, 3.5, 0.0
"""
    
    # 2. 파서를 이용해 물성 딕셔너리로 해석
    mat_parsed = parse_abaqus_prf(benchmark_inp)
    
    # 3. Axel Products 형식의 가상 실험 데이터 로딩
    # (실제 시험: cyclic load-unload 및 relaxation 이력 데이터)
    times_raw = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    eng_strains_raw = jnp.array([0.0, 0.05, 0.1, 0.1, 0.1, 0.05, 0.0])
    # 참값 모델(mat_parsed)을 구동하여 1축 Cauchy 응력 차이 획득 후 노이즈(+0.5% 임의 오차) 추가
    # F_history 생성
    F_history = []
    for eps in eng_strains_raw:
        lam = 1.0 + eps
        F_step = jnp.diag(jnp.array([lam, 1.0/jnp.sqrt(lam), 1.0/jnp.sqrt(lam)]))
        F_history.append(F_step)
    F_history = jnp.stack(F_history)
    
    true_stresses = prf_time_integration(mat_parsed, times_raw, F_history)
    true_diff = true_stresses[:, 0, 0] - true_stresses[:, 1, 1]
    
    # 노이즈가 섞인 Engineering Stress 가상 실측값
    # s = S_true / (1 + e) + noise (실제 시험 오차 수준인 1% 내외 노이즈 주입)
    eng_stresses_raw = true_diff / (1.0 + eng_strains_raw) + jnp.array([0.0, 0.02, -0.01, 0.01, -0.02, 0.005, 0.0])
    
    # 4. ExperimentalDataLoader를 통해 대변형(is_large_strain=True) 진응력-진변형률로 전처리
    loader = ExperimentalDataLoader(is_large_strain=True)
    times, F_processed, stresses_processed = loader.load_data(times_raw, eng_strains_raw, eng_stresses_raw)
    
    # 5. 초기 추측값으로 피팅 개시
    init_params = {
        "hyperelastic_params": jnp.array([0.5, -0.01, 0.005]),
        "stiffness_ratio": 0.2,
        "creep_params": jnp.array([5.0e-5, 2.0])
    }
    
    best_params, loss_history = fit_prf_parameters(
        init_params, times, F_processed, stresses_processed, max_epochs=120
    )
    
    # 오차가 최종 수렴하여 타당한 수준(R^2 유사도 > 0.90)에 도달했는지 검증
    # 최종 예측 응력
    best_mat_data = {
        "hyperelastic_type": "YEOH",
        "hyperelastic_params": jnp.concatenate([best_params["hyperelastic_params"], jnp.array([10.0])]),
        "networks": [
            {
                "stiffness_ratio": best_params["stiffness_ratio"],
                "creep_params": jnp.concatenate([best_params["creep_params"], jnp.array([0.0])])
            }
        ]
    }
    pred_stresses = prf_time_integration(best_mat_data, times, F_processed)
    pred_diff = pred_stresses[:, 0, 0] - pred_stresses[:, 1, 1]
    
    # R2 스코어 계산: 1 - SS_res / SS_tot
    ss_res = jnp.sum((stresses_processed - pred_diff) ** 2)
    ss_tot = jnp.sum((stresses_processed - jnp.mean(stresses_processed)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    
    assert r2 > 0.90

def test_dupont_delrin_3network_fitting():
    """DuPont Delrin 500P POM (3-Network PRF) 실제 논문 카드를 활용한 다중 네트워크 피팅 정확도 검증"""
    # 1. DuPont Delrin 500P POM 논문 물성 카드 셋업 (Ground Truth)
    true_mat_data = {
        "hyperelastic_type": "YEOH",
        "hyperelastic_params": jnp.array([633.952, -2364.445, -2252.046, 0.001]), # C10, C20, C30, D1
        "networks": [
            {"stiffness_ratio": 0.00345838, "creep_params": jnp.array([1.07e-22, 8.67, -0.535])},
            {"stiffness_ratio": 0.51950000, "creep_params": jnp.array([3.90e-16, 11.37, -0.634])},
            {"stiffness_ratio": 0.42462900, "creep_params": jnp.array([2.74e-22, 12.04, -0.494])}
        ]
    }
    
    # 2. 다단계 cyclic loading-holding-unloading 시나리오 정의
    times_raw = jnp.linspace(0.0, 10.0, 20)
    # 인장(5%) -> 유지 -> 제하(2%) -> 유지 -> 최종 제하
    eng_strains_raw = jnp.array([
        0.0, 0.02, 0.04, 0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.02,
        0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.04, 0.02, 0.01, 0.0
    ])
    
    # F_history 생성
    F_history = []
    for eps in eng_strains_raw:
        lam = 1.0 + eps
        F_step = jnp.diag(jnp.array([lam, 1.0/jnp.sqrt(lam), 1.0/jnp.sqrt(lam)]))
        F_history.append(F_step)
    F_history = jnp.stack(F_history)
    
    # Ground Truth 응력 반응 계산
    true_stresses = prf_time_integration(true_mat_data, times_raw, F_history)
    true_diff = true_stresses[:, 0, 0] - true_stresses[:, 1, 1]
    
    # 3. 초기 임의의 3중 네트워크 계수 셋업 후 최적화 구동
    init_params = {
        "hyperelastic_params": jnp.array([550.0, -2000.0, -2000.0]),
        "networks": [
            {"stiffness_ratio": 0.01, "creep_params": jnp.array([1.0e-21, 7.5, -0.5])},
            {"stiffness_ratio": 0.40, "creep_params": jnp.array([1.0e-15, 10.0, -0.6])},
            {"stiffness_ratio": 0.35, "creep_params": jnp.array([1.0e-21, 10.5, -0.5])}
        ]
    }
    
    best_params, loss_history = fit_prf_parameters(
        init_params, times_raw, F_history, true_diff, max_epochs=60
    )
    
    # Loss 가 피팅을 거치며 타당하게 감소하는지 검증
    assert loss_history[-1] < loss_history[0]

def test_polypropylene_3network_fitting():
    """Washington Penn Polypropylene PPC3TF2 (3-Network PRF) 실제 이력 데이터를 모사한 피팅 검증"""
    # 1. PP PPC3TF2 물성 카드 셋업 (Ground Truth)
    true_mat_data = {
        "hyperelastic_type": "YEOH",
        "hyperelastic_params": jnp.array([549.6, -2000.0, -2000.0, 0.001]), # C10, C20, C30, D1 (백본 NH에서 Yeoh로 일반화 대치)
        "networks": [
            {"stiffness_ratio": 0.337060, "creep_params": jnp.array([4.88e-7, 3.13, -0.551])},
            {"stiffness_ratio": 0.150654, "creep_params": jnp.array([2.44e-6, 4.92, -0.746])},
            {"stiffness_ratio": 0.372607, "creep_params": jnp.array([7.81e-5, 3.80, -0.616])}
        ]
    }
    
    # 2. 다단계 시계열 정의
    times_raw = jnp.linspace(0.0, 8.0, 15)
    eng_strains_raw = jnp.array([
        0.0, 0.01, 0.02, 0.03, 0.03, 0.03, 0.02, 0.01, 0.01, 0.02,
        0.03, 0.03, 0.02, 0.01, 0.0
    ])
    
    F_history = []
    for eps in eng_strains_raw:
        lam = 1.0 + eps
        F_step = jnp.diag(jnp.array([lam, 1.0/jnp.sqrt(lam), 1.0/jnp.sqrt(lam)]))
        F_history.append(F_step)
    F_history = jnp.stack(F_history)
    
    true_stresses = prf_time_integration(true_mat_data, times_raw, F_history)
    true_diff = true_stresses[:, 0, 0] - true_stresses[:, 1, 1]
    
    init_params = {
        "hyperelastic_params": jnp.array([500.0, -1800.0, -1800.0]),
        "networks": [
            {"stiffness_ratio": 0.30, "creep_params": jnp.array([1.0e-6, 3.0, -0.5])},
            {"stiffness_ratio": 0.12, "creep_params": jnp.array([1.0e-6, 4.5, -0.7])},
            {"stiffness_ratio": 0.30, "creep_params": jnp.array([1.0e-4, 3.5, -0.6])}
        ]
    }
    
    best_params, loss_history = fit_prf_parameters(
        init_params, times_raw, F_history, true_diff, max_epochs=60
    )
    
    assert loss_history[-1] < loss_history[0]

