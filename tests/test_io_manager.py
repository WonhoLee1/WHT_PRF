import pytest
import jax.numpy as jnp
from wht_prf.io_manager import (
    parse_abaqus_prf,
    generate_radioss_mnf,
    parse_radioss_mnf,
    generate_abaqus_prf,
    ExperimentalDataLoader
)

def test_parse_abaqus_prf():
    """Abaqus PRF 입력 텍스트에서 초탄성(Yeoh) 및 Power-Law 점성 파라미터를 정확히 파싱하는지 검증"""
    inp_content = """
*MATERIAL, NAME=TEST_POLYMER
*HYPERELASTIC, YEOH
0.5, 0.1, 0.02
*VISCOELASTIC, NONLINEAR, NETWORKID=1
*NETWORK STIFFNESS RATIO
0.3
*CREEP, LAW=POWER LAW
1.5e-5, 3.2, -0.5
"""
    result = parse_abaqus_prf(inp_content)
    assert result["material_name"] == "TEST_POLYMER"
    assert result["hyperelastic_type"] == "YEOH"
    assert jnp.allclose(result["hyperelastic_params"], jnp.array([0.5, 0.1, 0.02]))
    assert len(result["networks"]) == 1
    assert result["networks"][0]["stiffness_ratio"] == 0.3
    assert jnp.allclose(result["networks"][0]["creep_params"], jnp.array([1.5e-5, 3.2, -0.5]))

def test_generate_radioss_mnf():
    """JAX PyTree 데이터를 Altair Radioss /MAT/LAW100 (MNF) 블록 텍스트 형식으로 올바르게 변환하는지 검증"""
    data = {
        "material_name": "TEST_POLYMER",
        "hyperelastic_type": "YEOH",
        "hyperelastic_params": jnp.array([0.5, 0.1, 0.02]),
        "networks": [
            {
                "stiffness_ratio": 0.3,
                "creep_params": jnp.array([1.5e-5, 3.2, -0.5])
            }
        ]
    }
    rad_content = generate_radioss_mnf(data)
    assert "/MAT/LAW100/TEST_POLYMER" in rad_content
    # Radioss의 특정 점성/초탄성 파라미터가 텍스트에 포함되어 있는지 확인
    assert "0.5" in rad_content
    assert "0.3" in rad_content
    assert "1.5" in rad_content and "E-05" in rad_content

def test_cross_solver_translation():
    """Abaqus PRF -> JAX PyTree -> Radioss MNF -> JAX PyTree -> Abaqus PRF 양방향 변환에서 데이터 무결성이 유지되는지 확인"""
    inp_content = """
*MATERIAL, NAME=PTFE
*HYPERELASTIC, YEOH
0.8, 0.15, 0.03
*VISCOELASTIC, NONLINEAR, NETWORKID=1
*NETWORK STIFFNESS RATIO
0.4
*CREEP, LAW=POWER LAW
2.0e-6, 4.0, -0.2
"""
    # 1. Abaqus 파싱
    parsed_abaqus = parse_abaqus_prf(inp_content)
    
    # 2. Radioss MNF 생성
    rad_content = generate_radioss_mnf(parsed_abaqus)
    
    # 3. Radioss MNF 파싱
    parsed_radioss = parse_radioss_mnf(rad_content)
    
    # 4. 상호 무결성 검증 (파라미터 일치 여부)
    assert parsed_radioss["hyperelastic_type"] == parsed_abaqus["hyperelastic_type"]
    assert jnp.allclose(parsed_radioss["hyperelastic_params"], parsed_abaqus["hyperelastic_params"])
    assert len(parsed_radioss["networks"]) == len(parsed_abaqus["networks"])
    assert parsed_radioss["networks"][0]["stiffness_ratio"] == parsed_abaqus["networks"][0]["stiffness_ratio"]
    assert jnp.allclose(parsed_radioss["networks"][0]["creep_params"], parsed_abaqus["networks"][0]["creep_params"])

def test_experimental_data_loader():
    """대변형(Large Strain) 및 소변형(Small Strain) 변환 로직 검증"""
    # Engineering Strain (e)와 Engineering Stress (s) 가상 데이터
    # e = 0.1, s = 100.0 일 때, 
    # True Strain (E_true) = ln(1 + e) = ln(1.1) ~= 0.09531
    # True Stress (S_true) = s * (1 + e) = 100.0 * 1.1 = 110.0
    times = jnp.array([0.0, 1.0])
    eng_strains = jnp.array([0.0, 0.1])
    eng_stresses = jnp.array([0.0, 100.0])
    
    # 1. 대변형 모드 (True Strain / True Stress 변환)
    loader_large = ExperimentalDataLoader(is_large_strain=True)
    t_l, f_l, stress_l = loader_large.load_data(times, eng_strains, eng_stresses)
    
    # f_l[1] (변형률 구배 F) 의 0,0 성분 = 1 + e = 1.1
    assert jnp.allclose(f_l[1, 0, 0], 1.1)
    # Cauchy Stress (True Stress) = 110.0
    assert jnp.allclose(stress_l[1], 110.0)
    
    # 2. 소변형 모드 (Engineering Strain/Stress 유지)
    loader_small = ExperimentalDataLoader(is_large_strain=False)
    t_s, f_s, stress_s = loader_small.load_data(times, eng_strains, eng_stresses)
    
    # 소변형의 경우 변형률 구배 F ~= 1 + e (소변형 해석과 일치)
    assert jnp.allclose(f_s[1, 0, 0], 1.1)
    # 응력은 Eng Stress 그대로 유지 (100.0)
    assert jnp.allclose(stress_s[1], 100.0)
