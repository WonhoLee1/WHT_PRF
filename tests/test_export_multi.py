import os
import shutil
import pytest
import jax.numpy as jnp
from wht_prf.export import generate_umat_cpp_wrapper

def test_export_cpp_binding_generation(tmp_path):
    """JAX PRF 모델의 Abaqus/OptiStruct 연동용 C++ UMAT 소스 코드 및 빌드 시스템 생성 검증"""
    # 1. 벤치마크용 Yeoh PRF 재료 계수 정의
    mat_data = {
        "hyperelastic_type": "YEOH",
        "hyperelastic_params": jnp.array([0.75, -0.03, 0.012, 10.0]),
        "networks": [
            {
                "stiffness_ratio": 0.38,
                "creep_params": jnp.array([1.2e-5, 3.5, 0.0])
            }
        ]
    }
    
    # 2. 임시 폴더에 UMAT C++ 바인딩 파일 자동 생성 요청
    export_dir = os.path.join(tmp_path, "umat_export")
    os.makedirs(export_dir, exist_ok=True)
    
    # generate_umat_cpp_wrapper(mat_data, export_dir) 호출
    success = generate_umat_cpp_wrapper(mat_data, export_dir)
    
    assert success is True
    
    # 3. 필수 제너레이션 파일 목록 검사
    expected_files = [
        "umat_wrapper.cpp",
        "umat_wrapper.h",
        "CMakeLists.txt"
    ]
    for filename in expected_files:
        filepath = os.path.join(export_dir, filename)
        assert os.path.exists(filepath)
        
        # 파일 크기가 유효한지 검증
        assert os.path.getsize(filepath) > 0
        
    # 4. 생성된 C++ 파일에 필수 UMAT 함수 정의와 Fortran 인터페이스가 기재되어 있는지 파싱
    with open(os.path.join(export_dir, "umat_wrapper.cpp"), "r", encoding="utf-8") as f:
        cpp_content = f.read()
        # Abaqus UMAT Fortran 호환 시그니처가 외부에 열려 있는지 검사
        assert "umat" in cpp_content.lower()
        assert "extern \"C\"" in cpp_content
        # 강성 매트릭스 DDSDDE 업데이트 관련 문구가 포함되어 있는지 확인
        assert "ddsdde" in cpp_content.lower()
        # Stress 업데이트 관련 문구가 포함되어 있는지 확인
        assert "stress" in cpp_content.lower()
