import os

def generate_umat_cpp_wrapper(mat_data: dict, output_dir: str) -> bool:
    """JAX PRF 피팅 물성치를 직접 하드코딩된 정적 상수 카드로 주입하여, 
    순수 C++로 구현된 Parallel Rheological Framework (PRF) 연동용 UMAT 동적 라이브러리(DLL) 소스 및 빌드 환경을 자동 생성합니다.
    
    Args:
        mat_data (dict): 피팅 완료된 Yeoh / Power-law 크립 계수 딕셔너리
        output_dir (str): 출력 대상 디렉토리 경로
        
    Returns:
        bool: 성공 여부
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 물성치 추출 및 상문화
        he_params = mat_data.get("hyperelastic_params", [0.5, 0.0, 0.0, 10.0])
        # Yeoh 매개변수 바인딩
        c10 = float(he_params[0])
        c20 = float(he_params[1]) if len(he_params) > 1 else 0.0
        c30 = float(he_params[2]) if len(he_params) > 2 else 0.0
        d1 = float(he_params[3]) if len(he_params) > 3 else 10.0
        
        networks = mat_data.get("networks", [])
        if len(networks) > 0:
            ratio = float(networks[0].get("stiffness_ratio", 0.3))
            creep = networks[0].get("creep_params", [1e-5, 3.0, 0.0])
            creep_A = float(creep[0])
            creep_n = float(creep[1])
            creep_m = float(creep[2]) if len(creep) > 2 else 0.0
        else:
            ratio = 0.3
            creep_A = 1e-5
            creep_n = 3.0
            creep_m = 0.0
            
        # 2. UMAT C++ 헤더 생성
        h_content = f"""#ifndef UMAT_WRAPPER_H
#define UMAT_WRAPPER_H

// Abaqus 및 OptiStruct용 UMAT Fortran 인터페이스 선언
extern "C" {{
    void umat(
        double* stress, double* statev, double* ddsdde, double* sse, double* spd, double* scd,
        double* rpl, double* ddsddt, double* drplde, double* drpldt,
        double* stran, double* dstran, double* time, double* dtime, double* temp, double* dtemp,
        double* predef, double* dpred, char* cmname, double* ndi, double* nshr, double* nstatv,
        double* props, double* nprops, double* coords, double* drot, double* pnewdt, double* celent,
        double* dfgrd0, double* dfgrd1, double* noel, double* npt, double* layer, double* kspt,
        double* kstep, double* kinc
    );
}}

#endif // UMAT_WRAPPER_H
"""
        
        # 3. UMAT C++ 본문 생성 (PRF 비선형 거동 해석 탑재)
        cpp_content = f"""#include "umat_wrapper.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// 하드코딩된 PRF 물성치 사전 정의 (JAX 피팅 산출물 이식)
const double C10 = {c10};
const double C20 = {c20};
const double C30 = {c30};
const double D1 = {d1};

const double RATIO = {ratio};
const double CREEP_A = {creep_A};
const double CREEP_N = {creep_n};
const double CREEP_M = {creep_m};

// 3x3 행렬의 Determinant 대수적 계산 함수
double det_3x3(const double A[3][3]) {{
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
           A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
           A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}}

// 3x3 대각 성분만 존재하는 텐서(1축 인장 등)의 expm 근사 업데이트
void compute_expm_diagonal(const double D[3][3], double out[3][3]) {{
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            if (i == j) {{
                out[i][j] = std::exp(D[i][j]);
            }} else {{
                out[i][j] = 0.0;
            }}
        }}
    }}
}}

// Abaqus UMAT 외부 링크 함수 구현
extern "C" void umat(
    double* stress, double* statev, double* ddsdde, double* sse, double* spd, double* scd,
    double* rpl, double* ddsddt, double* drplde, double* drpldt,
    double* stran, double* dstran, double* time, double* dtime, double* temp, double* dtemp,
    double* predef, double* dpred, char* cmname, double* ndi, double* nshr, double* nstatv,
    double* props, double* nprops, double* coords, double* drot, double* pnewdt, double* celent,
    double* dfgrd0, double* dfgrd1, double* noel, double* npt, double* layer, double* kspt,
    double* kstep, double* kinc
) {{
    // 1. 변형구배 텐서 F_curr 구성 (Abaqus는 2차원 배열 포인터 dfgrd1로 넘겨줌)
    // Abaqus 포맷: dfgrd1[col + 3*row] (Column-major format 호환 처리)
    double F_curr[3][3];
    F_curr[0][0] = dfgrd1[0]; F_curr[0][1] = dfgrd1[3]; F_curr[0][2] = dfgrd1[6];
    F_curr[1][0] = dfgrd1[1]; F_curr[1][1] = dfgrd1[4]; F_curr[1][2] = dfgrd1[7];
    F_curr[2][0] = dfgrd1[2]; F_curr[2][1] = dfgrd1[5]; F_curr[2][2] = dfgrd1[8];
    
    double J = det_3x3(F_curr);
    if (J <= 0.0) J = 1.0; // 예외 처리
    
    // 2. 평형 Yeoh 초탄성 응력 유도 (sigma_eq)
    // b = F * F^T
    double b[3][3] = {{0}};
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            for (int k = 0; k < 3; ++k) {{
                b[i][j] += F_curr[i][k] * F_curr[j][k];
            }}
        }}
    }}
    
    double factor = std::pow(J, -2.0 / 3.0);
    double b_bar[3][3];
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            b_bar[i][j] = factor * b[i][j];
        }}
    }}
    
    double I1_bar = b_bar[0][0] + b_bar[1][1] + b_bar[2][2];
    double term = I1_bar - 3.0;
    
    // Yeoh dW/dI1
    double dW_dI1 = C10 + 2.0 * C20 * term + 3.0 * C30 * (term * term);
    double p_vol = (2.0 / D1) * (J - 1.0);
    
    // 편차 응력 성분 계산 (dev(b_bar))
    double b_bar_dev[3][3];
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            b_bar_dev[i][j] = b_bar[i][j] - (i == j ? (1.0 / 3.0) * I1_bar : 0.0);
        }}
    }}
    
    // sigma_eq = (2/J)*dW_dI1*b_bar_dev + p*I
    double sigma_eq[3][3];
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            sigma_eq[i][j] = (2.0 / J) * dW_dI1 * b_bar_dev[i][j] + (i == j ? p_vol : 0.0);
        }}
    }}
    
    // 3. 비평형 상태 크립 갱신 및 비평형 응력 계산
    // 이전 시간 크립 상태 변수 statev에서 F_cr 복원 (대각선 3성분 가정)
    double F_cr_old[3][3] = {{0}};
    F_cr_old[0][0] = (statev[0] <= 0.0) ? 1.0 : statev[0];
    F_cr_old[1][1] = (statev[1] <= 0.0) ? 1.0 : statev[1];
    F_cr_old[2][2] = (statev[2] <= 0.0) ? 1.0 : statev[2];
    
    // F_cr_inv 계산
    double F_cr_inv[3][3] = {{0}};
    F_cr_inv[0][0] = 1.0 / (F_cr_old[0][0] + 1e-12);
    F_cr_inv[1][1] = 1.0 / (F_cr_old[1][1] + 1e-12);
    F_cr_inv[2][2] = 1.0 / (F_cr_old[2][2] + 1e-12);
    
    // F_e = F * F_cr_inv
    double F_e[3][3] = {{0}};
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            F_e[i][j] = F_curr[i][j] * F_cr_inv[j][j];
        }}
    }}
    
    double J_e = det_3x3(F_e);
    if (J_e <= 0.0) J_e = 1.0;
    
    double b_e[3][3] = {{0}};
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            for (int k = 0; k < 3; ++k) {{
                b_e[i][j] += F_e[i][k] * F_e[j][k];
            }}
        }}
    }}
    
    double factor_e = std::pow(J_e, -2.0 / 3.0);
    double b_e_bar[3][3];
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            b_e_bar[i][j] = factor_e * b_e[i][j];
        }}
    }}
    
    double I1_e_bar = b_e_bar[0][0] + b_e_bar[1][1] + b_e_bar[2][2];
    double term_e = I1_e_bar - 3.0;
    double dW_dI1_e = C10 + 2.0 * C20 * term_e + 3.0 * C30 * (term_e * term_e);
    double p_e = (2.0 / D1) * (J_e - 1.0);
    
    double b_e_bar_dev[3][3];
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            b_e_bar_dev[i][j] = b_e_bar[i][j] - (i == j ? (1.0 / 3.0) * I1_e_bar : 0.0);
        }}
    }}
    
    // 점성 Cauchy 응력 sigma_v
    double sigma_v[3][3];
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            sigma_v[i][j] = (2.0 / J_e) * dW_dI1_e * b_e_bar_dev[i][j] + (i == j ? p_e : 0.0);
        }}
    }}
    
    // 4. 전체 응력 조립 및 출력 바인딩
    // sigma_total = (1 - ratio)*sigma_eq + ratio*sigma_v
    // Abaqus 응력은 1D 벡터로 리턴: [S11, S22, S33, S12, S13, S23]
    double sigma_total[3][3];
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            sigma_total[i][j] = (1.0 - RATIO) * sigma_eq[i][j] + RATIO * sigma_v[i][j];
        }}
    }}
    
    stress[0] = sigma_total[0][0]; // S11
    stress[1] = sigma_total[1][1]; // S22
    stress[2] = sigma_total[2][2]; // S33
    stress[3] = sigma_total[0][1]; // S12
    stress[4] = sigma_total[0][2]; // S13
    stress[5] = sigma_total[1][2]; // S23
    
    // 5. 다음 스텝을 위한 크립 상태(State variables) 업데이트
    double tr_v = sigma_v[0][0] + sigma_v[1][1] + sigma_v[2][2];
    double S_dev[3][3];
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            S_dev[i][j] = sigma_v[i][j] - (i == j ? (1.0 / 3.0) * tr_v : 0.0);
        }}
    }}
    
    // Von Mises
    double contraction = 0.0;
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            contraction += S_dev[i][j] * S_dev[i][j];
        }}
    }}
    double sigma_eff = std::sqrt(1.5 * contraction + 1e-12);
    
    // Power-law 크립 속도
    double dt = *dtime;
    if (dt <= 0.0) dt = 1e-5;
    double t_curr = time[0];
    double t_safe = std::max(t_curr, 1e-8);
    double dep_cr_dt = CREEP_A * std::pow(sigma_eff, CREEP_N) * std::pow(t_safe, CREEP_M);
    
    // N 방향 텐서
    double direction[3][3];
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            direction[i][j] = 1.5 * S_dev[i][j] / std::sqrt(sigma_eff * sigma_eff + 1e-12);
        }}
    }}
    
    // D_cr_dt 계산 및 클리핑
    double D_cr_dt[3][3];
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            double val = dep_cr_dt * direction[i][j] * dt;
            D_cr_dt[i][j] = std::max(-0.5, std::min(0.5, val));
        }}
    }}
    
    // expm 업데이트 적용하여 새로운 F_cr 저장
    double exp_D_dt[3][3];
    compute_expm_diagonal(D_cr_dt, exp_D_dt);
    
    double F_cr_new[3][3] = {{0}};
    for (int i = 0; i < 3; ++i) {{
        for (int j = 0; j < 3; ++j) {{
            for (int k = 0; k < 3; ++k) {{
                F_cr_new[i][j] += exp_D_dt[i][k] * F_cr_old[k][j];
            }}
        }}
    }}
    
    // statev 갱신
    statev[0] = F_cr_new[0][0];
    statev[1] = F_cr_new[1][1];
    statev[2] = F_cr_new[2][2];
    
    // 6. 강성 매트릭스 DDSDDE 초기화 (Abaqus 요구 사항: 탄성 강성으로 초기 근사 대입)
    // 단순 Yeoh 영률 전개 기반 대각 근사 적용
    double E_initial = 4.0 * C10 * (1.0 + RATIO);
    double nu = 0.495; // 비압축성 근사
    double factor_d = E_initial / ((1.0 + nu) * (1.0 - 2.0 * nu));
    
    for (int i = 0; i < 36; ++i) ddsdde[i] = 0.0;
    
    ddsdde[0 + 6*0] = factor_d * (1.0 - nu);
    ddsdde[1 + 6*1] = factor_d * (1.0 - nu);
    ddsdde[2 + 6*2] = factor_d * (1.0 - nu);
    
    ddsdde[0 + 6*1] = factor_d * nu;
    ddsdde[1 + 6*0] = factor_d * nu;
    ddsdde[0 + 6*2] = factor_d * nu;
    ddsdde[2 + 6*0] = factor_d * nu;
    ddsdde[1 + 6*2] = factor_d * nu;
    ddsdde[2 + 6*1] = factor_d * nu;
    
    double G = E_initial / (2.0 * (1.0 + nu));
    ddsdde[3 + 6*3] = G;
    ddsdde[4 + 6*4] = G;
    ddsdde[5 + 6*5] = G;
}}
"""
        
        # 4. CMakeLists.txt 생성 (MSVC, MinGW, GCC 지원 크로스플랫폼 DLL 빌드 시스템)
        cmake_content = """cmake_minimum_required(VERSION 3.10)
project(umat_export CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# DLL 동적 라이브러리 지정
add_library(umat_wrapper SHARED umat_wrapper.cpp)

# Windows 환경에서의 컴파일러별 DLL export 정의
if(MSVC)
    target_compile_options(umat_wrapper PRIVATE /W4)
else()
    target_compile_options(umat_wrapper PRIVATE -Wall -Wextra -O3)
endif()
"""
        
        # 5. 파일 기록 진행 (UTF-8 인코딩 명시)
        with open(os.path.join(output_dir, "umat_wrapper.h"), "w", encoding="utf-8") as f:
            f.write(h_content)
        with open(os.path.join(output_dir, "umat_wrapper.cpp"), "w", encoding="utf-8") as f:
            f.write(cpp_content)
        with open(os.path.join(output_dir, "CMakeLists.txt"), "w", encoding="utf-8") as f:
            f.write(cmake_content)
            
        return True
    except Exception as e:
        print(f"Error generating C++ UMAT: {e}")
        return False
