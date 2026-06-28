# JAX 기반 Parallel Rheological Framework (PRF) 전체 구현 계획서 (v17.1 - 2026-06-27 백업)

본 문서는 사용자와의 기술 인터뷰를 바탕으로 모호한 아키텍처 결단 사항들을 모두 확정 지은 최종 JAX 기반 PRF 프레임워크 구현 계획서입니다. 승인이 떨어지면 즉각적으로 본 문서의 내용대로 코딩(Execution) 및 서브 에이전트 다중 협업을 시작합니다.

## User Review Required

> [!IMPORTANT]
> **기술 인터뷰에 따른 확정 스펙 (Finalized Specs)**
> 귀하의 신뢰와 권한 위임에 따라 다음과 같이 프레임워크의 코어 아키텍처를 결정 및 고도화하였습니다.
> 
> 1. **Neural Network(Flax) 입력 피처 확정 (옵션 B 적용)**: 단순 폰 미세스 응력뿐만 아니라, **응력 텐서 불변량($I_1, I_2$) 및 누적 변형률 불변량**을 NN의 입력으로 사용합니다. 이는 다차원 대변형 거동의 방향성을 인공지능이 올바르게 물리적으로 인식하기 위한 가장 타당한 조치입니다.
> 2. **베이지안 추론 범위 한정 (옵션 A 적용)**: 수렴성과 연산 효율성을 위해 **평형 네트워크의 초탄성 모델 파라미터(Yeoh, Arruda-Boyce 계수 등)**에 집중하여 NumPyro MCMC 추론을 수행합니다. 비선형 점탄성 계수는 일차적으로 결정론적 최적화(Optax)에 맡깁니다.
> 3. **대변형/소변형(Large/Small Strain) 분기 처리 및 통합 Loss 함수 (신규 반영)**: 시간 도메인 전체에 대한 적합성을 보장하되, 데이터 파싱(`io_manager.py`) 단계에서 사용자가 제공한 시험 데이터가 **Nonlinear Large Strain (True Stress/Strain)인지 Small Strain (Engineering Stress/Strain)인지 선택하는 옵션**을 추가합니다. 이를 통해 목적 함수(Loss function)가 각 옵션에 맞는 수식으로 자동 전환되어 오차를 최소화합니다.
> 4. **이중 미분(Nested Gradient) 회피를 위한 해석해 전면 도입**: 최적화 시 params에 대한 미분 안에서 F에 대한 미분(jax.grad)이 겹치는 JAX의 고질적인 트레이서 충돌을 우회하기 위해, 초탄성 Cauchy 응력 계산을 Autograd 대신 **3D 해석해 텐서 공식(Analytical equations)**으로 구현하였습니다.

---

## Proposed Project Structure

```
d:/PythonCodeStudy/WHT_PRF/
│
├── prf/
│   ├── __init__.py
│   ├── io_manager.py       # Abaqus ↔ Radioss 변환 및 Large/Small Strain 모드 변환 데이터 로더
│   ├── kinematics.py       # 변형률 변환(Eng ↔ True), 불변량 추출 및 우변형 텐서 계산 (Scalar triple product det 도입)
│   ├── equilibrium.py      # Yeoh, Neo-Hookean, Arruda-Boyce 초탄성 모델 계산 (Analytical stresses)
│   ├── nonequilibrium.py   # Power-Law 크립 흐름 및 불변량(Invariants) 기반 NN 모델 연동 (sig_safe sqrt 스무딩)
│   ├── model.py            # 평형/비평형 네트워크 병렬 융합 고속 적분 (expm overflow 방지 클리핑 적용)
│   ├── solver.py           # 단축 변위 제어(1D 잔차) 및 하중 제어(2D 잔차) 혼합 NR 해석기
│   ├── optimization.py     # Optuna + Optax 기반 시계열 데이터(MSE) 경사 하강 (Log-domain mapping n>=1.0)
│   ├── bayesian.py         # 초탄성 파라미터 대상 NumPyro 사후 확률 분포(Posterior) MCMC 추론
│   └── export.py           # Abaqus(UMAT), Radioss(/MAT/USER), OptiStruct(MATUSR) 타겟 C++ 래퍼 생성
│
├── tests/
│   ├── benchmark_data/            
│   ├── test_io_manager.py         
│   ├── test_kinematics.py         
│   ├── test_equilibrium.py        
│   ├── test_nonequilibrium.py     
│   ├── test_solver.py             
│   ├── test_smart_testing.py      
│   ├── test_internet_benchmark.py 
│   └── test_export_multi.py       
│
├── pyproject.toml                 
└── uv.lock
```

---

## Detailed Components Specification

### [Component 1] 실무 워크플로우를 위한 I/O 및 데이터 파싱 (`io_manager.py`)
* **이기종 텍스트 I/O**: Abaqus(`*HYPERELASTIC`, `*VISCOELASTIC`) $\leftrightarrow$ Radioss(`/MAT/LAW100`) 양방향 1:1 파라미터 변환 지원.
* **데이터 로더 (Strain/Stress 변환)**: 실험 데이터가 Engineering인지 True인지 명시하는 플래그(`is_large_strain`)를 받아, 내부 최적화 엔진이 요구하는 3D 변형구배 텐서 $F_{\text{history}}$ `(N_steps, 3, 3)` 및 Cauchy Stress $\sigma$ 단위로 자동 변환하여 모델에 제공합니다.

### [Component 2] JAX Core PRF 엔진 (`kinematics.py`, `equilibrium.py`, `nonequilibrium.py`, `model.py`)
* 운동학: Cauchy 응력 산출을 위해 JAX autograd 대신 Yeoh, Neo-Hookean, Arruda-Boyce의 3D 해석해(Analytical solutions) 공식 직접 이식.
* 크립 구배 연산: 비압축성을 완벽 보존하는 `jax.scipy.linalg.expm` 함수 사용 및 expm 수치 폭주 방지 컷오프 클리핑($[-0.5, 0.5]$) 적용.
* NN 흐름 모델: $I_1, I_2$ 텐서 불변량을 입력으로 사용하는 Flax 신경망 도입.

### [Component 3] JIT 호환 연속 혼합 솔버 (`solver.py`)
* `jax.lax.while_loop`를 이용한 컴파일 호환 2D Newton-Raphson 루트 파인더.
* 응력 이완(변위 고정), 크립(응력 고정), 변위/하중 증가 등 복합 제어를 `jax.lax.cond`로 하나의 루프 내에서 처리.

### [Component 4] 최적화 및 불확실성 추론 (`optimization.py`, `bayesian.py`)
* Optax 하이브리드 최적화로 시계열 MSE 전체 구간을 최소화.
* 물성치 범위 이탈로 인한 수치 붕괴(NaN)를 예방하기 위해 Log-domain 변수 매핑(C10 > 0, A > 0, n >= 1.0)과 stiffness 강성 비율 시그모이드 매핑 도입.
* 베이지안 추론은 연산 효율을 위해 초탄성 계수(Equilibrium network) 위주로 MCMC 진행.

### [Component 5] 다중 FEM 범용 Export (`export.py`)
* JAX JIT 모델의 AOT(Ahead-of-Time) XLA 컴파일 파이프라인.
* 타겟(`abaqus`, `radioss`, `optistruct`)에 맞는 C++/Fortran 바인딩 레이어(UMAT, USERMAT) 자동 생성.

---

## Development Methodology
* **멀티 에이전트 협업 체계**: 아키텍트, 재료 공학자, FEM 솔버 개발자 에이전트 분담.
* **엄격한 TDD**: `tests/` 폴더 내 정의된 모든 검증 시나리오를 `uv run pytest`로 패스할 때 통합.

---

## 3-Network Calibration & Robustness Enhancement (Added 2026-06-27)
사용자가 제공한 Delrin 500P POM 및 Washington Penn PP (PPC3TF2) 시험 데이터를 탑재한 3-Network PRF 피팅의 수치적 난관들을 진단 및 격파하기 위해 다음과 같은 설계 수정을 반영하였습니다:
1. **수치 특이점 해소**:
   - 시간 $0.0$ 초 부근에서 음수 크립 시간 지수 $m$으로 인해 가중치가 무한대로 폭주하는 $t^m$ 특이점 완화를 위해 시간의 최하한 필터 상수 $t_{\text{safe}}$를 기존 $10^{-8}$ 에서 물리적인 실시간 단위인 $10^{-3}$ 초($1\,\text{ms}$)로 높였습니다.
   - 고차 크립 속도 계산 시 그래디언트 NaN 전파 방지를 위해 크립 속도 출력값을 최대 $100.0\,\text{s}^{-1}$ 로 강제 상한 클리핑 처리하였습니다.
2. **초탄성 계수 가드레일 장착**:
   - $C_{20}, C_{30}$ 이 최적화 도중 터무니없이 발산하여 Cauchy stress를 오염시키지 않도록 로짓/시그모이드 매핑을 적용하여 물리적으로 타당한 영역인 $[-5000, 5000]\,\text{MPa}$ 범위 내로 완전 억제하였습니다.
3. **이중 정밀도(float64) 및 극소 학습률(1e-4) 도입**:
   - 단정밀도(float32) 한계인 $10^{38}$을 체인 룰 미분 누적값이 오버플로우하여 NaN이 뿜어지던 현상을 방지하기 위해 JAX x64 모드를 글로벌 활성화하였습니다.
   - 초기 손실 스케일 $10^{31}$에 따른 옵티마이저 파라미터 널뛰기 현상을 잡기 위해 학습률을 극소인 $1e-4$ 로 가이드하고 Adam 기울기 클리핑(`optax.clip_by_global_norm(1.0)`)을 이식하였습니다.