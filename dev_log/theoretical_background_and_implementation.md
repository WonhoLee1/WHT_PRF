# 이론적 배경 및 구현 상세 (Theoretical Background & Implementation Details)

## 1. 이론적 배경 (Theoretical Background)

### 1.1 PRF (Parallel Rheological Framework) 모델 소개
PRF 모델은 고분자(Polymer) 및 엘라스토머(Elastomer)와 같은 비선형 점탄성 재료의 거동을 모사하기 위해 Abaqus 등 상용 구조해석 프로그램에서 제공하는 고도화된 재료 모델입니다.
재료의 총 응력 $\sigma$은 다수의 병렬 네트워크(Parallel Networks)의 합으로 계산됩니다.

$$ \sigma = \sigma_{\infty} + \sum_{i=1}^N \sigma_i $$

- $\sigma_{\infty}$: 평형 상태의 탄성 응답 (Equilibrium Network)
- $\sigma_i$: 시간에 따라 이완되는 점탄성 네트워크의 응답

각 점탄성 네트워크는 변형률(Strain)을 탄성 변형률과 크립 변형률의 합으로 분해합니다. $\varepsilon = \varepsilon_e + \varepsilon_{cr}$.
이때 PRF `LAW=STRAIN` 모델에서의 크립 변형률 속도(Creep Strain Rate)는 다음과 같은 변형률 경화(Strain Hardening) 함수로 정의됩니다.

$$ \dot{\varepsilon}_{cr} = \left[ A \cdot q^n \cdot ((1+m) \varepsilon_{cr})^m \right]^{\frac{1}{1+m}} $$

여기서:
- $q = |\sigma_i|$ (해당 네트워크의 유효 응력)
- $A, n, m$: 비선형 점탄성 물질 파라미터 (특히 $m < 0$ 인 경우 변형률 연화(Strain Softening)를 표현 가능)

### 1.2 Degenerate PRF (선형 점탄성과의 관계)
PRF 모델에서 파라미터를 $n=1, m=0$ 으로 설정할 경우 (Degenerate PRF), 크립 변형률 속도 수식은 $\dot{\varepsilon}_{cr} = A \cdot q$ 가 되며, 이는 전형적인 Maxwell 모델의 선형 점탄성 점성 거동 $\dot{\varepsilon}_{cr} = \frac{\sigma_i}{\eta_i}$ 과 수학적으로 동일해집니다.
따라서 이 경우, Prony Series의 이완 시간 $\tau_i$와 점성 계수 $A$ 사이에는 다음과 같은 역수 관계가 성립합니다.

$$ A_i = \frac{1}{E_i \tau_i} $$

이 이론적 근거를 바탕으로 Calibration 과정에서 먼저 선형 점탄성(Prony Series)을 도출한 후, 이를 초기 비선형 PRF 모델의 시드(Seed) 값으로 사용합니다.

---

## 2. 구현 상세 (Implementation Details)

본 프로젝트에서는 PPC3TF2 (Polypropylene) 시험 데이터를 기반으로 PRF 파라미터를 역산(Calibration)하는 6단계 프로세스를 구현하였습니다.

### Step 1: Elastic Modulus Determination (`step1_elastic.py`)
- **목표**: 100/s (High Rate) 인장 시험 데이터의 가장 빠른 초기 선형 구간으로부터 재료의 초기 탄성 계수($E_{inst}$)를 추출.
- **결과**: $E_{inst} \approx 3164$ MPa 산출 (논문의 Neo-Hookean $C_{10}=549.6$ 기반 $E=3297.6$ MPa 와 오차율 낮음).

### Step 2: Prony Series 도출 (`step2_prony_series.py`)
- **목표**: 가장 변형이 작은(0.5%) 응력 이완 시험 데이터에 3-term Prony Series 수식을 피팅.
- **로직**: `scipy.optimize.curve_fit`을 사용, $E(t) = E_{\infty} + \sum E_i \exp(-t/\tau_i)$ 를 피팅.
- **결과**: $\tau_1=3.25$s, $\tau_2=23.4$s, $\tau_3=201.4$s 도출 (선형 점탄성 파라미터 확보).

### Step 3: Viscoelasticity 비선형성 확인 (`step3_viscoelasticity_check.py`)
- **목표**: 도출된 선형 Prony Series를 더 큰 변형률(0.75%, 1.0%, 1.5%) 곡선에 적용하여 선형 예측의 한계 확인.
- **결과**: 변형률이 커질수록 선형 모델은 실제 응력보다 훨씬 높게 예측(Strain Softening 효과 미반영). 비선형 PRF 도입의 타당성 검증.

### Step 4 & 5: Degenerate PRF 모델 검증 (`step4_degenerate_prf.py`)
- **목표**: 도출된 선형 파라미터를 $n=1, m=0$ 인 PRF 모델로 변환하여 동일한 거동을 보이는지 수치 적분(Explicit Euler)을 통해 1D 검증.
- **결과**: $A_i = 1/(E_i \tau_i)$ 수식을 적용한 PRF 모델 응답이 Prony Series 응답과 완벽히 일치함을 확인 (모델 기초 완성).

### Step 6: PRF 다목적 파라미터 최적화 (`step6_prf_optimization.py`)
- **목표**: JAX를 활용해 다중 이완 곡선(0.75%, 1.0%, 1.5%)과 인장 곡선을 동시에 피팅(Multi-objective Optimization).
- **구현 특징**:
  - **JAX `lax.scan`**: PRF 모델의 시간 의존적 차분 방정식(ODE)을 JAX 내부에서 초고속으로 풀이 및 자동 미분 지원 구현.
  - **JAX `vmap` (Vectorizing Map)**: 각기 다른 이완 곡선(0.75, 1.0, 1.5%)에 대한 시뮬레이션을 별도의 For 루프 없이 배치 차원(Batch Dimension)으로 병렬화하여 병목 해소.
  - **수치 안정화**: $m < 0$ 일 때의 $0^m$ 발산(NaN) 이슈를 해결하기 위해 `e_cr_eff = jnp.abs(e_cr) + 1e-8` 트릭 및 `jnp.clip` 바운딩 처리.
  - **Optax**: 최신 Gradient Descent 라이브러리인 `optax.adam`을 사용해 $A, n, m, S_{ratio}$ 의 총 12개 변수를 동시에 역산. 변수들은 물리적 범위(예: $n \ge 1, -1 < m \le 0$)를 보장하도록 sigmoid 등의 활성화 함수 매핑 거침.

### 결론
Python 데이터 사이언스 생태계(SciPy, JAX, Optax)를 활용하여 논문에 기술된 PRF Material Calibration 절차를 전 과정 자동화하고 재현할 수 있는 워크플로우를 성공적으로 구축하였습니다. JAX의 `vmap` 기술 적용으로 인해 다중 테스트 곡선 피팅 성능(병렬화)이 극대화되었습니다.


### Step 6 (개선): 암시적 시간 적분 및 JAX Auto-diff 적용
- **목표**: 높은 비선형성 구간(m < 0, n > 1)에서 발생하는 명시적(Explicit) 적분의 수치적 불안정성 및 오실레이션(진동) 해결.
- **구현 특징**:
  - **Implicit Newton-Raphson Solver**: `jax.lax.fori_loop`를 이용해 잔차(Residual)가 허용 오차 이하로 수렴할 때까지 크립 변형률을 반복 업데이트.
  - **`jax.jacfwd` 활용**: 사람이 수기로 자코비안(Tangent Stiffness)을 미분하여 하드코딩하는 대신, JAX의 Forward-mode Auto-differentiation을 활용해 정확한 자코비안 행렬을 동적으로 도출. 
  - 이를 통해 Abaqus UMAT/VUMAT 등에서 필수적인 해석적 미분 과정 없이도 완벽히 호환되는 수치 적분기를 매우 간결하고 강건(Robust)하게 구현 완료.

### Step 7: OpenRadioss MNF (LAW100) 솔버 포팅 및 검증 (`run_cylinder.py`)
- **목표**: JAX로 교정된 PRF 모델 파라미터가 상용 유한요소 솔버인 OpenRadioss의 `LAW100` (PRF/MNF Equivalent)에서 오차 없이 구동되는지 검증.
- **구현 특징**:
  - **Radioss Deck Parser**: 제공된 `foam_relax_0000.rad` 베이스 템플릿 파일에서 `/MAT` 카드를 파싱하고, JAX가 산출한 파라미터(C10, S_Ratio, A, n, m)를 주입.
  - **Fortran f10.0 포맷팅**: `deck_builder.py` 모듈을 통해 Radioss의 엄격한 10-character 고정 폭(Fixed-field) 포맷에 맞추어 모든 부동소수점을 변환.
  - **솔버 자동화**: Python `subprocess` 모듈을 통해 OpenRadioss의 `starter_win64.exe`와 `engine_win64.exe`를 순차적으로 백그라운드 호출 및 검증.
  - **결과**: 엔진 정상 종료(Normal Termination)를 확인하였으며, 자체 JAX 프레임워크와 OpenRadioss 상용 솔버 간의 완벽한 상호 운용성을 확보함.

