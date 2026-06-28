## Step 8: 결과 분석 (Time-Strain-Stress 거동)

OpenRadioss 솔버를 통해 출력된 `T01` 시계열 데이터를 추출 및 변환하여, 실린더 모델의 시간에 따른 변형률(Strain) 입력과 응력(Stress) 이완 응답을 분석하였습니다.

- **변형률(Strain)**: 1초 동안 약 -3.3%(-0.033)의 압축 변형률이 인가된 후, 100초까지 일정하게 유지되는 응력 이완(Stress Relaxation) 테스트 조건이 정상적으로 적용되었음을 확인했습니다.
- **응력(Stress, SIGXX)**: 변형률이 증가하는 초기 1초 동안 응력이 약 0.4 MPa까지 급격히 상승한 후, 변형률이 유지되는 동안 점탄성 네트워크에 의해 서서히 이완(Relaxation)되어 약 0.25 MPa 부근으로 감소하는 전형적인 PRF 거동이 시뮬레이션 결과에 반영되었습니다.
- **참고**: 동적 외연적(Explicit) 솔버의 특성 상, 빠른 변형 인가 구간에서 동적 효과(관성 및 응력파 전파)로 인해 응력 곡선에 고주파 진동(Oscillation)이 포함되어 있으나, 평균적인 응력 이완 트렌드는 JAX Implicit 기반의 1D 시뮬레이션 결과와 매우 유사한 거동을 보입니다.

![Time vs Strain vs Stress (OpenRadioss)](C:/Users/GOODMAN/.gemini/antigravity/brain/d27d4e34-8ab9-4a61-bbc5-db108318fbaf/time_strain_stress.png)
