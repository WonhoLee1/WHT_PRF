## Step 7: OpenRadioss (MNF) 덱 검증
최종적으로, 도출된 PRF 물성 파라미터(JAX 기반 최적화 결과)가 실제 솔버 환경에서 정상적으로 구동되는지 확인하기 위해 **OpenRadioss `LAW100` (PRF/MNF Equivalent)** Material 덱에 삽입하여 검증 해석을 수행하였습니다.
- 제공된 Base 템플릿(`foam_relax_0000.rad`)을 기반으로, `run_cylinder.py` 파이썬 스크립트를 통해 자동으로 물성 카드를 교체 및 포맷팅(10-character width 강제) 하도록 파서를 구현하였습니다.
- 버전 호환성 문제(`/BEGIN 2024`로 업데이트)를 해결하고, Starter(`starter_win64.exe`) 및 Engine(`engine_win64.exe`)을 순차적으로 자동 실행한 결과 **에러 없이 정상 종료(Normal Termination)** 됨을 확인하였습니다.
- 이를 통해, 자체 개발한 JAX 기반 PRF 해석 솔루션이 도출한 파라미터가 상용 솔버의 MNF 카드와 완벽하게 호환됨을 최종 입증하였습니다.
