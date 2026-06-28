# JAX WHT_PRF vs Abaqus Bulk Validation Plan

This document outlines the procedure to run the JAX WHT_PRF model across all 50+ INP benchmarks, calculate the correlation metrics ($R^2$, RMSE), and generate a final summary table.

## Goal Description
1. Update the Python script generator (`generate_benchmarks.py`) so that all 50+ scripts load their respective Abaqus reference data (`_abaqus_results.txt`), not just the `REP_` models.
2. Embed metric calculation logic (interpolation of time steps, $R^2$ score for Stress and Strain) directly into the generated benchmark scripts.
3. Execute all JAX benchmark scripts.
4. Compile the resulting metrics into a final Markdown report table.

## Open Questions

> [!WARNING]
> **JIT Compilation Overhead & Runtime**
> 각 테스트 스크립트마다 JAX JIT 컴파일이 발생합니다. 모델 1개당 약 1~2분이 소요되므로 순차적으로 50개를 돌리면 1시간~2시간이 걸릴 수 있습니다.
> 이를 단축하기 위해 파이썬 `concurrent.futures`를 이용해 **4~6개의 프로세스를 병렬(Parallel)로 동시에 실행**하여 전체 시간을 약 15~20분 내외로 단축하고자 합니다. 시스템 메모리 및 CPU 자원을 일시적으로 많이 사용할 수 있는데, 이 병렬 실행 방식을 승인하시겠습니까?

## Proposed Changes

### 1. `generate_benchmarks.py`
- [MODIFY] `generate_benchmarks.py`: 
  - Remove the `is_rep` restriction so ALL scripts compare against their Abaqus reference.
  - Inject `scipy.interpolate.interp1d` logic to align Abaqus time steps with JAX time steps.
  - Calculate $R^2$ and $RMSE$ for Stress and Strain.
  - Save the calculated metrics to `{job_name}_metrics.json`.
  
### 2. `run_all_jax_benchmarks.py`
- [NEW] `run_all_jax_benchmarks.py`: A master script that uses `concurrent.futures.ProcessPoolExecutor` to run `python run_*.py` for all models concurrently, maximizing CPU usage to bypass the JIT overhead time limit.

### 3. `generate_jax_report.py`
- [NEW] `generate_jax_report.py`: Gathers all the `{job_name}_metrics.json` files and formats them into a GitHub Markdown table sorted by accuracy, and saves it as an artifact.

## Verification Plan
1. Re-run `generate_benchmarks.py` and inspect one generated `run_*.py` to ensure metric logic is correct.
2. Run `run_all_jax_benchmarks.py` in the background and monitor CPU/RAM.
3. Present the compiled artifact `jax_validation_report.md` to the user with the embedded table of $R^2$ values.
