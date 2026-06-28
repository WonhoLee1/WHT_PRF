import os
import jax.numpy as jnp
import pandas as pd

def parse_experimental_data(data_dir: str):
    """
    지정된 디렉토리 내의 실험 데이터(.txt) 파일들을 파싱하여 
    최적화 엔진에 입력할 수 있는 (시간, 변형률, F_history, 응력차이) 딕셔너리 리스트를 반환합니다.
    """
    datasets = []
    
    # 1. Rate 데이터 로드 (탭 분리, 헤더 없음)
    rate_file = os.path.join(data_dir, "rate_data_1e+2_partial.txt")
    if os.path.exists(rate_file):
        df_rate = pd.read_csv(rate_file, sep='\t', header=None, names=["Time", "EngStrain", "EngStress"])
        datasets.append(_process_dataframe(df_rate, "Rate_1e+2"))
        
    # 2. Relax 데이터 로드 (쉼표 분리, # 커멘트 있음)
    relax_files = ["relax_data_0050.txt", "relax_data_0075.txt", "relax_data_0100.txt", "relax_data_0150.txt"]
    for rfile in relax_files:
        fpath = os.path.join(data_dir, rfile)
        if os.path.exists(fpath):
            # comment='#' 를 사용하여 주석 무시
            df_relax = pd.read_csv(fpath, sep=',', comment='#', header=None, names=["Time", "EngStrain", "EngStress"])
            datasets.append(_process_dataframe(df_relax, rfile.replace(".txt", "")))
            
    return datasets

def _process_dataframe(df, name):
    # 중복 시간 제거 및 정렬 보장
    df = df.drop_duplicates(subset=["Time"]).sort_values("Time")
    
    # 시작점을 (0,0,0)으로 보정 (혹시 0부터 시작하지 않는 경우를 대비)
    if df["Time"].iloc[0] > 1e-6:
        zero_row = pd.DataFrame({"Time": [0.0], "EngStrain": [0.0], "EngStress": [0.0]})
        df = pd.concat([zero_row, df]).reset_index(drop=True)
    
    times = jnp.array(df["Time"].values)
    eng_strains = jnp.array(df["EngStrain"].values)
    eng_stresses = jnp.array(df["EngStress"].values)
    
    # Engineering -> True Stress/Strain 변환
    true_strains = jnp.log(1.0 + eng_strains)
    lam = 1.0 + eng_strains
    
    # 변형 구배 텐서 F (Incompressible Uniaxial Tension 가정)
    F_history = []
    for l in lam:
        F = jnp.diag(jnp.array([l, 1.0/jnp.sqrt(l), 1.0/jnp.sqrt(l)]))
        F_history.append(F)
    F_history = jnp.stack(F_history)
    
    # Cauchy Stress Difference (sigma_11 - sigma_22)
    # True Stress = Eng Stress * lam
    true_stress_11 = eng_stresses * lam
    target_diff = true_stress_11 - 0.0 # sigma_22 = 0
    
    return {
        "name": name,
        "times": times,
        "strains": eng_strains,  # Plot용으로 보존 (X축 시각화용)
        "F_history": F_history,
        "target_diff": target_diff
    }
