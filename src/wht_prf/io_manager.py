import re
import jax.numpy as jnp

def parse_abaqus_prf(inp_content: str) -> dict:
    """Abaqus PRF 입력 텍스트(.inp)를 파싱하여 JAX PyTree 규격의 딕셔너리로 반환합니다.
    
    Args:
        inp_content (str): Abaqus 입력 데크 문자열
        
    Returns:
        dict: 파싱된 초탄성 및 네트워크 물성 트리
    """
    lines = [line.strip() for line in inp_content.strip().split('\n') if line.strip()]
    
    result = {
        "material_name": "UNKNOWN",
        "hyperelastic_type": "UNKNOWN",
        "hyperelastic_params": jnp.array([]),
        "networks": []
    }
    
    current_network = None
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 1. Material Name
        if line.upper().startswith("*MATERIAL"):
            match = re.search(r"NAME\s*=\s*([a-zA-Z0-9_\-]+)", line, re.IGNORECASE)
            if match:
                result["material_name"] = match.group(1)
        
        # 2. Hyperelasticity
        elif line.upper().startswith("*HYPERELASTIC"):
            if "YEOH" in line.upper():
                result["hyperelastic_type"] = "YEOH"
            elif "NEO" in line.upper() or "NEO-HOOKEAN" in line.upper():
                result["hyperelastic_type"] = "NEO_HOOKEAN"
            elif "ARRUDA" in line.upper() or "ARRUDA-BOYCE" in line.upper():
                result["hyperelastic_type"] = "ARRUDA_BOYCE"
                
            i += 1
            if i < len(lines):
                # 쉼표로 분리하여 계수 파싱
                params = [float(x) for x in lines[i].split(',')]
                result["hyperelastic_params"] = jnp.array(params)
        
        # 3. Viscoelasticity (Nonlinear PRF or Linear Prony)
        elif line.upper().startswith("*VISCOELASTIC"):
            if "TIME=PRONY" in line.upper():
                if "prony_series" not in result:
                    result["prony_series"] = []
                i += 1
                while i < len(lines) and not lines[i].startswith("*"):
                    # PRONY format: g_i, k_i, tau_i
                    parts = lines[i].split(',')
                    g_i = float(parts[0].strip()) if len(parts) > 0 and parts[0].strip() else 0.0
                    
                    if len(parts) > 1 and parts[1].strip():
                        k_i = float(parts[1].strip())
                    else:
                        # 사용자가 k_i를 입력하지 않은 경우 g_i와 동일하게 처리 (일정 포아송비)
                        k_i = g_i
                        
                    tau_i = float(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else 0.0
                    result["prony_series"].append({"g_i": g_i, "k_i": k_i, "tau_i": tau_i})
                    i += 1
                i -= 1 # adjust back since the outer loop increments i
            else:
                current_network = {
                    "stiffness_ratio": 0.0,
                    "creep_params": jnp.array([])
                }
                result["networks"].append(current_network)
            
        # 4. Network Stiffness Ratio
        elif line.upper().startswith("*NETWORK STIFFNESS RATIO"):
            i += 1
            if i < len(lines) and current_network is not None:
                current_network["stiffness_ratio"] = float(lines[i])
                
        # 5. Viscous Creep (Power Law)
        elif line.upper().startswith("*CREEP"):
            i += 1
            if i < len(lines) and current_network is not None:
                params = [float(x) for x in lines[i].split(',')]
                current_network["creep_params"] = jnp.array(params)
                
        i += 1
        
    return result


def generate_radioss_mnf(data: dict) -> str:
    """JAX PyTree 데이터를 Altair Radioss /MAT/LAW100 (MNF) 블록 텍스트 형식으로 변환합니다.
    
    Args:
        data (dict): 초탄성 및 네트워크 데이터를 담은 딕셔너리
        
    Returns:
        str: Radioss .rad 물성 파일 블록 문자열
    """
    mat_name = data.get("material_name", "MAT_1")
    he_type = data.get("hyperelastic_type", "YEOH")
    he_params = data.get("hyperelastic_params", jnp.array([]))
    networks = data.get("networks", [])
    
    # 1. Header & Hyperelastic parameters
    lines = [
        f"/MAT/LAW100/{mat_name}",
        "#   Hyperelastic Type and Parameters",
        f"#   TYPE: {he_type}"
    ]
    
    # 초탄성 계수 기입 (Yeoh: C10, C20, C30)
    for idx, val in enumerate(he_params):
        lines.append(f"C{idx+1} = {float(val)}")
        
    # 2. Parallel viscoelastic networks
    lines.append("#   Viscoelastic Network definitions")
    for idx, net in enumerate(networks):
        ratio = net.get("stiffness_ratio", 0.0)
        creep = net.get("creep_params", jnp.array([]))
        
        lines.append(f"#   NETWORK {idx+1}")
        lines.append(f"RATIO_{idx+1} = {float(ratio)}")
        
        # Creep Parameters (Power law: A, n, m)
        # 지수 형식(1.5E-05 등)을 보장하기 위해 포맷팅 적용
        if len(creep) >= 2:
            A_val = float(creep[0])
            n_val = float(creep[1])
            m_val = float(creep[2]) if len(creep) >= 3 else 0.0
            lines.append(f"A_{idx+1} = {A_val:.5E}")
            lines.append(f"N_{idx+1} = {n_val}")
            lines.append(f"M_{idx+1} = {m_val}")
            
    return "\n".join(lines)


def parse_radioss_mnf(rad_content: str) -> dict:
    """Altair Radioss /MAT/LAW100 형식을 파싱하여 JAX PyTree 규격의 딕셔너리로 반환합니다.
    
    Args:
        rad_content (str): Radioss 물성 데크 문자열
        
    Returns:
        dict: 파싱된 데이터 트리
    """
    lines = [line.strip() for line in rad_content.strip().split('\n') if line.strip()]
    
    result = {
        "material_name": "UNKNOWN",
        "hyperelastic_type": "UNKNOWN",
        "hyperelastic_params": [],
        "networks": []
    }
    
    he_params_map = {}
    net_map = {}
    
    for line in lines:
        if line.startswith("#"):
            if "TYPE:" in line:
                result["hyperelastic_type"] = line.split("TYPE:")[-1].strip()
            continue
            
        if line.startswith("/MAT/LAW100/"):
            result["material_name"] = line.split("/")[-1].strip()
            continue
            
        # 변수 파싱 (예: C1 = 0.5 또는 RATIO_1 = 0.3)
        if "=" in line:
            key, val = [x.strip() for x in line.split("=")]
            val_float = float(val)
            
            # 초탄성 계수
            if key.startswith("C"):
                idx = int(key[1:]) - 1
                he_params_map[idx] = val_float
            # 네트워크 강성 비율
            elif key.startswith("RATIO_"):
                net_idx = int(key.split("_")[-1]) - 1
                if net_idx not in net_map:
                    net_map[net_idx] = {"stiffness_ratio": 0.0, "creep_params": {}}
                net_map[net_idx]["stiffness_ratio"] = val_float
            # 네트워크 점성 계수 (A, N, M)
            elif key.startswith("A_") or key.startswith("N_") or key.startswith("M_"):
                prefix, net_idx_str = key.split("_")
                net_idx = int(net_idx_str) - 1
                if net_idx not in net_map:
                    net_map[net_idx] = {"stiffness_ratio": 0.0, "creep_params": {}}
                
                param_idx = {"A": 0, "N": 1, "M": 2}[prefix]
                net_map[net_idx]["creep_params"][param_idx] = val_float

    # 딕셔너리를 JAX arrays 형태로 재배열
    sorted_he = [he_params_map[k] for k in sorted(he_params_map.keys())]
    result["hyperelastic_params"] = jnp.array(sorted_he)
    
    for net_idx in sorted(net_map.keys()):
        net_data = net_map[net_idx]
        creep_dict = net_data["creep_params"]
        sorted_creep = [creep_dict[k] for k in sorted(creep_dict.keys())]
        
        result["networks"].append({
            "stiffness_ratio": net_data["stiffness_ratio"],
            "creep_params": jnp.array(sorted_creep)
        })
        
    return result


def generate_abaqus_prf(data: dict) -> str:
    """JAX PyTree 데이터를 Abaqus PRF 입력 텍스트(.inp)로 변환합니다."""
    mat_name = data.get("material_name", "UNKNOWN")
    he_type = data.get("hyperelastic_type", "YEOH")
    he_params = data.get("hyperelastic_params", jnp.array([]))
    networks = data.get("networks", [])
    
    lines = [
        f"*MATERIAL, NAME={mat_name}",
        f"*HYPERELASTIC, {he_type.upper()}"
    ]
    
    # Hyperelastic parameters 콤마로 연결
    he_str = ", ".join([str(float(x)) for x in he_params])
    lines.append(he_str)
    
    for idx, net in enumerate(networks):
        lines.append(f"*VISCOELASTIC, NONLINEAR, NETWORKID={idx+1}")
        lines.append("*NETWORK STIFFNESS RATIO")
        lines.append(str(float(net.get("stiffness_ratio", 0.0))))
        lines.append("*CREEP, LAW=POWER LAW")
        
        creep = net.get("creep_params", jnp.array([]))
        if len(creep) >= 3:
            creep_str = f"{float(creep[0]):.5E}, {float(creep[1])}, {float(creep[2])}"
            lines.append(creep_str)
            
    return "\n".join(lines)


class ExperimentalDataLoader:
    """실험 인장/압축/완화 데이터를 Cauchy 스트레스 및 변형 구배 F로 가공합니다.
    Engineering/True 변형률/응력 변환을 지원합니다.
    """
    def __init__(self, is_large_strain: bool = True):
        """
        Args:
            is_large_strain (bool): True이면 대변형(True Strain/Stress)으로 환산, False이면 소변형(Eng Strain/Stress) 유지.
        """
        self.is_large_strain = is_large_strain
        
    def load_data(self, times: jnp.ndarray, eng_strains: jnp.ndarray, eng_stresses: jnp.ndarray) -> tuple:
        """Engineering Stress-Strain 데이터를 해석용 3D 변형구배 F (N_steps, 3, 3) 및 Cauchy Stress로 전처리합니다.
        
        Args:
            times (jnp.ndarray): 시간 배열
            eng_strains (jnp.ndarray): 공칭 변형률 (Engineering strain)
            eng_stresses (jnp.ndarray): 공칭 응력 (Engineering stress)
            
        Returns:
            tuple: (times, F_history, stresses)
        """
        # 1. 1축 인장/압축 조건 하의 3D 대각 변형구배 (Deformation Gradient) F 시계열 구축
        # F = diag(lam, 1/sqrt(lam), 1/sqrt(lam))
        lams = 1.0 + eng_strains
        
        F_list = []
        for lam in lams:
            F_step = jnp.diag(jnp.array([lam, 1.0 / jnp.sqrt(lam), 1.0 / jnp.sqrt(lam)]))
            F_list.append(F_step)
        F = jnp.stack(F_list)
        
        # 2. 응력 (Stress)
        if self.is_large_strain:
            # 대변형: Cauchy Stress (True Stress) = s * (1 + e)
            stresses = eng_stresses * (1.0 + eng_strains)
        else:
            # 소변형: Cauchy Stress ~= Engineering Stress
            stresses = eng_stresses
            
        return times, F, stresses
