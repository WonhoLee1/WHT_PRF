import jax
import jax.numpy as jnp
import optax
from wht_prf.model import prf_time_integration

def fit_prf_parameters(init_params: dict, datasets: list, max_epochs: int = 150) -> tuple:
    """Optax 경사 하강 최적화를 이용하여 다중 시계열(Multi-curve) 실험 데이터를 동시에 만족하는 PRF 물성 파라미터를 역산 식별합니다.
    
    Args:
        init_params (dict): 초기 매개변수 값들
        datasets (list): 파싱된 실험 데이터 딕셔너리 리스트 
                         [{"times": jnp.array, "F_history": jnp.array, "target_diff": jnp.array}, ...]
        max_epochs (int): 학습 Epoch 횟수
        
    Returns:
        tuple: (best_params_dict, loss_history)
    """
    # 0. 하위 호환성 보정: init_params에 networks가 없으면 단일 네트워크 구조로 패킹
    if "networks" not in init_params:
        init_params = {
            "hyperelastic_params": init_params["hyperelastic_params"],
            "networks": [
                {
                    "stiffness_ratio": init_params["stiffness_ratio"],
                    "creep_params": init_params["creep_params"]
                }
            ]
        }
        
    he_params = init_params["hyperelastic_params"]
    networks = init_params["networks"]
    num_nets = len(networks)
    
    # 각 네트워크 초기 계수 적층
    init_ratios = []
    init_A = []
    init_n = []
    init_m = []
    for net in networks:
        init_ratios.append(net["stiffness_ratio"])
        cp = net["creep_params"]
        init_A.append(cp[0])
        init_n.append(cp[1])
        init_m.append(cp[2] if len(cp) >= 3 else 0.0)
        
    # 1. 제약 조건 회피를 위한 내부 매핑 파라미터 (Log/Logit/Sigmoid-scale) 초기화
    init_m_arr = jnp.array(init_m)
    p_m = jnp.clip(-init_m_arr / 1.5, 1e-5, 0.999)
    logit_m = jnp.log(p_m / (1.0 - p_m))
    
    p_c20 = jnp.clip((he_params[1] + 5000.0) / 10000.0, 1e-5, 0.999)
    logit_c20 = jnp.log(p_c20 / (1.0 - p_c20))
    p_c30 = jnp.clip((he_params[2] + 5000.0) / 10000.0, 1e-5, 0.999)
    logit_c30 = jnp.log(p_c30 / (1.0 - p_c30))
    
    mapped_params = {
        "log_C10": jnp.log(jnp.maximum(he_params[0], 1e-5)),
        "logit_C20": logit_c20,
        "logit_C30": logit_c30,
        "logit_ratios": jnp.log(jnp.array(init_ratios) / (0.99 - jnp.array(init_ratios))),
        "log_A": jnp.log(jnp.maximum(jnp.array(init_A), 1e-25)),
        "log_n": jnp.log(jnp.maximum(jnp.array(init_n) - 1.0, 1e-5)),
        "m_scaled": logit_m
    }
    
    # 2. Adam 옵티마이저 정의
    learning_rate = 0.05
    optimizer = optax.chain(
        optax.clip_by_global_norm(50.0),
        optax.adam(learning_rate=learning_rate)
    )
    opt_state = optimizer.init(mapped_params)
    
    def reconstruct_mat_data(params):
        """내부 매핑 파라미터를 원래 물리적 파라미터로 재구성합니다."""
        C10 = jnp.exp(params["log_C10"])
        C20 = jax.nn.sigmoid(params["logit_C20"]) * 10000.0 - 5000.0
        C30 = jax.nn.sigmoid(params["logit_C30"]) * 10000.0 - 5000.0
        D1 = 10.0 # 벌크 모듈러스 페널티 고정
        
        # 강성 비율 합산 제약 (합이 0.99를 초과하지 못하도록 동적 보정)
        ratios = jax.nn.sigmoid(params["logit_ratios"]) * 0.99
        sum_ratios = jnp.sum(ratios)
        ratios_safe = jnp.where(sum_ratios > 0.99, ratios * (0.99 / (sum_ratios + 1e-12)), ratios)
        
        A_arr = jnp.exp(params["log_A"])
        n_arr = jnp.exp(params["log_n"]) + 1.0
        m_arr = jax.nn.sigmoid(params["m_scaled"]) * (-1.5) # m in [-1.5, 0.0]
        
        networks_list = []
        for i in range(num_nets):
            networks_list.append({
                "stiffness_ratio": ratios_safe[i],
                "creep_params": jnp.array([A_arr[i], n_arr[i], m_arr[i]])
            })
            
        mat_data = {
            "hyperelastic_type": "YEOH",
            "hyperelastic_params": jnp.array([C10, C20, C30, D1]),
            "networks": networks_list
        }
        return mat_data
        
    def loss_fn(params, times_tuple, F_hist_tuple, target_tuple):
        """다중 실험 데이터의 평균제곱오차(MSE)를 합산하여 Total Loss 계산"""
        mat_data = reconstruct_mat_data(params)
        total_loss = 0.0
        
        # JAX jitted function 안에서는 파이썬 튜플에 대한 루프가 정적으로 풀림(Unroll)
        for t_arr, f_arr, tgt_arr in zip(times_tuple, F_hist_tuple, target_tuple):
            pred_stresses = prf_time_integration(mat_data, t_arr, f_arr)
            pred_diff = pred_stresses[:, 0, 0] - pred_stresses[:, 1, 1]
            
            max_time = jnp.maximum(t_arr[-1], 1.0)
            tau = max_time * 0.2
            time_weights = 1.0 + 10.0 * jnp.exp(-t_arr / tau)
            
            loss = jnp.mean(time_weights * (pred_diff - tgt_arr) ** 2)
            total_loss += loss
            
        return total_loss
        
    loss_val_and_grad = jax.jit(jax.value_and_grad(loss_fn, argnums=0))
    
    # 3. 최적화 루프: 다중 데이터 셋 길이에 대응하는 점진적 윈도우 확장
    params_curr = mapped_params
    loss_history = []
    
    # 윈도우 확장의 기준을 첫번째 데이터셋의 총 스텝수로 설정 (보통 Rate Test)
    base_total_steps = len(datasets[0]["times"])
    
    stages = [
        {"frac": 0.2, "epochs": int(max_epochs * 0.2)},
        {"frac": 0.5, "epochs": int(max_epochs * 0.3)},
        {"frac": 1.0, "epochs": max_epochs - int(max_epochs * 0.2) - int(max_epochs * 0.3)}
    ]
    
    for stage_idx, stage in enumerate(stages):
        base_limit = max(2, int(base_total_steps * stage["frac"]))
        
        # 현재 스테이지의 진행률(frac)에 맞춰 모든 데이터셋의 윈도우 크기를 조절
        # (Relaxation 데이터들은 Rate 보다 스텝 수가 훨씬 길 수 있으므로 동일 비율로 잘라줌)
        sub_times_list = []
        sub_F_list = []
        sub_target_list = []
        for ds in datasets:
            limit_idx = max(2, int(len(ds["times"]) * stage["frac"]))
            sub_times_list.append(ds["times"][:limit_idx])
            sub_F_list.append(ds["F_history"][:limit_idx])
            sub_target_list.append(ds["target_diff"][:limit_idx])
            
        sub_times_tuple = tuple(sub_times_list)
        sub_F_tuple = tuple(sub_F_list)
        sub_target_tuple = tuple(sub_target_list)
        
        for epoch in range(stage["epochs"]):
            loss_val, grads = loss_val_and_grad(params_curr, sub_times_tuple, sub_F_tuple, sub_target_tuple)
            updates, opt_state = optimizer.update(grads, opt_state, params_curr)
            params_curr = optax.apply_updates(params_curr, updates)
            
            loss_history.append(float(loss_val))
        
    # 4. 최종 결과 복원
    final_mat = reconstruct_mat_data(params_curr)
    best_params = {
        "hyperelastic_params": final_mat["hyperelastic_params"][:-1],
        "networks": [
            {
                "stiffness_ratio": float(net["stiffness_ratio"]),
                "creep_params": net["creep_params"]
            } for net in final_mat["networks"]
        ]
    }
    
    # 하위 호환성
    best_params["stiffness_ratio"] = best_params["networks"][0]["stiffness_ratio"]
    best_params["creep_params"] = best_params["networks"][0]["creep_params"][:-1]
    
    return best_params, loss_history
