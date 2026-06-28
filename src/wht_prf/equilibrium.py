"""
WHTOOLs MATCALIB 2026
--------------------------------------------------------------------------------
Equilibrium (Hyperelastic) Models with Mullins Effect Support
"""
import jax
import jax.numpy as jnp
from wht_prf.kinematics import compute_right_cauchy_green, decompose_isochoric, compute_det_3x3
import jax.scipy.special as jsp

def neo_hookean_energy(F: jnp.ndarray, params: jnp.ndarray) -> float:
    C10 = params[0]
    D1 = params[1]
    J = compute_det_3x3(F)
    C = compute_right_cauchy_green(F)
    C_bar = decompose_isochoric(C, J)
    I1_bar = jnp.trace(C_bar)
    W_vol = (J - 1.0) ** 2 / jnp.maximum(D1, 1e-6)
    W_dev = C10 * (I1_bar - 3.0)
    return W_dev + W_vol

def yeoh_energy(F: jnp.ndarray, params: jnp.ndarray) -> float:
    C10 = params[0]
    C20 = params[1]
    C30 = params[2]
    D1 = params[3]
    J = compute_det_3x3(F)
    C = compute_right_cauchy_green(F)
    C_bar = decompose_isochoric(C, J)
    I1_bar = jnp.trace(C_bar)
    term = I1_bar - 3.0
    W_dev = C10 * term + C20 * (term ** 2) + C30 * (term ** 3)
    W_vol = (J - 1.0) ** 2 / jnp.maximum(D1, 1e-6)
    return W_dev + W_vol

def arruda_boyce_energy(F: jnp.ndarray, params: jnp.ndarray) -> float:
    mu = params[0]
    lambda_L = params[1]
    D1 = params[2]
    J = compute_det_3x3(F)
    C = compute_right_cauchy_green(F)
    C_bar = decompose_isochoric(C, J)
    I1_bar = jnp.trace(C_bar)
    c1, c2, c3, c4, c5 = 1/2, 1/20, 11/1050, 19/7000, 519/673750
    term1 = I1_bar - 3.0
    term2 = I1_bar**2 - 3.0**2
    term3 = I1_bar**3 - 3.0**3
    term4 = I1_bar**4 - 3.0**4
    term5 = I1_bar**5 - 3.0**5
    W_dev = mu * (c1 * term1 + (c2 / (lambda_L**2)) * term2 + (c3 / (lambda_L**4)) * term3 + (c4 / (lambda_L**6)) * term4 + (c5 / (lambda_L**8)) * term5)
    W_vol = (1.0 / jnp.maximum(D1, 1e-6)) * (0.5 * (J**2 - 1.0) - jnp.log(J))
    return W_dev + W_vol

def ogden_energy(F: jnp.ndarray, params: jnp.ndarray) -> float:
    mu1, a1, mu2, a2, D1, D2 = params[:6]
    D1_safe = jnp.where(D1 < 1e-9, 1e-5, D1)
    b = jnp.dot(F, F.T)
    w, _ = jnp.linalg.eigh(b)
    lambdas = jnp.sqrt(jnp.maximum(w, 1e-12))
    J = compute_det_3x3(F)
    lam_bar = lambdas * J**(-1.0/3.0)
    W_dev = (2.0*mu1/(a1**2)) * (lam_bar[0]**a1 + lam_bar[1]**a1 + lam_bar[2]**a1 - 3.0) + \
            (2.0*mu2/(a2**2)) * (lam_bar[0]**a2 + lam_bar[1]**a2 + lam_bar[2]**a2 - 3.0)
    W_vol = (1.0/D1_safe) * (J - 1.0)**2 + (1.0/(D2 + 1e-12)) * (J - 1.0)**4 * jnp.where(D2 > 1e-9, 1.0, 0.0)
    return W_dev + W_vol

def van_der_waals_energy(F: jnp.ndarray, params: jnp.ndarray) -> float:
    mu, lam_m, a, beta, D = params[:5]
    D_safe = jnp.where(D < 1e-9, 1e-5, D)
    J = compute_det_3x3(F)
    C = compute_right_cauchy_green(F)
    C_bar = decompose_isochoric(C, J)
    I1_bar = jnp.trace(C_bar)
    I2_bar = 0.5 * (I1_bar**2 - jnp.trace(jnp.dot(C_bar, C_bar)))
    I_bar = (1.0 - beta) * I1_bar + beta * I2_bar
    x = jnp.maximum((I_bar - 3.0) / (lam_m**2 - 3.0), 1e-12)
    eta_v = jnp.sqrt(x)
    term1 = -(lam_m**2 - 3.0) * (jnp.log(jnp.maximum(1.0 - eta_v, 1e-12)) + eta_v)
    term2 = -(2.0/3.0) * a * jnp.maximum((I_bar - 3.0)/2.0, 1e-12)**1.5
    W_dev = mu * (term1 + term2)
    W_vol = (1.0 / D_safe) * (0.5 * (J**2 - 1.0) - jnp.log(J))
    return W_dev + W_vol

def compute_mullins_damage(U_dev: float, U_dev_max: float, r: float, m: float, beta: float):
    U_dev_max_new = jnp.maximum(U_dev_max, U_dev)
    x = (U_dev_max_new - U_dev) / (m + beta * U_dev_max_new + 1e-12)
    eta = 1.0 - (1.0 / r) * jsp.erf(x)
    return jnp.clip(eta, 1e-6, 1.0), U_dev_max_new

def compute_hyperelastic_state(he_model_or_type, F: jnp.ndarray, params: jnp.ndarray, mullins_params=None, U_dev_max=0.0):
    if isinstance(he_model_or_type, str):
        he_type = he_model_or_type.upper()
    else:
        name = he_model_or_type.__name__.upper()
        if "YEOH" in name: he_type = "YEOH"
        elif "NEO" in name: he_type = "NEO_HOOKEAN"
        elif "ARRUDA" in name: he_type = "ARRUDA_BOYCE"
        elif "OGDEN" in name: he_type = "OGDEN"
        elif "VAN" in name: he_type = "VAN_DER_WAALS"
        else: he_type = "YEOH"
    
    J = compute_det_3x3(F)
    params_padded = jnp.pad(params, (0, max(0, 8 - len(params))), constant_values=0.0)
    
    if he_type == "OGDEN":
        fn = lambda f: ogden_energy(f, params_padded)
        P = jax.grad(fn)(F)
        sigma = (1.0 / J) * jnp.dot(P, F.T)
        W_total = fn(F)
        D1 = jnp.where(params_padded[4] < 1e-9, 1e-5, params_padded[4])
        D2 = params_padded[5]
        W_vol = (1.0/D1) * (J - 1.0)**2 + (1.0/(D2 + 1e-12)) * (J - 1.0)**4 * jnp.where(D2 > 1e-9, 1.0, 0.0)
        U_dev = W_total - W_vol
        
    elif he_type == "VAN_DER_WAALS":
        fn = lambda f: van_der_waals_energy(f, params_padded)
        P = jax.grad(fn)(F)
        sigma = (1.0 / J) * jnp.dot(P, F.T)
        W_total = fn(F)
        D = jnp.where(params_padded[4] < 1e-9, 1e-5, params_padded[4])
        W_vol = (1.0 / D) * (0.5 * (J**2 - 1.0) - jnp.log(J))
        U_dev = W_total - W_vol
        
    else:
        J_safe = jnp.maximum(J, 1e-6)
        b = jnp.dot(F, F.T)
        b_bar = (J_safe ** (-2.0 / 3.0)) * b
        I1_bar = jnp.trace(b_bar)
        b_bar_dev = b_bar - (1.0 / 3.0) * I1_bar * jnp.eye(3)
        
        if "NEO" in he_type or "HOOK" in he_type:
            C10 = params_padded[0]
            D1_safe = jnp.where(params_padded[1] < 1e-9, 1e-5, params_padded[1])
            dW_dI1 = C10
            p = (2.0 / D1_safe) * (J - 1.0)
            U_dev = C10 * (I1_bar - 3.0)
        elif "YEOH" in he_type:
            C10 = params_padded[0]
            C20 = params_padded[1]
            C30 = params_padded[2]
            D1_safe = jnp.where(params_padded[3] < 1e-9, 1e-5, params_padded[3])
            term = I1_bar - 3.0
            dW_dI1 = C10 + 2.0 * C20 * term + 3.0 * C30 * (term ** 2)
            p = (2.0 / D1_safe) * (J - 1.0)
            U_dev = C10 * term + C20 * (term ** 2) + C30 * (term ** 3)
        elif "ARRUDA" in he_type or "BOYCE" in he_type:
            mu = params_padded[0]
            lambda_L = params_padded[1]
            D1_safe = jnp.where(params_padded[2] < 1e-9, 1e-5, params_padded[2])
            c1, c2, c3, c4, c5 = 1/2, 1/20, 11/1050, 19/7000, 519/673750
            dW_dI1 = mu * (c1 + 2.0*(c2/(lambda_L**2))*I1_bar + 3.0*(c3/(lambda_L**4))*(I1_bar**2) + 4.0*(c4/(lambda_L**6))*(I1_bar**3) + 5.0*(c5/(lambda_L**8))*(I1_bar**4))
            p = (1.0 / D1_safe) * (J - 1.0 / (J_safe))
            term1, term2, term3, term4, term5 = I1_bar-3, I1_bar**2-3**2, I1_bar**3-3**3, I1_bar**4-3**4, I1_bar**5-3**5
            U_dev = mu * (c1*term1 + (c2/(lambda_L**2))*term2 + (c3/(lambda_L**4))*term3 + (c4/(lambda_L**6))*term4 + (c5/(lambda_L**8))*term5)
        else:
            dW_dI1 = 0.5; p = 0.0; U_dev = 0.0
        
        sigma_dev = (2.0 / J_safe) * dW_dI1 * b_bar_dev
        sigma_vol = p * jnp.eye(3)
        sigma = sigma_dev + sigma_vol

    # Apply Mullins effect
    if mullins_params is not None:
        sigma_dev = sigma - jnp.trace(sigma)/3.0 * jnp.eye(3)
        sigma_vol = jnp.trace(sigma)/3.0 * jnp.eye(3)
        eta, U_dev_max_new = compute_mullins_damage(U_dev, U_dev_max, mullins_params[0], mullins_params[1], mullins_params[2])
        sigma = eta * sigma_dev + sigma_vol
    else:
        U_dev_max_new = U_dev_max
        
    return sigma, U_dev_max_new

def compute_cauchy_stress(he_model_or_type, F: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    return compute_hyperelastic_state(he_model_or_type, F, params)[0]
