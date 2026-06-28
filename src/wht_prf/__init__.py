import jax
# 고차 비선형 지수(n~12) 역전파 미분 폭주 방지를 위해 double precision(float64) 활성화
jax.config.update("jax_enable_x64", True)

def hello() -> str:
    return "Hello from wht-prf!"
