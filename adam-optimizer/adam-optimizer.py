import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    grad=np.asarray(grad)
    param=np.asarray(param)
    m,v=np.asarray(m),np.asarray(v)

    newm=m*beta1+(1-beta1)*grad 

    newv= v*beta2 + (1-beta2)*(grad**2) 

    newmcap=newm/(1-beta1**t)
    newvcap=newv/(1-beta2**t)

    param=param - lr*(newmcap/(np.sqrt(newvcap)+eps))

    return param,newm,newv