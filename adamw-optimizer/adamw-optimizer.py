import numpy as np

def adamw_step(w: list, m: list, v: list, grad: list, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    w=np.asarray(w)
    m=np.asarray(m)
    v=np.asarray(v)
    grad=np.asarray(grad)

    m_new=beta1*(m)+(1-beta1)*grad 
    v_new=beta2*(v) + (1-beta2)*(grad**2)

    new_w=w -(lr*(m_new))/(np.sqrt(v_new)+eps) - lr*weight_decay*w

    return {
        'new_w':new_w,
        'new_v':v_new,
        'new_m':m_new
    }
    
    