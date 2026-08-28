import numpy as np

def tanh(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    
    x=np.asarray(x)
    ex=np.exp(x)
    e_x=np.exp(-x)

    return (ex-e_x)/(ex+e_x)
