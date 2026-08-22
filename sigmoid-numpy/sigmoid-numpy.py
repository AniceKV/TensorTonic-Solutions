import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Vectorized sigmoid function.
    """
    is_scalar=isinstance(x,float)
    x=np.array(x)
    
    result=np.where(x>=0, (1/(1+np.exp(-x))), (np.exp(x)/(1+np.exp(x))))

    if is_scalar:
        return float(result)
        
    return result