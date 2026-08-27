import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    y_pred=np.array(y_pred)
    y_true=np.array(y_true)
    
    return ((np.linalg.norm(y_true-y_pred))**2)/y_pred.shape[0]
    # Write code here
    pass