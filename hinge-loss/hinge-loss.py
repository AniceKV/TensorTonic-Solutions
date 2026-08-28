import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    
    """
    n=len(y_score)
    y_true=np.asarray(y_true)
    y_score=np.asarray(y_score)
    vals=margin-y_true*y_score

    loss=np.maximum(0,vals)

    if reduction=="mean":
        return float(np.mean(loss)) 
    else:
        return float(np.sum(loss))
    
    