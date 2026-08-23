import numpy as np

def dot_product(x: list, y: list) -> float:
    """Return the dot product of x and y."""
    return float(np.dot(np.array(x,dtype=float),np.array(y,dtype=float)))
    pass