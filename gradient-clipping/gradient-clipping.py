import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """Return g clipped by its global L2 norm."""
    g=np.asarray(g)
    L2norm=np.linalg.norm(g)
    
    if L2norm<=max_norm:
        return g
    else:
        return g*(max_norm/L2norm)