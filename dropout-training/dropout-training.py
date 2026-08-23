import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x=np.array(x)
    if rng is not None:
        random_vals = rng.random(x.shape)
    else:
        random_vals = np.random.random(x.shape)

    mask=np.where(random_vals>=p,1,0)
    dropout_pattern=mask/(1-p)
    
    
    return x*dropout_pattern , dropout_pattern