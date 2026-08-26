import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    rng = np.random.default_rng(seed)
    x=np.asarray(x)
    
    index_picker=rng.integers(0,x.size , size=(n_bootstrap,x.size))

    means=x[index_picker].mean(axis=1)

    lower_qunatile=(1-ci)/2
    upper_quantile=1-lower_qunatile 

    lower,upper=np.quantile(means,[lower_qunatile,upper_quantile])
    
    return {
        "bootstrap_mean":sum(means)/n_bootstrap,
        "lower":lower,
        "upper":upper
    }