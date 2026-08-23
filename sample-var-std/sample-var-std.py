import numpy as np

def sample_var_std(x: list) -> dict:
    """Return unbiased sample variance and standard deviation."""
    variance=np.var(np.array(x),ddof=1)
    std=np.sqrt(variance)

    return {
        "variance": float(variance),
        "standard_deviation": float(std)
    }
    pass