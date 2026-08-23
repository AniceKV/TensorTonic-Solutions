import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """Return the Pearson correlation matrix of X."""
    
    covar=np.cov(X,rowvar=False)
    std = np.sqrt(np.diag(covar))
    corr = covar / np.outer(std, std)

    return corr