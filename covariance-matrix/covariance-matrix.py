import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """Return the sample covariance matrix of X."""

    X=np.array(X)
    N=len(X)

    XC=(X-np.mean(X,axis=0,keepdims=True))

    return (XC.T @ XC)/(N-1)


