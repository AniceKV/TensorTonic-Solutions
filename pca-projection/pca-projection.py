import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    
    X=np.asarray(X)
    n=X.shape[0]
    Xmean=np.mean(X,axis=0,keepdims=True)
    X=X-Xmean

    C=np.matmul(X.T,X)/(n-1)

    eigvals,eigvecs=np.linalg.eig(C)
    idx=np.argsort(eigvals)[::-1]
    W=eigvecs[:,idx][:,:k]
    
    return np.matmul(X,W)