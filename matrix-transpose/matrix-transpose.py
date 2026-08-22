import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A=np.asarray(A)
    n,m=A.shape
    transpose= np.zeros((m, n), dtype=A.dtype)
    for i in range(n):
        for j in range(m):
            transpose[j,i]=A[i,j]
    
    return np.asarray(transpose)