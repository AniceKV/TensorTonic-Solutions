import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """Return the real eigenvalues in ascending order."""
    matrix=np.asarray(matrix,dtype=float)
    eigenvals,_=np.linalg.eig(matrix) 

    return np.array(sorted(eigenvals.real))
    

    