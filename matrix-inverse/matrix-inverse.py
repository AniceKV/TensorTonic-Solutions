import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    A=np.asarray(A)

    aug=np.hstack((A,np.eye(A.shape[0])))

    size=aug.shape[0]

    for column in range(size):
        pivot=column+np.argmax(np.abs(aug[column:,column]))
        if abs(aug[pivot,column])<1e-10:
            return None 
        aug[[column,pivot]]=aug[[pivot,column]]
        aug[column]/=aug[column,column]

        for row in range(size):
            if row!=column:
                aug[row]=aug[row]-aug[row,column]*aug[column]
                
    return aug[:,size:]
    