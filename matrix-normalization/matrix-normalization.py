import numpy as np

def matrix_normalization(matrix, axis=None, norm_type="l2"):
    matrix = np.asarray(matrix, dtype=float)
    if norm_type == "l1":
        norms = np.sum(np.abs(matrix), axis=axis, keepdims=True)
    elif norm_type == "l2":
        norms = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
    else:
        norms = np.max(np.abs(matrix), axis=axis, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return matrix / safe_norms
