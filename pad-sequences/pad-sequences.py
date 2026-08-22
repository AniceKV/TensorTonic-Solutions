import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    L=max_len or( 0 if not seqs else max(len(seq) for seq in seqs))
    res=[]
    for x in seqs:
        x=x[:L]
        x.extend([pad_value]*(L-len(x)))
        res.append(x)
    print(res)
    return np.array(res,dtype=int).reshape(len(seqs),L)