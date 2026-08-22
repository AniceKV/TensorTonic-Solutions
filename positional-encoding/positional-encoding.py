import numpy as np

def positional_encoding(
    seq_len: int,
    d_model: int,
    base: float = 10000.0
) -> np.ndarray:
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    positions = np.arange(seq_len, dtype=float)

    n_even = (d_model + 1) // 2
    n_odd = d_model // 2

    i = np.arange(n_even, dtype=float)

    div = np.power(base, (2 * i) / d_model)

    angles = positions[:, None] / div[None, :]

    pe = np.zeros((seq_len, d_model), dtype=float)

    pe[:, 0::2] = np.sin(angles[:, :n_even])
    pe[:, 1::2] = np.cos(angles[:, :n_odd])

    return pe