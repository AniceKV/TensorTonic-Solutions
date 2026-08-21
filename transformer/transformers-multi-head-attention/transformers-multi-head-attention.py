import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads):

    batch_size = Q.shape[0]

    d_k = Q.shape[-1] // num_heads
    d_v = V.shape[-1] // num_heads

    Q_proj = np.dot(Q, W_q)
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)

    Q_proj = Q_proj.reshape(
        Q.shape[0], Q.shape[1], num_heads, d_k
    )

    K_proj = K_proj.reshape(
        K.shape[0], K.shape[1], num_heads, d_k
    )

    V_proj = V_proj.reshape(
        V.shape[0], V.shape[1], num_heads, d_v
    )

    Q_proj = Q_proj.transpose(0, 2, 1, 3)
    K_proj = K_proj.transpose(0, 2, 1, 3)
    V_proj = V_proj.transpose(0, 2, 1, 3)

    scores = Q_proj @ K_proj.transpose(0, 1, 3, 2)

    scores = scores / np.sqrt(d_k)

    weights = softmax(scores, axis=-1)

    attention = weights @ V_proj

    attention = attention.transpose(0, 2, 1, 3)

    attention = attention.reshape(
        attention.shape[0],
        attention.shape[1],
        num_heads * d_v
    )

    return attention @ W_o