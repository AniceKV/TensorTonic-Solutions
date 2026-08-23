import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """Return the cosine similarity of a and b."""
    a=np.array(a)
    b=np.array(b)
    moda=np.linalg.norm(a)
    modb=np.linalg.norm(b)
    if moda==0 or modb==0:
        return float(0)
    a=a/moda 
    b=b/modb

    return float(np.dot(a,b))
    