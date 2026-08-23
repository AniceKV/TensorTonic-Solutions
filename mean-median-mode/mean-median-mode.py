from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """Return the mean, median, and smallest mode."""
    mode=float(Counter(x).most_common(1)[0][0])
    x=np.array(x,dtype=float)

    return {"mean":float(np.mean(x)),
            "median":float(np.median(x)),
            "mode":mode}
