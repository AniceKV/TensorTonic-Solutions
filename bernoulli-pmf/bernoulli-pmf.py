import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """Return the Bernoulli PMF, mean, and variance."""
    x=np.array(x,dtype=float)

    pmf=np.where(x>0.5,p,1-p) 
    men=np.mean(x)
    var=np.var(x) 

    return {
        "pmf":pmf,
        "mean":float(p),
        "variance":float(p*(1-p))
    }
    
    