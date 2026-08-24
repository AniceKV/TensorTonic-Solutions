import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """Return the one-sample t-statistic."""
    x=np.array(x)
    n=len(x)
    std=np.sqrt(np.var(x,ddof=1))
    mean=np.mean(x,axis=0)
    
    return  float((mean-mu0)/(std/(n**0.5))) 
    pass