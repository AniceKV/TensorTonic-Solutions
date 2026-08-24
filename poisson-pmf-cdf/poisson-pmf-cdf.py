import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """Return the Poisson PMF at k and CDF through k."""
    def pos_pmf(lam_val,k):
        return math.pow((math.e),-lam_val)*  (math.pow(lam_val,k)/math.factorial(k))

    cdf=sum(pos_pmf(lam,i) for i in range(k+1))

    return {
        "pmf": pos_pmf(lam,k),
        "cdf":cdf
    }