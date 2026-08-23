import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """Return the Binomial PMF at k and CDF through k."""
    pdf=math.comb(n,k)*(p**k)*(1-p)**(n-k)
    cdf=sum(math.comb(n,i)*(p**i)*(1-p)**(n-i) for i in range(k+1))

    return {
        'pmf':float(pdf),
        'cdf':float(cdf)
    }