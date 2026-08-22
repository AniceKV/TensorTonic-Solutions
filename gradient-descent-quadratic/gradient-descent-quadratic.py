def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Return final x after 'steps' iterations.
    """
    for _ in range(steps):
        dx=2*a*x0+b 
        x0=x0-dx*lr 

    return x0
