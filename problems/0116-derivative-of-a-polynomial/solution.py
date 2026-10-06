def poly_term_derivative(c: float, x: float, n: float) -> float:
    
    derivate_polynom = n * (x ** (n-1)) if n != 0 else 0

    return c * derivate_polynom
