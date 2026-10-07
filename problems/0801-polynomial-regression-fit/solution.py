import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    X = np.vander(x, degree + 1, increasing = True)

    c = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

    return c.tolist()