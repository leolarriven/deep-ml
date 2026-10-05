import numpy as np

def prelu_forward(x: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """
    Implements the forward pass of PReLU.
    Args:
        x: Input array of any shape
        alpha: Slope parameter for negative values (default: 0.25)
    Returns:
        np.ndarray: PReLU activation output, same shape as x
    """
    return np.where(x>0, x, alpha * x)


def prelu_backward(x: np.ndarray, alpha: float, grad_output: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Implements the backward pass of PReLU, computing gradients for both x and alpha.
    Args:
        x: Original input from forward pass
        alpha: Slope parameter used in forward pass
        grad_output: Upstream gradient, same shape as x
    Returns:
        grad_x: Gradient w.r.t. input x, same shape as x
        grad_alpha: Gradient w.r.t. alpha (scalar, summed over all elements)
    """
    grad_x = grad_output * np.where(x > 0, 1.0, alpha)

    grad_alpha = np.sum(grad_output[x <= 0] * x[x <= 0])

    return grad_x, float(grad_alpha)
