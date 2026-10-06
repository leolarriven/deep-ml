import numpy as np


def binary_cross_entropy(
    y_true: list[float],
    y_pred: list[float],
    epsilon: float = 1e-15
) -> float:
    """
    Compute binary cross-entropy loss.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)
        epsilon: Small value for numerical stability

    Returns:
        Mean binary cross-entropy loss
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    som = 0

    for position in range(len(y_pred)):
        som += (
            y_true[position] * np.log(y_pred[position])
            + (1 - y_true[position]) * np.log(1 - y_pred[position])
        )

    return -(som / len(y_pred))
