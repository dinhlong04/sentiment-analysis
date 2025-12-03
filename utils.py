# utils.py
import numpy as np
from typing import Union

def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Computes the sigmoid of z.

    Args:
        z (float or np.ndarray): Scalar or array input.

    Returns:
        float or np.ndarray: Sigmoid value between 0 and 1.
    """
    h = 1 / (1 + np.exp(-z))
    return h

def cost_function(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Computes the binary cross-entropy cost.

    Args:
        y_hat (np.ndarray): The predicted probabilities (m, 1).
        y (np.ndarray): The true labels (m, 1).

    Returns:
        float: The cost value.
    """
    m = len(y)
    epsilon = 1e-15  # Prevent log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    
    J = (-1 / m) * (y.T @ np.log(y_hat) + (1 - y).T @ np.log(1 - y_hat))
    return float(J)