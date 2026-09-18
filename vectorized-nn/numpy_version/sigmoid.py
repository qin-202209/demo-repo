"""Numerically stable sigmoid activation.

This mirrors the sigmoid function used on page 57 of the notes:
the forward pass applies the element-wise sigmoid to the whole matrix Z,
so that all m training examples are activated in a single call.
"""

import numpy as np


def sigmoid(z):
    """Element-wise sigmoid that avoids overflow for large |z|.

    Args:
        z: array-like (typically the (1, m) matrix Z) or a plain scalar.

    Returns:
        An array with the same shape as ``z`` (or a numpy scalar for 0-d input)
        containing ``1 / (1 + exp(-z))`` computed in a numerically stable way.
    """
    z = np.asarray(z, dtype=np.float64)

    if z.ndim == 0:
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        exp_z = np.exp(z)
        return exp_z / (1.0 + exp_z)

    out = np.empty_like(z)
    positive = z >= 0
    negative = ~positive

    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[negative])
    out[negative] = exp_z / (1.0 + exp_z)
    return out
