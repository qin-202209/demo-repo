"""Vectorized forward propagation for logistic regression.

Page 57 of the notes shows that, instead of looping over the m training
examples, we can stack the examples in a matrix X of shape (n_x, m) and
compute the forward pass for all examples at once:

    Z = np.dot(w.T, X) + b
    A = sigma(Z)

where ``w`` has shape (n_x, 1), ``b`` is a scalar and broadcasting expands
``b`` into the (1, m) row vector ``[b, b, ..., b]``.

Both a loop version (the "before" picture) and the vectorized version (the
"after" picture) are provided so their outputs can be compared.
"""

import numpy as np

from numpy_version.sigmoid import sigmoid


def forward_loop(X, w, b):
    """Reference implementation: one training example at a time.

    This is the naive ``for`` loop described before the vectorization trick.

    Args:
        X: feature matrix of shape (n_x, m).
        w: weight column vector of shape (n_x, 1).
        b: bias scalar.

    Returns:
        (Z, A) each of shape (1, m).
    """
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n_x, m = X.shape

    Z = np.zeros((1, m))
    for i in range(m):
        x_i = X[:, i].reshape(n_x, 1)
        Z[0, i] = np.dot(w.T, x_i).item() + b

    A = sigmoid(Z)
    return Z, A


def forward_vectorized(X, w, b):
    """Vectorized implementation straight from page 57.

    Args:
        X: feature matrix of shape (n_x, m).
        w: weight column vector of shape (n_x, 1).
        b: bias scalar (broadcast across all m examples).

    Returns:
        (Z, A) each of shape (1, m).
    """
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    Z = np.dot(w.T, X) + b  # b is broadcast to shape (1, m)
    A = sigmoid(Z)
    return Z, A


def describe_broadcast(X, w, b):
    """Return a small report showing how ``b`` is broadcast.

    Useful for the demo: prints the shapes involved so the broadcasting rule
    ``(1, 1) + scalar -> (1, m)`` is explicit.
    """
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n_x, m = X.shape
    b_array = np.asarray(b, dtype=np.float64)

    return {
        "X": (n_x, m),
        "w": w.shape,
        "w.T @ X": (np.dot(w.T, X)).shape,
        "b_before": b_array.shape,
        "b_broadcast": np.broadcast_to(b_array, (1, m)).shape,
        "Z": (np.dot(w.T, X) + b).shape,
    }
