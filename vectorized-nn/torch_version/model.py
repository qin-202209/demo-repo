"""PyTorch equivalent of the page-57 vectorized forward pass.

Two flavours are provided:

* ``forward_torch``  -- literal tensor translation of ``Z = w.T @ X + b``.
* ``LogisticModel``  -- the idiomatic ``nn.Linear`` + ``nn.Sigmoid`` module.

Both must produce exactly the same numbers as the NumPy implementation, which
is verified by ``tests/test_vectorization.py``.
"""

import numpy as np
import torch


def forward_torch(X, w, b):
    """Tensor version of ``Z = np.dot(w.T, X) + b`` followed by sigmoid.

    Args:
        X: feature matrix of shape (n_x, m).
        w: weight column vector of shape (n_x, 1).
        b: bias scalar.

    Returns:
        (Z, A) torch tensors of shape (1, m), dtype float64.
    """
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    w_t = torch.as_tensor(np.asarray(w), dtype=torch.float64)
    b_t = torch.as_tensor(b, dtype=torch.float64)

    Z = w_t.t() @ X_t + b_t
    A = torch.sigmoid(Z)
    return Z, A


class LogisticModel(torch.nn.Module):
    """Idiomatic equivalent: a single linear layer with a sigmoid.

    ``nn.Linear`` consumes a batch of shape (m, n_x), so inputs are transposed
    on the way in and outputs are transposed back to (1, m).
    """

    def __init__(self, n_x):
        super().__init__()
        self.linear = torch.nn.Linear(n_x, 1, dtype=torch.float64)

    def forward(self, X):
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float64)
        Z = self.linear(X_t.t()).t()
        return torch.sigmoid(Z)

    @torch.no_grad()
    def load_from_numpy(self, w, b):
        """Copy NumPy weights/bias into the underlying linear layer."""
        self.linear.weight.copy_(
            torch.as_tensor(np.asarray(w).T, dtype=torch.float64)
        )
        self.linear.bias.copy_(
            torch.as_tensor(np.asarray(b).reshape(-1), dtype=torch.float64)
        )
        return self
