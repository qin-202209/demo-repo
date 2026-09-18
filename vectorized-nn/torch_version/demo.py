"""Demo: PyTorch vs NumPy vectorized forward pass (page 57).

Run with::

    python torch_version/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.make_synthetic import make_synthetic
from numpy_version.forward import forward_vectorized
from torch_version.model import LogisticModel, forward_torch


def main():
    np.set_printoptions(precision=4, suppress=True)

    data = make_synthetic(n_x=3, m=5, seed=1)
    X, w = data["X"], data["w_true"]
    b = float(data["b_true"].reshape(-1)[0])

    Z_np, A_np = forward_vectorized(X, w, b)
    Z_t, A_t = forward_torch(X, w, b)

    model = LogisticModel(n_x=X.shape[0]).load_from_numpy(w, b)
    A_module = model(X).detach().numpy()

    print("NumPy  A:\n", A_np)
    print("Torch  A:\n", A_t.numpy())
    print("Module A:\n", A_module)
    print("allclose(Z_np, Z_t)  :", torch.allclose(torch.as_tensor(Z_np), Z_t))
    print("allclose(A_np, A_t)  :", torch.allclose(torch.as_tensor(A_np), A_t))
    print("allclose(A_np, module):", np.allclose(A_np, A_module))


if __name__ == "__main__":
    main()
