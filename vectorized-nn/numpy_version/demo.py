"""Demo: loop-based vs vectorized forward pass (page 57).

Run with::

    python numpy_version/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.load_public import load_breast_cancer
from data.make_synthetic import make_synthetic
from numpy_version.forward import (
    describe_broadcast,
    forward_loop,
    forward_vectorized,
)


def _check(X, w, b, title):
    Z_loop, A_loop = forward_loop(X, w, b)
    Z_vec, A_vec = forward_vectorized(X, w, b)

    print(f"--- {title} ---")
    print("X shape          :", X.shape)
    for key, shape in describe_broadcast(X, w, b).items():
        print(f"  {key:<14}: {shape}")
    print("Z shape          :", Z_vec.shape)
    print("A shape          :", A_vec.shape)
    print("max |Z_loop - Z_vec| :", float(np.max(np.abs(Z_loop - Z_vec))))
    print("max |A_loop - A_vec| :", float(np.max(np.abs(A_loop - A_vec))))
    print()


def main():
    np.set_printoptions(precision=4, suppress=True)

    data = make_synthetic(n_x=3, m=5, seed=1)
    b_true = float(data["b_true"].reshape(-1)[0])
    _check(data["X"], data["w_true"], b_true, "synthetic (5 examples)")

    public = load_breast_cancer()
    rng = np.random.default_rng(0)
    w = rng.normal(size=(public["X"].shape[0], 1))
    _check(public["X"][:, :8], w, 0.0, "breast_cancer (first 8 examples)")


if __name__ == "__main__":
    main()
