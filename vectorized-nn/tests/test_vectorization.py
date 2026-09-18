"""Equivalence tests for the page-57 vectorized forward pass."""

import numpy as np
import pytest
import torch

from data.load_public import load_breast_cancer, load_digits
from data.make_synthetic import make_synthetic
from numpy_version.forward import forward_loop, forward_vectorized
from numpy_version.sigmoid import sigmoid
from torch_version.model import LogisticModel, forward_torch


def _random_problem(n_x=4, m=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_x, m))
    w = rng.normal(size=(n_x, 1))
    b = float(rng.normal())
    return X, w, b


def test_forward_loop_matches_vectorized():
    X, w, b = _random_problem()
    Z_loop, A_loop = forward_loop(X, w, b)
    Z_vec, A_vec = forward_vectorized(X, w, b)

    assert Z_vec.shape == (1, X.shape[1])
    assert A_vec.shape == (1, X.shape[1])
    np.testing.assert_allclose(Z_loop, Z_vec, atol=1e-12)
    np.testing.assert_allclose(A_loop, A_vec, atol=1e-12)


def test_torch_matches_numpy():
    X, w, b = _random_problem(seed=1)
    Z_np, A_np = forward_vectorized(X, w, b)
    Z_t, A_t = forward_torch(X, w, b)

    assert torch.allclose(torch.as_tensor(Z_np), Z_t)
    assert torch.allclose(torch.as_tensor(A_np), A_t)


def test_torch_module_matches_numpy():
    X, w, b = _random_problem(seed=2)
    _, A_np = forward_vectorized(X, w, b)
    model = LogisticModel(n_x=X.shape[0]).load_from_numpy(w, b)
    A_module = model(X).detach().numpy()

    assert A_module.shape == A_np.shape
    np.testing.assert_allclose(A_module, A_np, atol=1e-12)


def test_sigmoid_is_numerically_stable():
    z = np.array([[-1000.0, -50.0, 0.0, 50.0, 1000.0]])
    out = sigmoid(z)

    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))
    assert out[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert out[0, 4] == pytest.approx(1.0, abs=1e-12)
    assert out[0, 2] == pytest.approx(0.5)


def test_bias_is_broadcast_over_examples():
    X, w, _ = _random_problem(m=7)
    b = 1.5
    Z, _ = forward_vectorized(X, w, b)
    Z_no_bias, _ = forward_vectorized(X, w, 0.0)

    np.testing.assert_allclose(Z - Z_no_bias, np.full((1, 7), b), atol=1e-12)


def test_synthetic_dataset_shapes():
    data = make_synthetic(n_x=5, m=40, seed=3)

    assert data["X"].shape == (5, 40)
    assert data["y"].shape == (1, 40)
    assert data["w_true"].shape == (5, 1)
    assert set(np.unique(data["y"]).tolist()) <= {0.0, 1.0}

    z = data["w_true"].T @ data["X"]
    predictions = (z > 0).astype(float)
    assert np.mean(predictions == data["y"]) > 0.9


@pytest.mark.parametrize("loader", [load_breast_cancer, load_digits])
def test_public_loaders_return_columns(loader):
    data = loader()
    n_x, m = data["X"].shape

    assert data["y"].shape == (1, m)
    assert set(np.unique(data["y"]).tolist()) == {0.0, 1.0}
    assert n_x > 0 and m > 0
