"""Generate a synthetic, linearly separable binary dataset.

The data matches the shapes used on page 57:

* ``X``  : (n_x, m)  -- m training examples, each with n_x features.
* ``y``  : (1, m)    -- binary labels (0 or 1).
* ``w_true`` : (n_x, 1), ``b_true`` : (1, 1) -- the ground-truth parameters.

Two well-separated Gaussian blobs are placed on opposite sides of a random
hyperplane so that logistic regression can separate them.
"""

import os

import numpy as np

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")


def make_synthetic(n_x=2, m=200, separation=2.0, noise=0.5, seed=0):
    """Create the synthetic dataset.

    Args:
        n_x: number of features.
        m: number of examples.
        separation: distance of each class centre from the hyperplane.
        noise: standard deviation of the Gaussian noise.
        seed: random seed for reproducibility.

    Returns:
        dict with keys ``X``, ``y``, ``w_true``, ``b_true``.
    """
    rng = np.random.default_rng(seed)

    w_true = rng.normal(size=(n_x, 1))
    w_true /= np.linalg.norm(w_true)
    b_true = np.zeros((1, 1), dtype=np.float64)

    signs = rng.integers(0, 2, size=(1, m))
    signs = np.where(signs == 0, -1.0, 1.0)

    X = separation * w_true * signs + rng.normal(scale=noise, size=(n_x, m))
    y = ((signs + 1.0) / 2.0).astype(np.float64)  # back to {0, 1}

    return {"X": X, "y": y, "w_true": w_true, "b_true": b_true}


def save_npz(data, path=None):
    """Save a dataset dict to a compressed ``.npz`` file."""
    if path is None:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        path = os.path.join(DEFAULT_OUT_DIR, "synthetic.npz")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **data)
    return path


def save_csv(data, path=None):
    """Save the dataset as a CSV with one row per example.

    Layout: feature_0, feature_1, ..., feature_{n_x-1}, label
    """
    if path is None:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        path = os.path.join(DEFAULT_OUT_DIR, "synthetic.csv")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    X = np.asarray(data["X"])
    y = np.asarray(data["y"]).reshape(1, -1)
    rows = np.vstack([X, y]).T  # (m, n_x + 1)

    header = ",".join([f"feature_{i}" for i in range(X.shape[0])] + ["label"])
    np.savetxt(path, rows, delimiter=",", header=header, comments="")
    return path


def load_npz(path=None):
    """Load a previously saved synthetic dataset."""
    if path is None:
        path = os.path.join(DEFAULT_OUT_DIR, "synthetic.npz")
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def main():
    data = make_synthetic(n_x=2, m=200, seed=0)
    npz_path = save_npz(data)
    csv_path = save_csv(data)
    print(f"saved: {npz_path}")
    print(f"saved: {csv_path}")
    print("X shape      :", data["X"].shape)
    print("y shape      :", data["y"].shape)
    print("class balance:", np.bincount(data["y"].reshape(-1).astype(int)).tolist())


if __name__ == "__main__":
    main()
