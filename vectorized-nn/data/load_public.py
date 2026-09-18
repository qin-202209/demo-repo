"""Load public datasets and shape them like the page-57 notation.

All loaders return a dict with:

* ``X`` : (n_x, m) feature matrix (float64)
* ``y`` : (1, m) binary labels (0 or 1)

Datasets
--------
* ``load_breast_cancer`` -- offline via scikit-learn (30 features, 2 classes).
* ``load_digits``        -- offline via scikit-learn (8x8 images -> 64 features).
* ``load_mnist_binary``  -- via torchvision, digit 0 vs 1 (784 features).
  Requires an internet connection the first time it is called.
"""

import numpy as np


def _as_columns(X, y):
    return np.asarray(X, dtype=np.float64).T, np.asarray(y, dtype=np.float64).reshape(1, -1)


def load_breast_cancer():
    """Breast cancer Wisconsin dataset (binary, 30 features)."""
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer()
    X, y = _as_columns(bunch.data, bunch.target)
    return {"X": X, "y": y, "name": "breast_cancer"}


def load_digits(class_a=0, class_b=1):
    """8x8 handwritten digits, keeping only two classes.

    Args:
        class_a, class_b: the two digit classes to keep (labels become 0 and 1).
    """
    from sklearn.datasets import load_digits as _load_digits

    bunch = _load_digits()
    mask = (bunch.target == class_a) | (bunch.target == class_b)
    X = bunch.data[mask] / 16.0  # raw pixels are 0..16
    y = (bunch.target[mask] == class_b).astype(np.float64)
    X, y = _as_columns(X, y)
    return {"X": X, "y": y, "name": f"digits_{class_a}_vs_{class_b}"}


def load_mnist_binary(class_a=0, class_b=1, root=None):
    """MNIST restricted to two classes, flattened to (784, m).

    Downloads the dataset on first use. Requires ``torchvision``.
    """
    import os

    from torchvision import datasets, transforms

    if root is None:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated", "mnist")

    to_tensor = transforms.ToTensor()
    train = datasets.MNIST(root=root, train=True, download=True, transform=to_tensor)
    test = datasets.MNIST(root=root, train=False, download=True, transform=to_tensor)

    xs, ys = [], []
    for dataset in (train, test):
        for image, label in dataset:
            if label in (class_a, class_b):
                xs.append(image.numpy().reshape(-1))
                ys.append(1.0 if label == class_b else 0.0)

    X, y = _as_columns(np.stack(xs), np.asarray(ys))
    return {"X": X, "y": y, "name": f"mnist_{class_a}_vs_{class_b}"}


def main():
    for loader in (load_breast_cancer, load_digits):
        data = loader()
        print(f"{data['name']:>14}: X={data['X'].shape}, y={data['y'].shape}")


if __name__ == "__main__":
    main()
