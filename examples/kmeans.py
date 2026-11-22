"""K-means clustering with comprehensions - no explicit distance matrix is built."""

import numpy as np

from parakeet import jit


def sqr_dist(x, y):
    return ((x - y) ** 2).sum()


@jit
def kmeans(X, k, niters=10):
    C = X[:k, :]
    for _ in range(niters):
        A = np.array([np.argmin([sqr_dist(x, c) for c in C]) for x in X])
        C = np.array([np.mean(X[A == i, :], axis=0) for i in range(k)])
    return C


def kmeans_numpy(X, k, niters=10):
    C = X[:k, :]
    for _ in range(niters):
        A = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        C = np.array([X[A == i].mean(axis=0) for i in range(k)])
    return C


def main(n: int = 2000, d: int = 16, k: int = 4):
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(k, d)) * 5
    X = centers[rng.integers(0, k, n)] + rng.normal(size=(n, d))
    C = kmeans(X, k, 15)
    assert np.allclose(C, kmeans_numpy(X, k, 15))
    return C


if __name__ == "__main__":
    print(main())
