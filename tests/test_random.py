"""np.random in compiled code: distributions, shapes and reproducibility."""

import numpy as np
import pytest

import parakeet
from parakeet import jit


@jit
def draws(n):
    return (
        np.random.rand(n), np.random.rand(3, n), np.random.random(n), np.random.uniform(2.0, 5.0, n),
        np.random.randn(n), np.random.normal(10.0, 2.0, size=(n,)), np.random.randint(3, 9, size=n),
        np.random.randint(5, size=(2, n)), np.random.rand(), np.random.randint(10), np.random.normal(),
    )  # fmt: skip


def test_shapes_types_and_ranges():
    n = 50_000
    u, u2, r, uni, z, norm, ints, ints2, s, si, sn = draws(n)
    assert u.shape == (n,) and u2.shape == (3, n) and ints2.shape == (2, n)
    assert u.dtype == np.float64 and ints.dtype == np.int64
    assert type(s) is float and type(si) is int and type(sn) is float
    assert 0 <= u.min() and u.max() < 1 and 0 <= r.min() and r.max() < 1
    assert abs(u.mean() - 0.5) < 0.01 and abs(u.var() - 1 / 12) < 0.01
    assert 2 <= uni.min() and uni.max() < 5 and abs(uni.mean() - 3.5) < 0.03
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1) < 0.02
    assert abs(norm.mean() - 10) < 0.05 and abs(norm.std() - 2) < 0.05
    assert ints.min() == 3 and ints.max() == 8 and set(np.unique(ints2)) == {0, 1, 2, 3, 4}
    assert 0 <= s < 1 and 0 <= si < 10
    # independent streams: no two draws of the same call are identical or correlated
    assert not np.array_equal(u, r) and abs(np.corrcoef(u, r)[0, 1]) < 0.02
    assert abs(np.corrcoef(u[:-1], u[1:])[0, 1]) < 0.02


def test_seed_makes_results_reproducible():
    parakeet.seed(123)
    a = draws(100)
    b = draws(100)
    parakeet.seed(123)
    a2 = draws(100)
    parakeet.seed(124)
    c = draws(100)
    assert all(np.array_equal(x, y) for x, y in zip(a, a2))
    assert not np.array_equal(a[0], b[0]) and not np.array_equal(a[0], c[0])


def test_results_do_not_depend_on_backend_or_thread_count():
    @jit
    def f(n):
        return np.random.rand(n) + np.random.randn(n)

    parakeet.seed(7)
    a = f(200_000, _backend="c")
    parakeet.seed(7)
    b = f(200_000, _backend="multicore")
    assert np.array_equal(a, b)


def test_permutation_and_shuffle():
    @jit
    def f(n):
        p = np.random.permutation(n)
        x = np.arange(n) * 2.0
        np.random.shuffle(x)
        return p, x

    p, x = f(1000)
    assert sorted(p) == list(range(1000)) and not np.array_equal(p, np.arange(1000))
    assert sorted(x) == [2.0 * i for i in range(1000)] and not np.array_equal(x, np.arange(1000) * 2.0)


def test_random_inside_parallel_map():
    @jit
    def noisy(x):
        return parakeet.map(lambda v: v + np.random.rand(), x)

    x = np.zeros(100_000)
    r = noisy(x)
    assert 0 <= r.min() and r.max() < 1 and len(np.unique(r)) > 99_000


def test_errors():
    @jit
    def bad_range():
        return np.random.randint(5, 5)

    @jit
    def reseed():
        np.random.seed(1)

    with pytest.raises(ValueError, match="low >= high"):
        bad_range()
    with pytest.raises(parakeet.UnsupportedError, match="parakeet.seed"):
        reseed()
