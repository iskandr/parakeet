"""The general forms of the operators: any rank, any number of arrays."""

import numpy as np
import pytest

import parakeet as pk
from parakeet import jit

RNG = np.random.default_rng(0)
A = RNG.random((3, 4, 5))
B = RNG.random((3, 5, 6))
M = RNG.random((5, 6))
V = RNG.random(5)
C = RNG.random((2, 3, 4, 5))


def same(f, *args):
  got, want = jit(f)(*args), f(*args)
  assert np.shape(got) == np.shape(want)
  np.testing.assert_allclose(got, want, rtol=1e-10)


def test_matmul_of_stacks():
  same(lambda a, b: a @ b, A, B)
  same(lambda a, b: a @ b, A, M)
  same(lambda a, b: a @ b, A, V)
  same(lambda a, b: a @ b, RNG.random((4, 5)), B)
  same(lambda a, b: a @ b, V, B)
  same(lambda a, b: a @ b, C, B)  # the stacks broadcast
  same(lambda a, b: np.dot(a, b), A, M)


def test_stacked_matmul_is_tiled_and_handles_odd_shapes():
  f = jit(lambda a, b: a @ b)
  assert "tile" in f.specialize(A, B).ir
  for n in (0, 1, 2, 5):
    for m in (0, 1, 3, 4, 9):
      a, b = RNG.random((n, 3, 7, 5)), RNG.random((n, 3, 5, m))
      np.testing.assert_allclose(f(a, b), a @ b)


def test_einsum():
  same(lambda a, b: np.einsum("ij,jk->ik", a, b), M.T.copy(), M)
  same(lambda a, b: np.einsum("ij,jk", a, b), M.T.copy(), M)
  same(lambda a, b: np.einsum("bij,bjk->bik", a, b), A, B)
  same(lambda a, b: np.einsum("bij,bij->b", a, b), A, A)
  same(lambda a, b: np.einsum("i,j->ij", a, b), V, V)
  same(lambda a, b, c: np.einsum("ij,jk,k->i", a, b, c), M.T.copy(), M, RNG.random(6))
  same(lambda a: np.einsum("ijk->kji", a), A)
  same(lambda a: np.einsum("ij->", a), M)


def test_einsum_errors():
  with pytest.raises(pk.UnsupportedError, match="ellipsis"):
    jit(lambda a: np.einsum("...i->...", a))(M)
  with pytest.raises(pk.UnsupportedError, match="diagonal"):
    jit(lambda a: np.einsum("ii->i", a))(np.eye(3))
  with pytest.raises(pk.TypeInferenceError, match="do not match an operand"):
    jit(lambda a: np.einsum("ijk->i", a))(M)


def test_reductions_over_several_axes():
  same(lambda a: np.sum(a, axis=(0, 2)), C[0])
  same(lambda a: np.mean(a, axis=(1, 2), keepdims=True), C[0])
  same(lambda a: np.max(a, axis=(1, 3)), C)
  same(lambda a: np.min(a, axis=(0, 1, 2)), C)


def test_adverbs_over_several_arrays():
  x, y = np.arange(1.0, 7.0), np.arange(6.0)[::-1].copy()
  X = np.arange(12.0).reshape(3, 4)
  assert pk.reduce(pk.add, x, y, map=pk.multiply) == x @ y
  np.testing.assert_allclose(pk.reduce(pk.add, X, X, map=lambda a, b: a * b, axis=0), (X * X).sum(0))
  np.testing.assert_allclose(pk.scan(pk.add, x, y, map=lambda a, b: a - b), np.cumsum(x - y))
  np.testing.assert_allclose(pk.scan(pk.add, x, emit=lambda a: a * 0.5), np.cumsum(x) * 0.5)
  np.testing.assert_allclose(pk.filter(lambda a, b: a > b, x, y, map=lambda a, b: a + b), (x + y)[x > y])
  # the old positional init still works
  assert pk.reduce(pk.add, x, 10.0) == pk.reduce(pk.add, x, init=10.0) == 31.0
  with pytest.raises(pk.TypeInferenceError, match="needs map="):
    pk.reduce(pk.add, x, y)
  with pytest.raises(pk.TypeInferenceError, match="needs map="):
    pk.filter(lambda a, b: a > b, x, y)


@pytest.mark.parametrize("n", [0, 1, 2, 7, 100, 50_000])
def test_parallel_scan_compaction_and_argmin(n):
  rng = np.random.default_rng(n)
  x, xi = rng.random(n), rng.integers(-5, 5, n)
  mask = x > 0.5
  # a tiny grain cuts even short arrays into several chunks
  with pk.options(parallel_grain=1 if n < 1000 else 8192, num_threads=4):
    np.testing.assert_allclose(jit(lambda a: np.cumsum(a))(x), np.cumsum(x))
    np.testing.assert_array_equal(jit(lambda a: np.cumsum(a))(xi), np.cumsum(xi))
    np.testing.assert_allclose(pk.scan(pk.add, x, init=5.0), 5.0 + np.cumsum(x))
    np.testing.assert_allclose(pk.scan(pk.maximum, x), np.maximum.accumulate(x) if n else x)
    np.testing.assert_array_equal(jit(lambda a: np.nonzero(a)[0])(mask), np.nonzero(mask)[0])
    np.testing.assert_array_equal(jit(lambda a, m: a[m])(x, mask), x[mask])
    if n:
      # ties: the first one wins, across chunk boundaries too
      assert jit(lambda a: np.argmin(a))(xi) == np.argmin(xi)
      assert jit(lambda a: np.argmax(a))(xi) == np.argmax(xi)
      with_nan = x.copy()
      with_nan[n // 2] = np.nan
      assert jit(lambda a: np.argmax(a))(with_nan) == np.argmax(with_nan)


def test_device_launch_goes_by_work_not_by_count():
  ir = jit(lambda x, c: pk.allpairs(lambda u, v: np.sum((u - v) ** 2), x, c)).specialize(A[0], A[1]).function
  from parakeet.syntax.ir import ParLoop, walk_stmts

  loops = [s for s in walk_stmts(ir.body) if isinstance(s, ParLoop)]
  assert any(s.work is not None for s in loops)  # the inner reduction's length
  flat = jit(lambda x: x * 2.0 + 1.0).specialize(V).function
  assert all(s.work is None for s in walk_stmts(flat.body) if isinstance(s, ParLoop))
