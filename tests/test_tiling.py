"""Tiled evaluation of matmul-shaped maps must not change what is computed."""

import itertools

import numpy as np
import pytest

import parakeet
from parakeet import allpairs, jit
from parakeet.transforms import tiling


@pytest.fixture(autouse=True)
def generated_products():
  """These tests are about Parakeet's own loops, not the BLAS it would otherwise call."""
  with parakeet.options(blas=False):
    yield


@jit
def matmul(a, b):
  return a @ b


@jit
def sqr_dists(x, y):
  return allpairs(lambda u, v: np.sum((u - v) ** 2), x, y)


@jit
def norm_dists(x, y):
  # code after the reduction has to run after the jammed loop
  return allpairs(lambda u, v: np.sqrt(np.sum((u - v) ** 2)) + 1.0, x, y)


@jit
def tropical(x, y):
  # a reduction without an identity: starts from the first term
  return allpairs(lambda u, v: np.min(u + v), x, y)


@jit
def row_sums_of_products(xs, b):
  return parakeet.each(lambda a: np.sum(a @ b), xs)


# around the block sizes (2 rows by 4 columns), plus empty and single
ROWS = (0, 1, 2, 3, 5, 17)
INNER = (0, 1, 3, 33)
COLS = (0, 1, 3, 4, 5, 7, 8, 13)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_matmul_shapes(dtype):
  rng = np.random.default_rng(0)
  for n, k, m in itertools.product(ROWS, INNER, COLS):
    a = (rng.random((n, k)) * 10).astype(dtype)
    b = (rng.random((k, m)) * 10).astype(dtype)
    got = matmul(a, b)
    assert got.shape == (n, m) and got.dtype == dtype
    np.testing.assert_allclose(got, a @ b, rtol=1e-4, err_msg=f"{n}x{k} @ {k}x{m}")


@pytest.mark.parametrize("grain", [1, 7, 8192])
def test_chunks_cut_rows_anywhere(grain):
  # a tiny grain makes many small chunks, so chunks begin and end mid-row
  rng = np.random.default_rng(1)
  a, b = rng.random((23, 9)), rng.random((9, 11))
  with parakeet.options(parallel_grain=grain, num_threads=4):
    np.testing.assert_allclose(matmul(a, b), a @ b)


@pytest.mark.parametrize("tile_bytes", [1024, 4096, 1 << 20])
def test_tile_size_does_not_matter(tile_bytes):
  rng = np.random.default_rng(2)
  a, b = rng.random((40, 70)), rng.random((70, 50))
  with parakeet.options(tile_bytes=tile_bytes):
    np.testing.assert_allclose(matmul(a, b), a @ b)


def test_allpairs_with_user_functions():
  rng = np.random.default_rng(3)
  for n, m in itertools.product((1, 2, 5, 9), (1, 4, 6, 11)):
    x, y = rng.random((n, 7)), rng.random((m, 7))
    d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(sqr_dists(x, y), d2)
    np.testing.assert_allclose(norm_dists(x, y), np.sqrt(d2) + 1.0)
    np.testing.assert_allclose(tropical(x, y), (x[:, None, :] + y[None, :, :]).min(-1))


def test_reduction_without_identity_still_rejects_empty_input():
  with pytest.raises(ValueError, match="no identity"):
    tropical(np.zeros((3, 0)), np.zeros((4, 0)))


def test_nested_in_a_parallel_map():
  rng = np.random.default_rng(4)
  xs, b = rng.random((5, 30, 20)), rng.random((20, 40))
  np.testing.assert_allclose(row_sums_of_products(xs, b), (xs @ b).sum((1, 2)))


def test_same_answer_with_tiling_off():
  rng = np.random.default_rng(5)
  a, b = rng.random((31, 17)).astype(np.float32), rng.random((17, 29)).astype(np.float32)
  with parakeet.options(tiling=False):
    plain = matmul(a, b)
  np.testing.assert_allclose(matmul(a, b), plain, rtol=1e-5)


def test_tiled_loops_are_in_the_ir():
  a = np.zeros((8, 8))
  assert "tile" in matmul.specialize(a, a).ir
  with parakeet.options(tiling=False):
    assert "tile" not in matmul.specialize(a, a).ir


def test_small_constant_shapes_are_left_alone():
  @jit
  def rotate(r, v):
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ r @ v

  r = np.arange(9.0).reshape(3, 3)
  np.testing.assert_allclose(rotate(r, np.ones(3)), r @ np.ones(3))


def test_tile_holds_whole_blocks():
  assert tiling.UNROLL_I >= 1 and tiling.UNROLL_J >= 1


def test_find_broken_transform_names_tiling_as_a_step():
  from parakeet.frontend import diagnose

  assert "tiling" in [name for name, _ in diagnose.STEPS]
  rng = np.random.default_rng(6)
  args = [rng.random((6, 5)), rng.random((5, 4))]
  assert parakeet.find_broken_transform(lambda a, b: a @ b, args, print_transforms=False) is None
  assert (
    parakeet.find_broken_transform(lambda a, b: a @ b, args, expected=np.zeros((6, 4)), print_transforms=False)
    == "frontend"
  )
