import numpy as np
import pytest

import parakeet as pk
from benchmarks.keops_pairwise import exact_1nn, numpy_1nn
from parakeet.syntax import ir
from parakeet.transforms.pairwise_fold import COLUMNS, ROWS


@pk.jit
def farthest_l1(x, y):
  return np.argmax(np.sum(np.abs(x[:, None, :] - y[None, :, :]), axis=-1), axis=1)


@pytest.mark.parametrize("fn", [exact_1nn, numpy_1nn, farthest_l1])
def test_pairwise_scratch_bound_and_schedule(fn):
  x, y = np.empty((19, 5), "f"), np.empty((271, 5), "f")

  def inspect(name, fun):
    if name == "pairwise fold tiling":
      roots = {a.buf.uid for a in fun.results if isinstance(a, ir.ArrayResult)}
      scratch = [s.count for s in ir.walk_stmts(fun.body) if isinstance(s, ir.Alloc) and s.var.uid not in roots]
      assert scratch and all(isinstance(n, ir.Const) and 0 < n.value <= ROWS * COLUMNS for n in scratch)
      assert any(isinstance(s, ir.IndexTile) and s.level == "register" for s in ir.walk_stmts(fun.body))

  lowered = fn.trace_passes(x, y, _trace=inspect, _backend="c")
  roots = {result.buf.uid for result in lowered.results if isinstance(result, ir.ArrayResult)}
  scratch = [s.count for s in ir.walk_stmts(lowered.body) if isinstance(s, ir.Alloc) and s.var.uid not in roots]
  assert all(isinstance(n, ir.Const) and 0 < n.value <= ROWS * COLUMNS for n in scratch)
  assert "Pairwise map and ordered tile-summary fold" in pk.dsl.show_rewrite(fn, x, y, backend="c").laws_triggered


@pytest.mark.parametrize("backend", ["c", "multicore"])
def test_ties_nans_and_edges(backend):
  rng = np.random.default_rng(77)
  x = rng.normal(size=(35, 12)).astype("f")[::2, ::2]
  y = rng.normal(size=(2 * COLUMNS + 3, 6)).astype("f")
  x[0] = y[COLUMNS + 1] = y[3]
  nan_refs = y.copy()
  nan_refs[[11, COLUMNS + 3], 0] = np.nan
  for refs in [y, y[::-1], y[:1], y[:COLUMNS], y[: COLUMNS + 1], nan_refs]:
    dist = np.sum((x[:, None, :] - refs[None, :, :]) ** 2, axis=2)
    expected = np.argmin(dist, axis=1)
    for f in (numpy_1nn, exact_1nn):
      np.testing.assert_array_equal(f(x, refs, _backend=backend), expected)
    expected = np.argmax(np.sum(np.abs(x[:, None, :] - refs[None, :, :]), axis=2), axis=1)
    np.testing.assert_array_equal(farthest_l1(x, refs, _backend=backend), expected)

  for f in (numpy_1nn, exact_1nn):
    assert f(x[:0], y, _backend=backend).size == 0
    with pytest.raises(ValueError, match="empty"):
      f(x, y[:0], _backend=backend)
    with pytest.raises(ValueError, match="broadcast"):
      f(x, y[:, :3], _backend=backend)
    np.testing.assert_array_equal(f(x[:, :0], y[:, :0], _backend=backend), np.zeros(len(x), dtype=np.int64))


def test_empty_pairwise_domains_preserve_source_evaluation():
  points = np.empty((0, 3), dtype=np.float32)
  with pytest.raises(ValueError, match="argmin.*empty"):
    numpy_1nn(points, points, _backend="c")
  assert exact_1nn(points, points, _backend="c").shape == (0,)


def test_empty_reduced_dimensions_with_no_output_lanes():
  @pk.jit
  def reduce_axis(fn, array, axis):
    return fn(array, axis=axis)

  array = np.empty((0, 0, 2))
  for fn in (np.argmin, np.argmax, np.min, np.max):
    with pytest.raises(ValueError):
      reduce_axis(fn, array, 1)
    np.testing.assert_array_equal(reduce_axis(fn, array, 2), fn(array, axis=2))
  for fn in (np.min, np.max):
    with pytest.raises(ValueError):
      reduce_axis(fn, array, (1, 2))


def _check_gpu_row_fold(backend):
  rng = np.random.default_rng(44)
  x = rng.normal(size=(10, 12)).astype("f")[::2, ::2]
  y = rng.normal(size=(277, 6)).astype("f")[::-1]
  x[0] = y[3] = y[199]
  with pk.options(backend=backend, device_threshold=1, device_fallback="error"):
    for fn in (numpy_1nn, exact_1nn, farthest_l1):

      def inspect(stage, fun):
        if stage != "pairwise fold tiling":
          return
        [tile] = [s for s in ir.walk_stmts(fun.body) if isinstance(s, ir.IndexTile) and s.shared_tile]
        row_fold = tile.shared_tile.row_fold
        assert row_fold is not None
        scratch = {s.var.uid for s in ir.walk_stmts(tile.body) if isinstance(s, ir.Alloc)}
        assert scratch == {row_fold.scores.uid}
        # Outside the block only the result and the two partial-state arrays survive.
        outer = [s for s in fun.body if isinstance(s, ir.Alloc)]
        assert len(outer) == 3
        assert outer[1].count is outer[2].count

      fn.trace_passes(x, y, _trace=inspect)
      for refs in (y, y[:1], y[:65], y[:130]):
        delta = x[:, None, :] - refs[None, :, :]
        expected = (
          np.argmax(np.abs(delta).sum(axis=-1), axis=1)
          if fn is farthest_l1
          else np.argmin((delta * delta).sum(axis=-1), axis=1)
        )
        np.testing.assert_array_equal(fn(x, refs), expected)
      with_nan = y.copy()
      with_nan[[7, 202], 1] = np.nan
      np.testing.assert_array_equal(fn(x, with_nan), np.full(len(x), 7))
      assert fn(x[:0], y).size == 0
      if fn is exact_1nn:
        assert fn(x[:0], y[:0]).size == 0
      else:
        with pytest.raises(ValueError, match="empty"):
          fn(x[:0], y[:0])
      np.testing.assert_array_equal(fn(x[:, :0], y[:, :0]), np.zeros(len(x)))
      with pytest.raises(ValueError, match="empty"):
        fn(x, y[:0])
      with pytest.raises(ValueError, match="broadcast"):
        fn(x, y[:, :2])


@pytest.mark.skipif(pk.available_backends().get("metal") != "ok", reason="Metal device required")
def test_metal_ordered_pairwise_fold():
  _check_gpu_row_fold("metal")


def test_cuda_simulated_ordered_pairwise_fold(tmp_path):
  import sys

  from tests.test_backends import run_script

  if sys.platform == "win32":
    pytest.skip("the CUDA simulator needs a POSIX C++ toolchain")
  proc = run_script(
    tmp_path,
    "from tests.test_pairwise_fold import _check_gpu_row_fold\n_check_gpu_row_fold('cuda')",
    PARAKEET_CUDA_SIMULATE="1",
  )
  assert proc.returncode == 0, proc.stdout + proc.stderr
