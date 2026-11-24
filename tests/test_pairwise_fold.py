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
