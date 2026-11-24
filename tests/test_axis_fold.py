import numpy as np
import pytest

import parakeet as pk
from examples.algebraic_gallery import nearest_fold, nearest_numpy, smooth_fold, smooth_numpy, squared_distance
from parakeet import dsl
from parakeet.syntax import ir, ops
from parakeet.transforms.analysis import IndexFacts


def test_structured_fold_axes_seed_and_order():
  # Each pair represents the transform t -> a*t+b. Composition is associative
  # and order-sensitive, and (2, 7) is intentionally not its identity.
  def lift(value, i):
    return np.int64(-1), value + i

  def compose(left, right):
    return right[0] * left[0], right[0] * left[1] + right[1]

  @pk.jit
  def run(x, k):
    return dsl.fold(lift, compose, x, (np.int64(2), np.int64(7)), axis=k, indexed=True, emit=lambda pair: pair[1])

  for shape, axis in [((2, 139, 3), 1), ((7, 11), -1), ((0, 4), 0), ((0, 3), 1)]:
    x = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    lanes = np.moveaxis(x, axis, -1)
    expected = np.full(lanes.shape[:-1], 7)
    for j in range(lanes.shape[-1]):
      expected = -expected + lanes[..., j] + j
    np.testing.assert_array_equal(run(x, axis), expected)


@pk.jit
def seeded_score(queries, refs):
  scores = dsl.allpairs(squared_distance, queries, refs)
  return dsl.fold(
    lambda v: (v, v * v),
    lambda a, b: (a[0] + b[0], a[1] + b[1]),
    scores,
    (np.float32(2), np.float32(17)),
    axis=-1,
    emit=lambda s: s[1],
  )


def check_pairwise(backend):
  rng = np.random.default_rng(491)
  x = rng.standard_normal((19, 7), dtype=np.float32)
  refs = rng.standard_normal((273, 7), dtype=np.float32)[::-1]
  values = rng.standard_normal(546, dtype=np.float32)[::2]
  x[0] = refs[5] = refs[210]
  with pk.options(backend=backend, device_threshold=1, device_fallback="error", verify=True):
    squared = np.sum((x[:, None] - refs) ** 2, axis=2)
    cases = [
      (nearest_fold, (x, refs), nearest_numpy.fn(x, refs)),
      (smooth_fold, (x, refs, values), smooth_numpy.fn(x, refs, values)),
      (seeded_score, (x, refs), 17 + np.sum(squared * squared, axis=1)),
    ]
    for fn, args, expected in cases:
      tiles = []

      def inspect(name, fun):
        if name != "pairwise fold tiling":
          return
        facts = IndexFacts.from_function(fun)
        matrix_size = ops.imul(fun.params[0].shape[0], fun.params[1].shape[0])
        for op in ir.walk_stmts(fun.body):
          if isinstance(op, ir.IndexTile) and op.level in ("cache", "shared"):
            tiles.append(op)
          if isinstance(op, ir.Alloc):
            assert not facts.same_expr(op.count, matrix_size)

      fn.trace_passes(*args, _trace=inspect)
      assert len(tiles) == 1
      if backend in ("metal", "cuda"):
        assert tiles[0].shared_tile.row_fold is not None
      if fn is nearest_fold:
        np.testing.assert_array_equal(fn(*args), expected)
      else:
        np.testing.assert_allclose(fn(*args), expected, rtol=3e-5, atol=5e-6)
    nan_refs = refs.copy()
    nan_refs[[9, 191], -1] = np.nan
    np.testing.assert_array_equal(nearest_fold(x, nan_refs), np.full(len(x), 9))
    np.testing.assert_array_equal(seeded_score(x, refs[:0]), np.full(len(x), 17))
    with pytest.raises(ValueError, match="empty"):
      nearest_fold(x[:0], refs[:0])
    with pytest.raises(ValueError):
      smooth_fold(x, refs, values[:-1])


def test_cpu_custom_tile_fold():
  check_pairwise("c")


@pytest.mark.skipif(pk.available_backends().get("metal") != "ok", reason="Metal device required")
def test_metal_custom_tile_fold():
  check_pairwise("metal")


def test_cuda_simulated_custom_tile_fold(tmp_path):
  import sys
  from tests.test_backends import run_script

  if sys.platform == "win32":
    pytest.skip("the CUDA simulator needs a POSIX C++ toolchain")
  result = run_script(
    tmp_path, "from tests.test_axis_fold import check_pairwise\ncheck_pairwise('cuda')", PARAKEET_CUDA_SIMULATE="1"
  )
  assert result.returncode == 0, result.stdout + result.stderr
