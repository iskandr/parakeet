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


# ---- the middle end ---------------------------------------------------------


def test_operators_are_printed_before_the_passes():
  a = np.zeros((8, 8))
  spec = matmul.specialize(a, a)
  assert "map " in spec.operators and "reduce " in spec.operators and "fold(add" in spec.operators
  assert "parallel" not in spec.operators and "parallel" in spec.ir


def test_passes_can_be_skipped_by_name():
  from parakeet.transforms import pipeline

  assert pipeline.names() == [
    "dead code",
    "vertical fusion",
    "filter fusion",
    "sibling fusion",
    "reduction interchange",
    "tiling",
  ]
  a = np.arange(64.0).reshape(8, 8)
  with parakeet.options(skip_transforms=("tiling",)):
    assert "tile" not in matmul.specialize(a, a).ir
    np.testing.assert_allclose(matmul(a, a), a @ a)


@jit
def extremes(x):
  return x.min(), x.max()


@jit
def moments(x):
  return np.sum(x), np.sum(x * x)


@jit
def chained(x):
  m = np.sum(x) / len(x)
  return np.sum((x - m) ** 2)  # needs the first result: cannot be fused with it


@jit
def mixed(x):
  return np.sum(x), np.max(x)  # fusing would cost the sum its vector loop


def n_loops(spec):
  return spec.ir.count("parallel[reduce]")


def test_sibling_reductions_share_a_loop():
  x = np.random.default_rng(7).random(1000)
  for fn, want in ((extremes, (x.min(), x.max())), (moments, (x.sum(), (x * x).sum()))):
    np.testing.assert_allclose(fn(x), want)
    assert n_loops(fn.specialize(x)) == 1
    with parakeet.options(skip_transforms=("sibling fusion",)):
      np.testing.assert_allclose(fn(x), want)
      assert n_loops(fn.specialize(x)) == 2
  with pytest.raises(ValueError, match="no identity"):
    extremes(np.zeros(0))


def test_reductions_that_must_not_be_fused_are_not():
  x = np.random.default_rng(8).random(1000)
  np.testing.assert_allclose(chained(x), ((x - x.mean()) ** 2).sum())
  assert n_loops(chained.specialize(x)) == 2
  np.testing.assert_allclose(mixed(x), (x.sum(), x.max()))
  assert n_loops(mixed.specialize(x)) == 2


@jit
def wasteful(x):
  unused = np.sum(x * x)  # noqa: F841
  return x.max()


@jit
def wasteful_but_checked(x, i):
  unused = np.min(x[i:])  # noqa: F841 - raises when there is nothing left
  return x.max()


def test_reductions_nobody_reads_are_removed():
  x = np.random.default_rng(9).random(1000)
  assert wasteful(x) == x.max()
  assert n_loops(wasteful.specialize(x)) == 1
  with parakeet.options(skip_transforms=("dead code", "sibling fusion")):
    assert n_loops(wasteful.specialize(x)) == 2
  with pytest.raises(ValueError, match="no identity"):
    wasteful_but_checked(x, 1000)


# ---- FilterReduce -----------------------------------------------------------


@jit
def sum_kept(x):
  return np.sum(x[x > 0.5])


@jit
def mean_square_kept(x):
  y = x[x > 0.5]
  return np.sum(y * y) / len(y)  # the number of survivors is one more fold


@jit
def weights_kept(x, w):
  return np.sum(w[x > 0.5])  # a filter of one array selecting from another


@jit
def kept_and_returned(x):
  y = x[x > 0.5]
  return np.sum(y), y  # the kept elements are needed: the filter stays


@jit
def min_kept(x):
  return np.min(x[x > 0.5])  # no init: an empty selection has to raise


def test_a_reduction_over_a_filter_is_one_loop():
  x = np.random.default_rng(10).random(5000)
  w = np.arange(5000)
  y = x[x > 0.5]
  for fn, args, want in (
    (sum_kept, (x,), y.sum()),
    (mean_square_kept, (x,), (y * y).mean()),
    (weights_kept, (x, w), w[x > 0.5].sum()),
  ):
    np.testing.assert_allclose(fn(*args), want)
    spec = fn.specialize(*args)
    assert "filter" in spec.operators and n_loops(spec) == 1 and "idx" not in spec.ir
    with parakeet.options(skip_transforms=("filter fusion",)):
      np.testing.assert_allclose(fn(*args), want)
      assert n_loops(fn.specialize(*args)) == 3
  assert sum_kept(np.zeros(0)) == 0.0 and sum_kept(np.zeros(7)) == 0.0
  with parakeet.options(num_threads=1):
    np.testing.assert_allclose(sum_kept(x), y.sum())


def test_filters_that_are_needed_stay():
  x = np.random.default_rng(11).random(5000)
  y = x[x > 0.5]
  total, kept = kept_and_returned(x)
  np.testing.assert_allclose(total, y.sum())
  np.testing.assert_array_equal(kept, y)
  assert min_kept(x) == y.min()
  with pytest.raises(ValueError, match="no identity"):
    min_kept(np.zeros(9))


def test_every_adverb_has_a_statement():
  @jit
  def all_four(x):
    return np.cumsum(x), np.nonzero(x > 0.5)[0], np.sum(x), x * 2.0

  x = np.random.default_rng(12).random(3000)
  ops = all_four.specialize(x).operators
  for word in ("scan ", "filter ", "reduce ", "map "):
    assert word in ops
  assert "for " not in ops  # no loops before lowering
  got = all_four(x)
  np.testing.assert_allclose(got[0], np.cumsum(x))
  np.testing.assert_array_equal(got[1], np.nonzero(x > 0.5)[0])


# ---- vertical fusion, filter-filter, horizontal maps, interchange, DCE ------


@jit
def scan_then_map(x):
  return np.cumsum(x) * 2.0 + 1.0


@jit
def matmul_epilogue(a, b, c):
  # .copy() produces a concrete IndexMap in spec.operators so middle-end
  # vertical fusion fuses the tanh(... + c) epilogue right into the tiled matmul
  m = (a @ b).copy()
  return np.tanh(m + c)


@jit
def double_filter(x):
  y = parakeet.filter(lambda v: v > 0.0, x)
  return parakeet.filter(lambda v: 1.0 / v < 2.0, y)


@jit
def double_filter_sum(x):
  y = parakeet.filter(lambda v: v > 0.0, x)
  return np.sum(parakeet.filter(lambda v: v < 0.75, y))


@jit
def sibling_maps(x):
  return np.sin(x), np.cos(x)


@jit
def dependent_maps(x):
  a = x.copy()
  b = a * 2.0
  return a, b


@jit
def col_sum(x):
  return np.sum(x, axis=0)


@jit
def col_mean_bias(x, bias):
  return np.tanh(np.mean(x, axis=0) + bias)


@jit
def col_min(x):
  return np.min(x, axis=0)


@jit
def col_sum_3d(x):
  return np.sum(x, axis=(0, 1))


@jit
def dead_ops(x):
  _a = x.copy()  # noqa: F841
  _b = x[x > 0.5]  # noqa: F841
  _c = np.cumsum(x)  # noqa: F841
  return np.sum(x)


@jit
def cumsum_2d(x):
  return np.cumsum(x, axis=1)


def test_vertical_fusion_scan_and_matmul_epilogue():
  rng = np.random.default_rng(20)
  x = rng.random(2000)
  np.testing.assert_allclose(scan_then_map(x), np.cumsum(x) * 2.0 + 1.0)
  spec = scan_then_map.specialize(x)
  assert "parallel[map]" not in spec.ir
  with parakeet.options(skip_transforms=("vertical fusion",)):
    np.testing.assert_allclose(scan_then_map(x), np.cumsum(x) * 2.0 + 1.0)
    assert "parallel[map]" in scan_then_map.specialize(x).ir

  a, b, c = rng.random((32, 24)), rng.random((24, 32)), rng.random((32, 32))
  np.testing.assert_allclose(matmul_epilogue(a, b, c), np.tanh(a @ b + c))
  mspec = matmul_epilogue.specialize(a, b, c)
  assert mspec.operators.count("map ") == 3
  # 1 map to transpose b + 1 tiled map with the tanh(... + c) epilogue fused in
  assert "tile" in mspec.ir and mspec.ir.count("parallel[map]") == 2
  with parakeet.options(skip_transforms=("vertical fusion",)):
    assert matmul_epilogue.specialize(a, b, c).ir.count("parallel[map]") == 3


def test_filter_filter_fusion_and_short_circuit():
  rng = np.random.default_rng(21)
  x = rng.uniform(-1.0, 1.0, 2000)
  x[0] = 0.0  # must short-circuit before 1.0 / v
  want = x[(x > 0.0) & (np.where(x > 0.0, 1.0 / np.where(x > 0.0, x, 1.0), 9.0) < 2.0)]
  np.testing.assert_allclose(double_filter(x), want)
  # One filter needs 2 parallel[reduce] passes (count + scatter), not 4
  assert n_loops(double_filter.specialize(x)) == 2
  with parakeet.options(skip_transforms=("filter fusion",)):
    np.testing.assert_allclose(double_filter(x), want)
    assert n_loops(double_filter.specialize(x)) == 4

  np.testing.assert_allclose(double_filter_sum(x), x[(x > 0.0) & (x < 0.75)].sum())
  assert n_loops(double_filter_sum.specialize(x)) == 1


def test_horizontal_map_fusion():
  x = np.random.default_rng(22).random(2000)
  s, c = sibling_maps(x)
  np.testing.assert_allclose(s, np.sin(x))
  np.testing.assert_allclose(c, np.cos(x))
  assert sibling_maps.specialize(x).ir.count("parallel[map]") == 1
  with parakeet.options(skip_transforms=("sibling fusion",)):
    assert sibling_maps.specialize(x).ir.count("parallel[map]") == 2

  # dependent maps must not merge when the second reads what the first writes
  a, b = dependent_maps(x)
  np.testing.assert_allclose(a, x)
  np.testing.assert_allclose(b, x * 2.0)
  assert dependent_maps.specialize(x).ir.count("parallel[map]") == 2


def test_reduction_interchange():
  rng = np.random.default_rng(23)
  for rows, cols in ((0, 0), (0, 5), (1, 1), (3, 4), (7, 63), (7, 64), (7, 65), (7, 255), (7, 256), (7, 257)):
    x = rng.random((rows, cols))
    np.testing.assert_allclose(col_sum(x), x.sum(axis=0))
    if rows > 0:
      b = rng.random(cols)
      np.testing.assert_allclose(col_mean_bias(x, b), np.tanh(x.mean(axis=0) + b))
      np.testing.assert_allclose(col_min(x), x.min(axis=0))
    elif cols > 0:
      with pytest.raises(ValueError, match="no identity"):
        col_min(x)

  x = rng.random((32, 300))
  assert "tile_acc" in col_sum.specialize(x).ir
  with parakeet.options(skip_transforms=("reduction interchange",)):
    assert "tile_acc" not in col_sum.specialize(x).ir
    np.testing.assert_allclose(col_sum(x), x.sum(axis=0))

  x3 = rng.random((5, 7, 300))
  np.testing.assert_allclose(col_sum_3d(x3), x3.sum(axis=(0, 1)))
  assert "tile_acc" in col_sum_3d.specialize(x3).ir


def test_dead_maps_filters_and_scans_are_removed():
  x = np.random.default_rng(24).random(1000)
  np.testing.assert_allclose(dead_ops(x), x.sum())
  spec = dead_ops.specialize(x)
  assert "parallel[map]" not in spec.ir and n_loops(spec) == 1


def test_multidim_scan_emits_index_scan():
  x = np.random.default_rng(25).random((8, 12))
  ops = cumsum_2d.specialize(x).operators
  assert "map " in ops and "scan " in ops and "for " not in ops
  np.testing.assert_allclose(cumsum_2d(x), np.cumsum(x, axis=1))


def test_ir_body_sharing_and_diagnose_c_backend():
  @jit
  def repeated_maps(a, b, c):
    # Each call produces an identical ParLoop structure (and includes bounds checks
    # if indexing); structural IR hashing must share one compiled pk_body_.
    return np.sin(a) + 1.0, np.sin(b) + 1.0, np.sin(c) + 1.0

  rng = np.random.default_rng(26)
  a, b, c = rng.random(500), rng.random(500), rng.random(500)
  with parakeet.options(skip_transforms=("sibling fusion",)):
    src = repeated_maps.specialize(a, b, c).source
    assert src.count("static void pk_body_") == 1
    out = repeated_maps(a, b, c)
    for got, arr in zip(out, (a, b, c)):
      np.testing.assert_allclose(got, np.sin(arr) + 1.0)

  # find_broken_transform on backend="c" must not duplicate "c" as "parallel loops"
  with parakeet.options(backend="c"):
    assert parakeet.find_broken_transform(lambda x: np.sum(x * x), [a], print_transforms=False) is None


def test_filter_reduce_adverb_and_combine_nested_maps():
  rng = np.random.default_rng(27)
  x = rng.uniform(-1.0, 1.0, 2000)
  w = rng.random(2000)

  @jit
  def weighted_pos_sum(a, b):
    return parakeet.filter_reduce(parakeet.add, lambda u, v: u > 0.0, a, b, map=lambda u, v: u * v, init=0.0)

  np.testing.assert_allclose(weighted_pos_sum(x, w), (x[x > 0.0] * w[x > 0.0]).sum())
  assert n_loops(weighted_pos_sum.specialize(x, w)) == 1
  # Also callable directly from Python
  np.testing.assert_allclose(parakeet.filter_reduce(parakeet.add, lambda v: v > 0.0, x, init=0.0), x[x > 0.0].sum())

  # CombineNestedMaps (thesis Fig. 6.2): nested expensive maps flatten into one 2D parallel map
  @jit
  def nested_expensive(m):
    def inner(v):
      for _ in range(6):
        v = np.tanh(v) + 0.1
      return v

    return parakeet.each(lambda row: parakeet.each(inner, row), m)

  m = rng.random((4, 64))
  want = m.copy()
  for _ in range(6):
    want = np.tanh(want) + 0.1
  np.testing.assert_allclose(nested_expensive(m), want)
  nspec = nested_expensive.specialize(m)
  # Elaboration emits outer map + inner map + row-copy map with a per-row temp buffer;
  # vertical fusion eliminates the per-row temp buffer and CombineNestedMaps flattens into one 2D parallel map!
  assert nspec.operators.count("map ") == 3 and "release(" in nspec.operators
  assert nspec.ir.count("parallel[map]") == 1 and "release(" not in nspec.ir
  assert "(m_n0_" in nspec.ir and "m_n1_" in nspec.ir


def test_ir_verifier_catches_invalid_ir_and_legacy_config_aliases():
  from parakeet.errors import InternalError
  from parakeet.syntax import ir

  assert parakeet.config.opt_verify is parakeet.config.verify
  assert parakeet.config.print_indexified_function is parakeet.config.print_operators

  # Construct a deliberately broken Function referencing an unbound variable
  bad_var = ir.Var("ghost", parakeet.Float64)
  fn = ir.Function("broken", [], [ir.Assign(ir.Var("y", parakeet.Float64), bad_var)], [], ir.Scope("fn"))
  with pytest.raises(InternalError, match="not in scope"):
    parakeet.verify(fn)
