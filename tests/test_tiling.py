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
    "normalize",
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

  # Same-index dependent maps fuse write-through into 1 loop (keeping both outputs!)
  a, b = dependent_maps(x)
  np.testing.assert_allclose(a, x)
  np.testing.assert_allclose(b, x * 2.0)
  assert dependent_maps.specialize(x).ir.count("parallel[map]") == 1

  # Cross-index dependent maps (b[i] reads a[N - 1 - i]) must NOT merge!
  @jit
  def cross_index_dependent(x):
    a = x.copy()
    b = a[::-1].copy()
    return a, b

  a2, b2 = cross_index_dependent(x)
  np.testing.assert_allclose(a2, x)
  np.testing.assert_allclose(b2, x[::-1])
  assert cross_index_dependent.specialize(x).ir.count("parallel[map]") == 2


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
  # CPU targets use L1 column-strip interchange (tile_acc); CUDA skips it so warps coalesce X[k*N+j] directly
  with parakeet.options(backend="multicore"):
    assert "tile_acc" in col_sum.specialize(x).ir
    with parakeet.options(skip_transforms=("reduction interchange",)):
      assert "tile_acc" not in col_sum.specialize(x).ir
      np.testing.assert_allclose(col_sum(x), x.sum(axis=0))

  x3 = rng.random((5, 7, 300))
  np.testing.assert_allclose(col_sum_3d(x3), x3.sum(axis=(0, 1)))
  with parakeet.options(backend="multicore"):
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


def test_argmin_argmax_are_carry_index_reduce_and_sibling_fuse():
  @jit
  def arg_extremes(x):
    return np.argmin(x), np.argmax(x)

  rng = np.random.default_rng(28)
  x = rng.random(5000)
  spec = arg_extremes.specialize(x)
  # Pure operator IR: no lowered ParLoop or ForRange before lowering
  assert "reduce " in spec.operators and "parallel[" not in spec.operators and "for " not in spec.operators
  # Sibling fusion merges argmin + argmax into a single parallel reduction loop!
  assert n_loops(spec) == 1
  assert arg_extremes(x) == (int(np.argmin(x)), int(np.argmax(x)))
  # Ties and NaN follow NumPy (first wins)
  tied = np.array([3.0, 1.0, 5.0, 1.0, 5.0])
  assert arg_extremes(tied) == (1, 2)
  with_nan = np.array([3.0, np.nan, 5.0, np.nan])
  assert arg_extremes(with_nan) == (1, 1)
  with pytest.raises(ValueError, match="empty"):
    arg_extremes(np.zeros(0))


def test_index_tile_and_tile_stage_ir_verify_clone_and_lower():
  from parakeet.ndtypes import BufType, Float64, Int64
  from parakeet.syntax import ir, ops
  from parakeet.syntax.clone import Cloner
  from parakeet.transforms import lowering

  # Build a 2D IndexTile with a TileStage that stages an input strip into a tile-local
  # buffer and copies its doubled values into an output buffer, then verify, clone, and lower.
  inp = ir.Var("inp", BufType(Float64))
  out = ir.Var("out", BufType(Float64))
  off_in = ir.Var("off_in", Int64)
  off_out = ir.Var("off_out", Int64)
  n = ir.Var("n", Int64)
  m = ir.Var("m", Int64)
  p_in = ir.ArrayParam(inp, off_in, (n, m), None)
  p_out = ir.ArrayParam(out, off_out, (n, m), None, writes=True)

  t0, t1 = ir.Var("t0", Int64), ir.Var("t1", Int64)
  te0, te1 = ir.Var("te0", Int64), ir.Var("te1", Int64)
  sbuf = ir.Var("stage", BufType(Float64))
  si, sj = ir.Var("si", Int64), ir.Var("sj", Int64)
  sval = ir.Load(inp, ops.iadd(ops.imul(ops.iadd(t0, si), m), ops.iadd(t1, sj)))
  stage = ir.TileStage(sbuf, (te0, te1), (si, sj), [], sval, ir.Scope("stage"))

  ui, uj = ir.Var("ui", Int64), ir.Var("uj", Int64)
  loaded = ir.Load(sbuf, ops.iadd(ops.imul(ui, te1), uj))
  doubled = ops.prim("mul", (loaded, ir.Const(2.0, Float64)), Float64)
  imap_scope = ir.Scope("loop")
  imap = ir.IndexMap(
    (te0, te1),
    (ui, uj),
    [
      ir.Store(out, ops.iadd(ops.imul(ops.iadd(t0, ui), m), ops.iadd(t1, uj)), doubled),
      ir.ReleaseScope(imap_scope),
    ],
    imap_scope,
    False,
    False,
  )
  tile_scope = ir.Scope("tile")
  tile = ir.IndexTile(
    (n, m),
    (ir.Const(4, Int64), ir.Const(8, Int64)),
    (t0, t1),
    (te0, te1),
    [stage],
    [imap, ir.ReleaseScope(tile_scope)],
    tile_scope,
    level="cache",
    parallel=True,
  )
  fn = ir.Function("staged_double", [p_in, p_out], [tile], [], ir.Scope("fn"))
  parakeet.verify(fn, "index_tile")
  text = ir.fmt_function(fn)
  assert "tile[cache]" in text and "stage " in text

  # Cloner must preserve validity with fresh variable uids
  cloned_body = Cloner({inp.uid: inp, out.uid: out, n.uid: n, m.uid: m}).block(fn.body)
  fn2 = ir.Function("staged_double_copy", [p_in, p_out], cloned_body, [], ir.Scope("fn"))
  parakeet.verify(fn2, "cloned_index_tile")

  # Lowering must produce valid verified loop IR with no remaining IndexTile
  lowering.lower(fn)
  parakeet.verify(fn, "lowered_index_tile")
  assert "tile[cache]" not in ir.fmt_function(fn)
  assert "parallel[map]" in ir.fmt_function(fn)


def test_squeeze_unit_map_and_split_invariant_map_softmax_layernorm():
  @jit
  def softmax(x):
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=1, keepdims=True)

  @jit
  def layer_norm(x, g, b):
    mu = np.mean(x, axis=1, keepdims=True)
    var = np.mean((x - mu) ** 2, axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * g + b

  rng = np.random.default_rng(29)
  for rows, cols in ((0, 5), (1, 1), (4, 7), (32, 64)):
    x = rng.standard_normal((rows, cols))
    g = rng.standard_normal(cols)
    b = rng.standard_normal(cols)
    if rows > 0:
      e = np.exp(x - x.max(axis=1, keepdims=True))
      np.testing.assert_allclose(softmax(x), e / e.sum(axis=1, keepdims=True), rtol=1e-6)
      mu = x.mean(axis=1, keepdims=True)
      want_ln = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=1, keepdims=True) + 1e-5) * g + b
      np.testing.assert_allclose(layer_norm(x, g, b), want_ln, rtol=1e-6)
    else:
      assert softmax(x).shape == (0, cols)
      assert layer_norm(x, g, b).shape == (0, cols)

  # Empty trailing axis preserves exact semantics (zero elements -> empty (3, 0) result);
  # non-empty output with empty reduced operand still raises on np.max without identity.
  assert softmax(np.zeros((3, 0))).shape == (3, 0)

  @jit
  def add_row_max(x, y):
    return x + np.max(y, axis=1, keepdims=True)

  with pytest.raises(ValueError, match="no identity"):
    add_row_max(np.zeros((3, 4)), np.zeros((3, 0)))

  x = rng.standard_normal((32, 64))
  g = rng.standard_normal(64)
  b = rng.standard_normal(64)
  # layer_norm collapses to 1 single parallel loop and 1 allocation (0 temp buffers!)
  ln_spec = layer_norm.specialize(x, g, b)
  assert ln_spec.ir.count("alloc(") == 1
  assert ln_spec.ir.count("parallel[map]") == 1
  # Verify with parakeet.config.verify enabled across all passes
  with parakeet.options(verify=True):
    np.testing.assert_allclose(
      softmax(x),
      np.exp(x - x.max(axis=1, keepdims=True)) / np.exp(x - x.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True),
      rtol=1e-6,
    )
    mu = x.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(
      layer_norm(x, g, b),
      (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=1, keepdims=True) + 1e-5) * g + b,
      rtol=1e-6,
    )


def test_audit_fixes_cosine_allpairs_col_var_filter_dedup_and_sunk_sibling_reduce():
  @jit
  def cosine_allpairs(x, y):
    return allpairs(
      lambda u, v: np.sum(u * v) / (np.sqrt(np.sum(u * u)) * np.sqrt(np.sum(v * v)) + 1e-6),
      x,
      y,
    )

  @jit
  def sum_pos_scaled(x, w):
    mask = x > 0.0
    return np.sum(x[mask] * w[mask])

  @jit
  def col_center(x):
    return x - np.mean(x, axis=0, keepdims=True)

  @jit
  def mean_and_sumsq(x):
    m = np.sum(x) / len(x)
    s2 = np.sum(x * x)
    return m, s2

  rng = np.random.default_rng(31)
  with parakeet.options(verify=True):
    # 1. Checked-equality propagation lets cosine_allpairs sibling-fuse all 3 reductions and tile
    x = rng.standard_normal((16, 20))
    y = rng.standard_normal((12, 20))
    want_cos = (x @ y.T) / (np.linalg.norm(x, axis=1, keepdims=True) * np.linalg.norm(y, axis=1)[None, :] + 1e-6)
    np.testing.assert_allclose(cosine_allpairs(x, y), want_cos, rtol=1e-6)
    cos_spec = cosine_allpairs.specialize(x, y)
    assert "tile" in cos_spec.ir and cos_spec.ir.count("parallel[map]") == 1

    # 2. Duplicate IndexFilter CSE collapses mask=x>0; sum(x[mask]*w[mask]) into 1 FilterReduce loop
    a = rng.standard_normal(2000)
    w = rng.standard_normal(2000)
    np.testing.assert_allclose(sum_pos_scaled(a, w), (a[a > 0.0] * w[a > 0.0]).sum(), rtol=1e-6)
    assert n_loops(sum_pos_scaled.specialize(a, w)) == 1
    assert sum_pos_scaled(np.array([-1.0, -2.0]), np.array([3.0, 4.0])) == 0.0
    assert sum_pos_scaled(np.array([2.0, -1.0]), np.array([5.0, 4.0])) == 10.0  # nnz == 1 broadcast edge case
    with pytest.raises(IndexError):
      sum_pos_scaled(np.ones(4), np.ones(5))

    # 3. Generalized split_invariant_map hoists leading-axis-invariant reductions (axis=0)
    np.testing.assert_allclose(col_center(x), x - x.mean(axis=0, keepdims=True), rtol=1e-6)
    assert col_center(np.zeros((0, 5))).shape == (0, 5)
    assert col_center(np.zeros((5, 0))).shape == (5, 0)

    # 4. Dependent scalar post-processing (m = sum(x)/n) sinks past independent sibling reduction
    m_got, s2_got = mean_and_sumsq(a)
    np.testing.assert_allclose(m_got, a.mean(), rtol=1e-6)
    np.testing.assert_allclose(s2_got, (a * a).sum(), rtol=1e-6)
    assert n_loops(mean_and_sumsq.specialize(a)) == 1


def test_index_facts_affine_fusion_and_target_aware_interchange():
  from parakeet.ndtypes import Int64
  from parakeet.syntax import ir, ops
  from parakeet.transforms.analysis import IndexFacts

  # 1. IndexFacts canonical polynomial equality, transitive checked equalities, stride_of, decompose_row_major
  i, j, n, m, k = (ir.Var(name, Int64) for name in ("i", "j", "n", "m", "k"))
  facts = IndexFacts()
  for v in (i, j, n, m, k):
    facts.note_var(v)
  facts.bind_index(i, n)
  facts.bind_index(j, m)
  assert facts.same_expr(ops.iadd(ops.imul(i, m), j), ops.iadd(j, ops.imul(m, i)))
  assert facts.same_expr(ops.isub(ops.iadd(i, 1), 1), i)
  facts.record_equality(n, k)
  assert facts.same_expr(n, k)
  assert facts.same_expr(facts.stride_of(ops.iadd(ops.imul(i, m), j), i), m)
  assert facts.same_expr(facts.stride_of(ops.iadd(ops.imul(i, m), j), j), ops.ONE)
  coords = facts.decompose_row_major(ops.iadd(ops.imul(i, m), j), (n, m))
  assert coords is not None and facts.same_expr(coords[0], i) and facts.same_expr(coords[1], j)
  # Flat reshape index (bound n * m, NOT < m) must be safely rejected
  flat = ir.Var("flat", Int64)
  facts.bind_index(flat, ops.imul(n, m))
  assert facts.decompose_row_major(flat, (n, m)) is None

  # 2. Affine transpose fusion: cos(a).copy().T.copy() collapses into 1 map & 1 alloc!
  @jit
  def fused_transpose(a):
    y = np.cos(a).copy()
    return y.T.copy()

  # 3. Cheap stencil fusion vs expensive non-duplication
  @jit
  def cheap_stencil(x):
    y = (x * 2.0 + 1.0).copy()
    return y[1:] - y[:-1]

  @jit
  def expensive_stencil(x):
    y = np.exp(x).copy()
    return y[1:] - y[:-1]

  # 4. Cheap multi-consumer fusion: y = (x * 2 + 1).copy(); return sin(y), cos(y) -> 2 allocs, 1 map!
  @jit
  def multi_consumer_cheap(x):
    y = (x * 2.0 + 1.0).copy()
    return np.sin(y), np.cos(y)

  rng = np.random.default_rng(32)
  a = rng.standard_normal((16, 24))
  x = rng.standard_normal(500)
  with parakeet.options(verify=True):
    np.testing.assert_allclose(fused_transpose(a), np.cos(a).T, rtol=1e-6)
    tspec = fused_transpose.specialize(a)
    assert tspec.ir.count("alloc(") == 1 and tspec.ir.count("parallel[map]") == 1

    np.testing.assert_allclose(cheap_stencil(x), (x * 2.0 + 1.0)[1:] - (x * 2.0 + 1.0)[:-1], rtol=1e-6)
    cspec = cheap_stencil.specialize(x)
    assert cspec.ir.count("alloc(") == 1 and cspec.ir.count("parallel[map]") == 1

    np.testing.assert_allclose(expensive_stencil(x), np.exp(x)[1:] - np.exp(x)[:-1], rtol=1e-6)
    espec = expensive_stencil.specialize(x)
    assert espec.ir.count("alloc(") == 2  # expensive exp must stay materialized

    s_got, c_got = multi_consumer_cheap(x)
    np.testing.assert_allclose(s_got, np.sin(x * 2.0 + 1.0), rtol=1e-6)
    np.testing.assert_allclose(c_got, np.cos(x * 2.0 + 1.0), rtol=1e-6)
    mspec = multi_consumer_cheap.specialize(x)
    assert mspec.ir.count("alloc(") == 2 and mspec.ir.count("parallel[map]") == 1

    # 5. Invariant check hoisting preserves empty-array semantics on allpairs
    assert sqr_dists(np.zeros((0, 4)), np.zeros((3, 5))).shape == (0, 3)
    with pytest.raises(ValueError, match="broadcast"):
      sqr_dists(np.zeros((2, 4)), np.zeros((3, 5)))


def test_index_scatter_bincount_scatter_add_fusion_and_autodiff():
  @jit
  def hist_unweighted(idx, minlength):
    return np.bincount(idx, minlength=minlength)

  @jit
  def hist_weighted_fused(idx, x):
    # Map (x * 2 + 1) fuses vertically into IndexScatter!
    w = (x * 2.0 + 1.0).copy()
    return np.bincount(idx, weights=w)

  @jit
  def scatter_f32(n, idx, vals):
    return parakeet.scatter_add(n, idx, vals)

  rng = np.random.default_rng(33)
  with parakeet.options(verify=True):
    # Small (sequential path) AND large (privatized parallel path >= 4096)
    for size, nbins in ((0, 0), (0, 5), (17, 4), (10_000, 64)):
      idx = rng.integers(0, max(nbins, 1), size=size, dtype=np.int64)
      w = rng.standard_normal(size)
      np.testing.assert_array_equal(hist_unweighted(idx, nbins), np.bincount(idx, minlength=nbins))
      if size > 0:
        np.testing.assert_allclose(hist_weighted_fused(idx, w), np.bincount(idx, weights=w * 2.0 + 1.0), rtol=1e-10)
        # Float32 scatter_add (works natively on CPU, CUDA, and Metal)
        w32 = w.astype(np.float32)
        np.testing.assert_allclose(
          scatter_f32(nbins, idx, w32),
          np.bincount(idx, weights=w32, minlength=nbins).astype(np.float32),
          rtol=1e-4,
          atol=1e-4,
        )
    # Operator IR uses IndexScatter, and Map -> Scatter vertical fusion eliminates the temp buffer for w!
    idx_big = rng.integers(0, 32, size=8000, dtype=np.int64)
    w_big = rng.standard_normal(8000)
    spec = hist_weighted_fused.specialize(idx_big, w_big)
    assert "scatter[add]" in spec.operators
    assert "arr_" not in spec.ir  # w = (x * 2 + 1).copy() fused away completely into IndexScatter!
    assert "2.0" in spec.ir and "priv_" in spec.ir

    # Negative index & error checks
    neg_idx = np.array([-1, 0, -2, 1], dtype=np.int64)
    vals = np.array([1.5, 2.0, 3.0, 4.5])
    np.testing.assert_allclose(parakeet.scatter_add(4, neg_idx, vals), np.array([2.0, 4.5, 3.0, 1.5]))
    with pytest.raises(ValueError, match="negative"):
      hist_unweighted(np.array([0, -1, 2], dtype=np.int64), 0)
    with pytest.raises(IndexError, match="out of bounds"):
      scatter_f32(4, np.array([0, 4], dtype=np.int64), np.array([1.0, 2.0], dtype=np.float32))

    # Reverse-mode AD through scatter_add AND weighted bincount (adjoint is gather!)
    idx_ad = np.array([0, 2, 0, 1, 2], dtype=np.int64)
    v_ad = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scale = np.array([10.0, 20.0, 30.0])

    @jit
    def loss_scatter(v):
      s = parakeet.scatter_add(3, idx_ad, v)
      return np.sum(s * s * scale)

    @jit
    def loss_bincount(v):
      s = np.bincount(idx_ad, weights=v, minlength=3)
      return np.sum(s * s * scale)

    # d/dv_i sum_k (scale_k * s_k^2) = 2 * scale[idx_i] * s[idx_i]
    s_ref = np.bincount(idx_ad, weights=v_ad, minlength=3)
    want_grad = 2.0 * scale[idx_ad] * s_ref[idx_ad]
    np.testing.assert_allclose(parakeet.grad(loss_scatter)(v_ad), want_grad, rtol=1e-10)
    np.testing.assert_allclose(parakeet.grad(loss_bincount)(v_ad), want_grad, rtol=1e-10)


def test_online_monoidal_logsumexp_and_softmax():
  @jit
  def logsumexp_1d(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

  @jit
  def logsumexp_2d(x):
    m = np.max(x, axis=1, keepdims=True)
    return m[:, 0] + np.log(np.sum(np.exp(x - m), axis=1))

  @jit
  def online_softmax_2d(x):
    m = np.max(x, axis=1, keepdims=True)
    return np.exp(x - m) / np.sum(np.exp(x - m), axis=1, keepdims=True)

  rng = np.random.default_rng(34)
  with parakeet.options(verify=True):
    # 1D logsumexp fuses into 1 SINGLE parallel reduction loop (1 pass over x!)
    for x in (
      rng.standard_normal(10_000),
      rng.standard_normal(500) + 1000.0,  # large values: must not overflow!
      np.array([42.0]),
      np.array([1.0, -np.inf, 2.0]),
    ):
      m = x.max()
      want = float(m + np.log(np.exp(x - m).sum()))
      np.testing.assert_allclose(logsumexp_1d(x), want, rtol=1e-12)
    spec_1d = logsumexp_1d.specialize(rng.standard_normal(10_000))
    assert n_loops(spec_1d) == 1
    with parakeet.options(skip_transforms=("sibling fusion",)):
      assert n_loops(logsumexp_1d.specialize(rng.standard_normal(10_000))) == 2
    with pytest.raises(ValueError, match="no identity"):
      logsumexp_1d(np.zeros(0))

    # 2D row-wise logsumexp and online_softmax (zero temp array!)
    x2 = rng.standard_normal((32, 64)) + 500.0
    m2 = x2.max(axis=1, keepdims=True)
    want_lse2 = m2[:, 0] + np.log(np.exp(x2 - m2).sum(axis=1))
    want_sm2 = np.exp(x2 - m2) / np.exp(x2 - m2).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(logsumexp_2d(x2), want_lse2, rtol=1e-12)
    np.testing.assert_allclose(online_softmax_2d(x2), want_sm2, rtol=1e-12)
    sm_spec = online_softmax_2d.specialize(x2)
    assert sm_spec.ir.count("alloc(") == 1 and sm_spec.ir.count("parallel[map]") == 1


def test_queried_target_profile_matrix_unit_and_device_array():
  from parakeet.target import MatrixUnit, TargetProfile

  # 1. CPU TargetProfile queries real host attributes
  cpu_prof = parakeet.target_profile("multicore")
  assert isinstance(cpu_prof, TargetProfile)
  assert cpu_prof.kind == "cpu"
  assert cpu_prof.num_compute_units >= 1
  assert cpu_prof.l1_bytes >= 16 * 1024
  assert cpu_prof.l2_bytes >= 64 * 1024
  assert cpu_prof.vector_width >= 4
  assert cpu_prof.prefer_interchange is True
  assert 128 <= cpu_prof.interchange_tile() <= 512

  # 2. TargetProfile heuristics for CUDA & Metal
  cuda_prof = TargetProfile(
    kind="cuda",
    name="NVIDIA L4",
    num_compute_units=58,
    vector_width=32,
    shared_mem_bytes=48 * 1024,
    max_threads_per_block=1024,
    unified_memory=False,
    supports_float64=True,
    compute_capability=(8, 9),
    matrix_unit=MatrixUnit("wmma", 16, 16, 8, ("float32",), ("float32",)),
    prefer_interchange=False,
    tensor_cores=True,
  )
  assert cuda_prof.shared_budget() == 48 * 1024
  assert cuda_prof.reduce_chunks(8192) == 58 * 32
  assert cuda_prof.reduce_chunks(1024) == 1024  # explicit user override respected

  # 3. GPU backends (Metal if available, CUDA if available or simulated):
  #    verify hardware matrix-unit codegen + persistent DeviceArray residency!
  @jit
  def matmul_f32(a, b):
    return a @ b

  @jit
  def fused_mlp_f32(a, b, bias):
    return np.tanh(np.sin(a) @ np.cos(b) + bias)

  @jit
  def batched_mm_f32(a, b):
    return a @ b

  @jit
  def sgd_step(params, x, y, lr):
    w1, w2 = params
    h = np.tanh(x @ w1)
    pred = h @ w2
    err = pred - y
    dw2 = h.T @ err
    dh = (err @ w2.T) * (np.float32(1.0) - h * h)
    dw1 = x.T @ dh
    return (w1 - lr * dw1, w2 - lr * dw2)

  @jit(backend="c")
  def cpu_sum(a):
    return np.sum(a)

  avail = parakeet.available_backends()
  gpu_backends = [b for b in ("metal", "cuda") if avail.get(b) == "ok"]
  rng = np.random.default_rng(77)

  for bname in gpu_backends:
    with parakeet.options(backend=bname, device_threshold=1, verify=True):
      prof = parakeet.target_profile(bname)
      assert prof.kind == bname
      assert prof.shared_mem_bytes >= 16 * 1024
      assert prof.prefer_interchange is False

      # Non-multiple-of-64 and large shapes, fused prologue+epilogue, and 3D batched matmul
      for M, N, K in ((7, 5, 13), (65, 63, 35), (128, 128, 96)):
        a = rng.standard_normal((M, K)).astype(np.float32) * 0.2
        b = rng.standard_normal((K, N)).astype(np.float32) * 0.2
        bias = rng.standard_normal((1, N)).astype(np.float32) * 0.1
        np.testing.assert_allclose(matmul_f32(a, b), a @ b, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(
          fused_mlp_f32(a, b, bias), np.tanh(np.sin(a) @ np.cos(b) + bias), rtol=1e-3, atol=1e-4
        )
      a3 = rng.standard_normal((3, 70, 35)).astype(np.float32) * 0.2
      b3 = rng.standard_normal((3, 35, 66)).astype(np.float32) * 0.2
      np.testing.assert_allclose(batched_mm_f32(a3, b3), a3 @ b3, rtol=1e-3, atol=1e-4)

      # Verify hardware matrix-unit kernel is actually emitted when supported
      if bname == "metal" and prof.matrix_unit is not None:
        spec_m = matmul_f32.specialize(a, b)
        assert any("simdgroup_multiply_accumulate" in k for k in spec_m.kernels)
        spec_f = fused_mlp_f32.specialize(a, b, bias)
        assert any("simdgroup_multiply_accumulate" in k for k in spec_f.kernels)
      if bname == "cuda" and prof.matrix_unit is not None:
        with parakeet.options(tensor_cores=True):
          a_tc = rng.standard_normal((65, 35)).astype(np.float32) * 0.2
          b_tc = rng.standard_normal((35, 63)).astype(np.float32) * 0.2
          bias_tc = rng.standard_normal((1, 63)).astype(np.float32) * 0.1
          np.testing.assert_allclose(matmul_f32(a_tc, b_tc), a_tc @ b_tc, rtol=2e-3, atol=2e-3)
          np.testing.assert_allclose(
            fused_mlp_f32(a_tc, b_tc, bias_tc), np.tanh(np.sin(a_tc) @ np.cos(b_tc) + bias_tc), rtol=2e-3, atol=2e-3
          )
          spec_tc = matmul_f32.specialize(a_tc, b_tc)
          assert any("mma_sync" in k for k in spec_tc.kernels)

      # Persistent DeviceArray residency across a multi-step training loop:
      # ZERO host->device uploads and ZERO device->host downloads during steady-state steps!
      w1_h = rng.standard_normal((64, 64)).astype(np.float32) * 0.1
      w2_h = rng.standard_normal((64, 64)).astype(np.float32) * 0.1
      x_h = rng.standard_normal((64, 64)).astype(np.float32) * 0.2
      y_h = rng.standard_normal((64, 64)).astype(np.float32) * 0.2
      lr = np.float32(0.01)

      params_d = parakeet.to_device((w1_h, w2_h), backend=bname)
      x_d = parakeet.to_device(x_h, backend=bname)
      y_d = parakeet.to_device(y_h, backend=bname)
      assert all(isinstance(p, parakeet.DeviceArray) and p.on_device for p in params_d)

      # Warmup compile + first step
      params_d = sgd_step(params_d, x_d, y_d, lr)
      assert all(isinstance(p, parakeet.DeviceArray) and p.on_device for p in params_d)

      dev = parakeet.api.get_backend(bname).device()
      up_before, dn_before = dev.uploads, dev.downloads
      for _ in range(3):
        params_d = sgd_step(params_d, x_d, y_d, lr)
      assert dev.uploads == up_before, f"expected 0 H2D uploads in resident loop, got {dev.uploads - up_before}"
      assert dev.downloads == dn_before, f"expected 0 D2H downloads in resident loop, got {dev.downloads - dn_before}"

      # Verify numerical match vs NumPy reference over 4 steps
      rw1, rw2 = w1_h.copy(), w2_h.copy()
      for _ in range(4):
        rh = np.tanh(x_h @ rw1)
        rerr = rh @ rw2 - y_h
        rdw2 = rh.T @ rerr
        rdh = (rerr @ rw2.T) * (np.float32(1.0) - rh * rh)
        rdw1 = x_h.T @ rdh
        rw1 = rw1 - lr * rdw1
        rw2 = rw2 - lr * rdw2
      got_w1, got_w2 = parakeet.to_host(params_d)
      np.testing.assert_allclose(got_w1, rw1, rtol=1e-3, atol=1e-4)
      np.testing.assert_allclose(got_w2, rw2, rtol=1e-3, atol=1e-4)

      # Cross-backend interop: passing a DeviceArray to CPU backend auto-syncs cleanly
      np.testing.assert_allclose(cpu_sum(params_d[0]), float(rw1.sum()), rtol=1e-4)
