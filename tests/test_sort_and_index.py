"""Tests for #51: np.searchsorted, parallel np.sort/argsort/unique, SortedIndex, FrozenHashTable."""

from __future__ import annotations

import os

import numpy as np
import pytest

import parakeet as pk
from parakeet import FrozenHashTable, SortedIndex


# ---------------------------------------------------------------------------
# 1. Fusible np.searchsorted
# ---------------------------------------------------------------------------


def test_searchsorted_int_float_nan_empty_and_fused():
  @pk.jit
  def ss_left(a, v):
    return np.searchsorted(a, v, side="left")

  @pk.jit
  def ss_right(a, v):
    return np.searchsorted(a, v, side="right")

  # Duplicates & out-of-range
  a = np.array([1, 2, 2, 2, 5, 9], dtype=np.int64)
  v = np.array([-3, 1, 2, 3, 5, 9, 100], dtype=np.int64)
  np.testing.assert_array_equal(ss_left(a, v), np.searchsorted(a, v, side="left"))
  np.testing.assert_array_equal(ss_right(a, v), np.searchsorted(a, v, side="right"))
  assert int(ss_left(a, 2)) == int(np.searchsorted(a, 2, side="left"))
  assert int(ss_right(a, 2)) == int(np.searchsorted(a, 2, side="right"))

  # Empty array
  empty = np.empty(0, dtype=np.int64)
  np.testing.assert_array_equal(ss_left(empty, v), np.zeros(len(v), dtype=np.int64))

  # Floats with -inf, -0.0, +0.0, +inf, NaN
  af = np.array([-np.inf, -1.0, 0.0, 2.5, np.inf, np.nan], dtype=np.float64)
  vf = np.array([-np.inf, -2.0, -0.0, 0.0, 2.5, 3.0, np.inf, np.nan], dtype=np.float64)
  np.testing.assert_array_equal(ss_left(af, vf), np.searchsorted(af, vf, side="left"))
  np.testing.assert_array_equal(ss_right(af, vf), np.searchsorted(af, vf, side="right"))

  # Fused inside comprehension (interval overlap count)
  starts = np.array([10, 20, 30, 50, 80], dtype=np.int64)
  ends = np.array([15, 28, 45, 60, 90], dtype=np.int64)
  q_s = np.array([12, 25, 48], dtype=np.int64)
  q_e = np.array([35, 55, 85], dtype=np.int64)

  @pk.jit
  def count_overlaps(ex_s, ex_e, qs, qe):
    # Number of sorted non-overlapping exons overlapping [qs[i], qe[i])
    return np.searchsorted(ex_s, qe, side="left") - np.searchsorted(ex_e, qs, side="right")

  np.testing.assert_array_equal(
    count_overlaps(starts, ends, q_s, q_e),
    np.searchsorted(starts, q_e, side="left") - np.searchsorted(ends, q_s, side="right"),
  )


# ---------------------------------------------------------------------------
# 2. Stable parallel np.sort / np.argsort / np.unique across all dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  "dtype",
  [np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64],
)
def test_sort_argsort_all_dtypes_and_edge_cases(dtype):
  rng = np.random.default_rng(51)

  @pk.jit
  def do_sort(x):
    return np.sort(x), np.argsort(x)

  # Empty & length-1
  e0 = np.empty(0, dtype=dtype)
  s0, o0 = do_sort(e0)
  assert len(s0) == 0 and len(o0) == 0
  e1 = np.array([1], dtype=dtype)
  s1, o1 = do_sort(e1)
  np.testing.assert_array_equal(s1, e1)
  np.testing.assert_array_equal(o1, np.array([0], dtype=np.int64))

  # Small (insertion sort path) and medium (radix sort path)
  for n in (25, 500):
    if dtype is np.bool_:
      x = rng.integers(0, 2, size=n).astype(np.bool_)
    elif np.issubdtype(dtype, np.integer):
      info = np.iinfo(dtype)
      lo = max(info.min, -1000)
      hi = min(info.max, 1000)
      x = rng.integers(lo, hi, size=n, dtype=dtype)
      x[0] = info.min
      x[1] = info.max
    else:
      x = rng.standard_normal(n).astype(dtype)
      x[0] = np.nan
      x[1] = -np.inf
      x[2] = np.inf
      x[3] = dtype(-0.0)
      x[4] = dtype(0.0)
      x[5] = np.nan
    s, o = do_sort(x)
    ref_o = np.argsort(x, kind="stable")
    ref_s = x[ref_o]
    if np.issubdtype(dtype, np.floating):
      np.testing.assert_array_equal(np.isnan(s), np.isnan(ref_s))
      fin = ~np.isnan(ref_s)
      np.testing.assert_array_equal(s[fin], ref_s[fin])
    else:
      np.testing.assert_array_equal(s, ref_s)
    # Exact stability on duplicate keys
    np.testing.assert_array_equal(o, ref_o)


def test_sort_parallel_large_2d_and_unique():
  rng = np.random.default_rng(99)
  # Large parallel radix sort (n = 80_000 triggers multi-chunk parallel radix)
  big = rng.integers(-50000, 50000, size=80_000, dtype=np.int64)

  @pk.jit
  def sort_and_args(x):
    return np.sort(x), np.argsort(x)

  s_ser, o_ser = sort_and_args(big, _backend="c")
  s_par, o_par = sort_and_args(big, _backend="multicore")
  ref_o = np.argsort(big, kind="stable")
  np.testing.assert_array_equal(o_ser, ref_o)
  np.testing.assert_array_equal(o_par, ref_o)
  np.testing.assert_array_equal(s_par, big[ref_o])

  # 2D sort along axis=0 and axis=-1
  mat = rng.standard_normal((12, 17)).astype(np.float32)

  @pk.jit
  def sort_axes(m):
    return np.sort(m, axis=0), np.sort(m, axis=-1), np.argsort(m, axis=1)

  s0, s1, o1 = sort_axes(mat)
  np.testing.assert_array_equal(s0, np.sort(mat, axis=0))
  np.testing.assert_array_equal(s1, np.sort(mat, axis=-1))
  np.testing.assert_array_equal(o1, np.argsort(mat, axis=1, kind="stable"))

  # np.unique with return_index and return_counts (k-mer counting pattern!)
  kmers = np.array([42, 7, 42, 99, 7, 42, 13, 99], dtype=np.uint64)

  @pk.jit
  def count_kmers(xs):
    return np.unique(xs, return_index=True, return_counts=True)

  u, idx, cnt = count_kmers(kmers)
  ref_u, ref_idx, ref_cnt = np.unique(kmers, return_index=True, return_counts=True)
  np.testing.assert_array_equal(u, ref_u)
  np.testing.assert_array_equal(idx, ref_idx)
  np.testing.assert_array_equal(cnt, ref_cnt)


# ---------------------------------------------------------------------------
# 3. SortedIndex & FrozenHashTable (CPU + GPU)
# ---------------------------------------------------------------------------


def test_sorted_index_and_frozen_hashtable():
  keys = np.array([100, 10, 50, 10, 75, 50, 50], dtype=np.int64)
  vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
  si = SortedIndex.from_unsorted(keys, vals)

  @pk.jit
  def query_si(idx, queries):
    n = len(queries)
    out = np.empty(n, dtype=np.float32)
    hits = np.empty(n, dtype=np.bool_)
    for i in range(n):
      q = queries[i]
      hits[i] = q in idx
      out[i] = idx.get(q, np.float32(-1.0))
    r_lo, r_hi = idx.range(50, 76)
    return out, hits, r_lo, r_hi, len(idx)

  qs = np.array([5, 10, 50, 60, 75, 100, 200], dtype=np.int64)
  out, hits, r_lo, r_hi, sz = query_si(si, qs)
  np.testing.assert_array_equal(hits, np.array([False, True, True, False, True, True, False]))
  np.testing.assert_allclose(out, np.array([-1.0, 2.0, 3.0, -1.0, 5.0, 1.0, -1.0], dtype=np.float32))
  assert (int(r_lo), int(r_hi)) == si.range(50, 76)
  assert sz == len(keys)

  # FrozenHashTable with int64 and uint64 keys + collisions + empty table
  u_keys = np.array([17, 1024, 9999991, -42, 0, 123456789], dtype=np.int64)
  u_vals = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=np.float32)
  ht = FrozenHashTable.from_items(u_keys, u_vals)

  @pk.jit
  def query_ht(table, queries):
    n = len(queries)
    out = np.empty(n, dtype=np.float32)
    hits = np.empty(n, dtype=np.bool_)
    for i in range(n):
      q = queries[i]
      hits[i] = q in table
      out[i] = table.get(q, np.float32(-9.0))
    return out, hits

  q_ht = np.array([17, 18, -42, 0, 9999991, 555], dtype=np.int64)
  ht_out, ht_hits = query_ht(ht, q_ht)
  np.testing.assert_array_equal(ht_hits, np.array([True, False, True, True, True, False]))
  np.testing.assert_allclose(ht_out, np.array([1.5, -9.0, 4.5, 5.5, 3.5, -9.0], dtype=np.float32))

  # Empty FrozenHashTable & uint64 FrozenHashTable
  ht_empty = FrozenHashTable.from_items(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
  eo, eh = query_ht(ht_empty, q_ht)
  assert not np.any(eh)
  np.testing.assert_allclose(eo, np.full(len(q_ht), -9.0, dtype=np.float32))

  ht_u64 = FrozenHashTable.from_items(
    np.array([1, 1 << 62, (1 << 63) + 5], dtype=np.uint64),
    np.array([10.0, 20.0, 30.0], dtype=np.float32),
  )
  uo, uh = query_ht(ht_u64, np.array([1, 2, (1 << 63) + 5], dtype=np.uint64))
  np.testing.assert_array_equal(uh, np.array([True, False, True]))
  np.testing.assert_allclose(uo, np.array([10.0, -9.0, 30.0], dtype=np.float32))


@pytest.mark.parametrize("backend", ["cuda", "metal"])
def test_searchsorted_and_indices_on_gpu(backend):
  if backend == "cuda" and not os.environ.get("PARAKEET_CUDA_SIMULATE"):
    if pk.available_backends().get("cuda") != "ok":
      pytest.skip("CUDA not available")
  if backend == "metal" and pk.available_backends().get("metal") != "ok":
    pytest.skip("Metal not available")

  rng = np.random.default_rng(17)
  n_tab = 512
  keys = np.unique(rng.integers(0, 100_000, size=n_tab * 2, dtype=np.int64))[:n_tab]
  vals = rng.standard_normal(len(keys)).astype(np.float32)
  si = SortedIndex.from_unsorted(keys, vals)
  ht = FrozenHashTable.from_items(keys, vals)
  queries = rng.integers(0, 100_000, size=2048, dtype=np.int64)
  queries[:16] = keys[:16]

  @pk.jit(backend=backend)
  def gpu_lookup(s_idx, h_tab, qs):
    ss = np.searchsorted(s_idx.keys, qs, side="left")
    v_si = pk.map(lambda q: s_idx.get(q, np.float32(0.0)), qs)
    v_ht = pk.map(lambda q: h_tab.get(q, np.float32(0.0)), qs)
    return ss, v_si, v_ht

  si_d = pk.to_device(si, backend=backend)
  ht_d = pk.to_device(ht, backend=backend)
  qs_d = pk.to_device(queries, backend=backend)
  with pk.options(device_fallback="error", device_threshold=1):
    ss_h, v_si_h, v_ht_h = pk.to_host(gpu_lookup(si_d, ht_d, qs_d))

  np.testing.assert_array_equal(ss_h, np.searchsorted(si.keys, queries, side="left"))
  ref_vals = np.array([ht.get(int(q), np.float32(0.0)) for q in queries], dtype=np.float32)
  np.testing.assert_allclose(v_si_h, ref_vals, rtol=1e-6)
  np.testing.assert_allclose(v_ht_h, ref_vals, rtol=1e-6)
