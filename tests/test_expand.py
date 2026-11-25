"""Tests for #49: 1-to-K flatmap / ragged compaction (IndexExpand) and filter_pairs."""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import pytest

import parakeet as pk
from parakeet import RaggedArray, dsl


class Edge(NamedTuple):
  src: int
  k: int
  val: float


# ---------------------------------------------------------------------------
# 1. flatmap (IndexExpand): edge cases, skewed counts, multi-output NamedTuple
# ---------------------------------------------------------------------------


def test_flatmap_basic_edge_cases_and_skew():
  @pk.jit
  def repeat_idx(counts):
    return dsl.flatmap(lambda c: c, lambda c, k: k + 1, counts)

  # Empty input (n == 0)
  r0 = repeat_idx(np.empty(0, dtype=np.int64))
  assert isinstance(r0, RaggedArray)
  assert len(r0) == 0
  np.testing.assert_array_equal(r0.offsets, np.array([0], dtype=np.int64))
  assert len(r0.values) == 0

  # All counts == 0
  rz = repeat_idx(np.zeros(5, dtype=np.int64))
  assert len(rz) == 5
  np.testing.assert_array_equal(rz.offsets, np.zeros(6, dtype=np.int64))
  assert len(rz.values) == 0

  # Alternating 0 and >0 counts
  c = np.array([0, 3, 0, 1, 0, 4, 2], dtype=np.int64)
  r = repeat_idx(c)
  np.testing.assert_array_equal(r.lengths, c)
  np.testing.assert_array_equal(r.values, np.array([1, 2, 3, 1, 1, 2, 3, 4, 1, 2], dtype=np.int64))

  # Skewed larger workload across serial and multicore
  rng = np.random.default_rng(13)
  big = np.where(rng.random(5000) < 0.05, rng.integers(10, 50, size=5000), rng.integers(0, 3, size=5000)).astype(
    np.int64
  )
  r_ser = repeat_idx(big, _backend="c")
  r_par = repeat_idx(big, _backend="multicore")
  np.testing.assert_array_equal(r_ser.offsets, r_par.offsets)
  np.testing.assert_array_equal(r_ser.values, r_par.values)
  assert int(r_par.offsets[-1]) == int(big.sum())


def test_flatmap_multi_output_namedtuple_and_count_only_dce():
  @pk.jit
  def expand_edges(xs):
    return dsl.flatmap(
      lambda x: int(x),
      lambda x, k: Edge(src=int(x), k=k, val=x * 0.5 + float(k)),
      xs,
    )

  xs = np.array([2.0, 0.0, 3.0], dtype=np.float64)
  r = expand_edges(xs)
  assert isinstance(r, RaggedArray)
  assert isinstance(r.values, Edge)
  np.testing.assert_array_equal(r.offsets, np.array([0, 2, 2, 5], dtype=np.int64))
  np.testing.assert_array_equal(r.values.src, np.array([2, 2, 3, 3, 3], dtype=np.int64))
  np.testing.assert_array_equal(r.values.k, np.array([0, 1, 0, 1, 2], dtype=np.int64))
  np.testing.assert_allclose(r.values.val, np.array([1.0, 2.0, 1.5, 2.5, 3.5]))

  # Count-only DCE: only offsets are returned -> output value allocation disappears from IR!
  @pk.jit
  def offsets_only(xs):
    rag = dsl.flatmap(lambda x: int(x), lambda x, k: x + float(k), xs)
    return rag.offsets

  spec = offsets_only.specialize(xs)
  assert "exp_v0" not in spec.ir
  np.testing.assert_array_equal(offsets_only(xs), np.array([0, 2, 2, 5], dtype=np.int64))


def test_flatmap_negative_count_and_purity_errors():
  @pk.jit
  def bad_count(c):
    return dsl.flatmap(lambda x: x, lambda x, k: k, c)

  with pytest.raises(ValueError, match="non-negative"):
    bad_count(np.array([2, -1, 3], dtype=np.int64))

  @pk.jit
  def impure_count(c, side):
    def cnt(x):
      side[0] = x
      return x

    return dsl.flatmap(cnt, lambda x, k: k, c)

  with pytest.raises(pk.UnsupportedError, match="pure"):
    impure_count(np.array([1, 2], dtype=np.int64), np.zeros(1, dtype=np.int64))


# ---------------------------------------------------------------------------
# 2. filter_pairs: 3D protein radius neighbor graph + AD w.r.t. coordinates
# ---------------------------------------------------------------------------


def test_filter_pairs_3d_radius_graph_and_grad():
  rng = np.random.default_rng(7)
  coords = rng.uniform(0.0, 12.0, size=(32, 3)).astype(np.float64)
  cutoff2 = 25.0

  @pk.jit
  def radius_neighbors(pts, r2):
    def within(a, b):
      dx = a[0] - b[0]
      dy = a[1] - b[1]
      dz = a[2] - b[2]
      d2 = dx * dx + dy * dy + dz * dz
      return (d2 > 0.0) and (d2 < r2)

    def dist(a, b):
      dx = a[0] - b[0]
      dy = a[1] - b[1]
      dz = a[2] - b[2]
      return np.sqrt(dx * dx + dy * dy + dz * dz)

    return dsl.filter_pairs(within, pts, pts, value_fn=dist)

  rag = radius_neighbors(coords, cutoff2)
  dst, dists = rag.values
  # Check exact match against dense O(N^2) reference
  diff = coords[:, None, :] - coords[None, :, :]
  d2_mat = np.sum(diff * diff, axis=-1)
  mask = (d2_mat > 0.0) & (d2_mat < cutoff2)
  for i in range(len(coords)):
    ref_j = np.flatnonzero(mask[i])
    s, e = int(rag.offsets[i]), int(rag.offsets[i + 1])
    np.testing.assert_array_equal(dst[s:e], ref_j)
    np.testing.assert_allclose(dists[s:e], np.sqrt(d2_mat[i, ref_j]), rtol=1e-12)

  # Differentiate pairwise spring energy over sparse radius graph w.r.t. coords!
  def spring_energy(pts):
    def within(a, b):
      dx = a[0] - b[0]
      dy = a[1] - b[1]
      dz = a[2] - b[2]
      d2 = dx * dx + dy * dy + dz * dz
      return (d2 > 0.0) and (d2 < cutoff2)

    def energy(a, b):
      dx = a[0] - b[0]
      dy = a[1] - b[1]
      dz = a[2] - b[2]
      r = np.sqrt(dx * dx + dy * dy + dz * dz)
      return (r - 3.0) * (r - 3.0)

    edges = dsl.filter_pairs(within, pts, pts, value_fn=energy)
    return np.sum(edges.values[1])

  g = pk.grad(spring_energy)(coords)
  # Finite-difference check on fixed neighbor graph
  eps = 1e-5
  ref_g = np.zeros_like(coords)
  for i in range(coords.shape[0]):
    for c in range(3):
      p_plus = coords.copy()
      p_plus[i, c] += eps
      p_minus = coords.copy()
      p_minus[i, c] -= eps
      d_p = np.sqrt(np.sum((p_plus[:, None, :] - p_plus[None, :, :]) ** 2, axis=-1))
      d_m = np.sqrt(np.sum((p_minus[:, None, :] - p_minus[None, :, :]) ** 2, axis=-1))
      ep = float(np.sum(((d_p - 3.0) ** 2)[mask]))
      em = float(np.sum(((d_m - 3.0) ** 2)[mask]))
      ref_g[i, c] = (ep - em) / (2 * eps)
  np.testing.assert_allclose(g, ref_g, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. GPU execution (CUDA simulator + native Metal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["cuda", "metal"])
def test_flatmap_and_filter_pairs_on_gpu(backend):
  if backend == "cuda" and not os.environ.get("PARAKEET_CUDA_SIMULATE"):
    if pk.available_backends().get("cuda") != "ok":
      pytest.skip("CUDA not available")
  if backend == "metal" and pk.available_backends().get("metal") != "ok":
    pytest.skip("Metal not available")

  @pk.jit(backend=backend)
  def gpu_flatmap(counts):
    return dsl.flatmap(lambda c: c, lambda c, k: np.float32(c) + np.float32(k), counts)

  @pk.jit(backend=backend)
  def gpu_pairs(pts):
    def pred(a, b):
      d = a - b
      d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
      return (d2 > np.float32(0.0)) and (d2 < np.float32(4.0))

    return dsl.filter_pairs(pred, pts, pts)

  rng = np.random.default_rng(23)
  counts = rng.integers(0, 6, size=128, dtype=np.int64)
  pts = rng.uniform(0.0, 5.0, size=(64, 3)).astype(np.float32)

  with pk.options(device_fallback="error", device_threshold=1):
    rf = gpu_flatmap(counts)
    rp = gpu_pairs(pts)

  np.testing.assert_array_equal(rf.lengths, counts)
  assert len(rf.values) == int(counts.sum())
  diff = pts[:, None, :] - pts[None, :, :]
  d2 = np.sum(diff * diff, axis=-1)
  mask = (d2 > 0.0) & (d2 < 4.0)
  np.testing.assert_array_equal(rp.lengths, mask.sum(axis=1))
