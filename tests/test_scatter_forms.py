"""Tests for declarative scatter-add forms:
- Filtered comprehension inversion:
  * [np.sum(vals[idx == b]) for b in range(n)] (and .sum(), np.where, extra & mask, counts)
  * [sum(vals[i] for i in range(N) if idx[i] == b) for b in range(n)] (and zip, extra if)
- Keyed generator expression scatter:
  * par.scatter_add(n, ((idx[i], vals[i]) for i in range(N) [if ...]))
  * par.scatter_add(n, ((b, v) for b, v in zip(idx, vals) [if ...]))
- Verification of O(N) IndexScatter lowering, 2 -> 1 sibling fusion, OOB semantics,
  and reverse-mode grad + forward-mode jvp differentiation.
"""

from __future__ import annotations

import numpy as np
import pytest

import parakeet as par


def test_filtered_numpy_comprehension_inversion_and_fusion():
  @par.jit
  def hist_masked(idx, vals, n):
    return np.array([np.sum(vals[idx == b]) for b in range(n)])

  @par.jit
  def hist_where(idx, vals, n):
    return np.array([np.sum(np.where(b == idx, vals, 0.0)) for b in range(n)])

  @par.jit
  def hist_method_and_mask(idx, vals, mask, n):
    return np.array([vals[(idx == b) & mask].sum() for b in range(n)])

  @par.jit
  def hist_counts(idx, n):
    return np.array([np.sum(idx == b) for b in range(n)])

  @par.jit
  def dual_gbdt_comp(idx, g, h, n):
    gh = np.array([np.sum(g[idx == b]) for b in range(n)])
    hh = np.array([np.sum(h[idx == b]) for b in range(n)])
    return gh, hh

  # Include out-of-range indices (-1, 5) to confirm exact pull-comprehension OOB ignore semantics
  idx = np.array([1, 3, -1, 1, 0, 5, 3, 2], dtype=np.int64)
  vals = np.array([1.5, 2.0, 99.0, 3.5, 4.0, -50.0, 5.0, 6.0], dtype=np.float64)
  hvals = np.array([0.5, 1.0, 10.0, 0.25, 2.0, 7.0, 1.5, 0.75], dtype=np.float64)
  mask = np.array([True, True, True, False, True, True, True, False])
  n = 4

  expected = np.array([np.sum(vals[idx == b]) for b in range(n)])
  expected_m = np.array([np.sum(vals[(idx == b) & mask]) for b in range(n)])
  expected_c = np.array([np.sum(idx == b) for b in range(n)], dtype=np.int64)

  np.testing.assert_allclose(hist_masked(idx, vals, n), expected, rtol=1e-12)
  np.testing.assert_allclose(hist_where(idx, vals, n), expected, rtol=1e-12)
  np.testing.assert_allclose(hist_method_and_mask(idx, vals, mask, n), expected_m, rtol=1e-12)
  np.testing.assert_array_equal(hist_counts(idx, n), expected_c)

  # Confirm O(N) IndexScatter lowering (not O(n * N) IndexMap-of-IndexFilter)
  rep = par.show_rewrite(hist_masked, idx, vals, n)
  assert rep.after_scatters == 1

  # Confirm sibling fusion (2 -> 1) on dual histogram comprehensions!
  gh, hh = dual_gbdt_comp(idx, vals, hvals, n)
  np.testing.assert_allclose(gh, expected, rtol=1e-12)
  np.testing.assert_allclose(hh, np.array([np.sum(hvals[idx == b]) for b in range(n)]), rtol=1e-12)
  rep_dual = par.show_rewrite(dual_gbdt_comp, idx, vals, hvals, n)
  assert rep_dual.before_scatters == 2 and rep_dual.after_scatters == 1


def test_filtered_generator_comprehension_inversion_and_fusion():
  @par.jit
  def hist_gen_range(idx, vals, n):
    return np.array([sum(vals[i] for i in range(len(idx)) if idx[i] == b) for b in range(n)])

  @par.jit
  def hist_gen_zip_masked(idx, vals, mask, n):
    return np.array([sum(v for k, v, m in zip(idx, vals, mask) if m and k == b) for b in range(n)])

  @par.jit
  def dual_gen_comp(idx, g, h, n):
    gh = np.array([sum(g[i] for i in range(len(idx)) if idx[i] == b) for b in range(n)])
    hh = np.array([sum(h[i] for i in range(len(idx)) if idx[i] == b) for b in range(n)])
    return gh, hh

  idx = np.array([2, 0, -2, 2, 1, 7, 0], dtype=np.int64)
  vals = np.array([1.0, 2.0, 5.0, 3.0, 4.0, 9.0, 6.0], dtype=np.float64)
  hvals = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], dtype=np.float64)
  mask = np.array([True, False, True, True, True, True, True])
  n = 4

  expected = np.array([sum(vals[i] for i in range(len(idx)) if idx[i] == b) for b in range(n)])
  expected_m = np.array([sum(v for k, v, m in zip(idx, vals, mask) if m and k == b) for b in range(n)])

  np.testing.assert_allclose(hist_gen_range(idx, vals, n), expected, rtol=1e-12)
  np.testing.assert_allclose(hist_gen_zip_masked(idx, vals, mask, n), expected_m, rtol=1e-12)

  rep = par.show_rewrite(hist_gen_range, idx, vals, n)
  assert rep.after_scatters == 1

  gh, hh = dual_gen_comp(idx, vals, hvals, n)
  np.testing.assert_allclose(gh, expected, rtol=1e-12)
  np.testing.assert_allclose(
    hh, np.array([sum(hvals[i] for i in range(len(idx)) if idx[i] == b) for b in range(n)]), rtol=1e-12
  )
  rep_dual = par.show_rewrite(dual_gen_comp, idx, vals, hvals, n)
  assert rep_dual.before_scatters == 2 and rep_dual.after_scatters == 1


def test_keyed_generator_scatter_add_jit_and_python():
  @par.jit
  def scatter_gen_range(idx, vals, mask, n):
    return par.scatter_add(n, ((idx[i], vals[i]) for i in range(len(idx)) if mask[i]))

  @par.jit
  def scatter_gen_zip(idx, vals, n):
    return par.scatter_add(n, ((k, v * 2.0) for k, v in zip(idx, vals) if v > 0.0))

  @par.jit
  def dual_scatter_gen(idx, g, h, mask, n):
    gh = par.scatter_add(n, ((idx[i], g[i]) for i in range(len(idx)) if mask[i]))
    hh = par.scatter_add(n, ((idx[i], h[i]) for i in range(len(idx)) if mask[i]))
    return gh, hh

  idx = np.array([1, 3, 99, 1, 0, 3], dtype=np.int64)
  vals = np.array([1.0, 2.0, 50.0, -3.0, 4.0, 5.0], dtype=np.float64)
  hvals = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)
  # Mask is False at the OOB index 99 -> must NOT raise IndexError!
  mask = np.array([True, True, False, True, True, True])
  n = 4

  expected_masked = np.array([4.0, -2.0, 0.0, 7.0])
  np.testing.assert_allclose(scatter_gen_range(idx, vals, mask, n), expected_masked, rtol=1e-12)

  # Direct call from plain Python (outside @par.jit)
  py_out = par.scatter_add(n, ((idx[i], vals[i]) for i in range(len(idx)) if mask[i]))
  np.testing.assert_allclose(py_out, expected_masked, rtol=1e-12)

  # Positive-only zip generator
  idx_ok = np.array([1, 3, 2, 1, 0, 3], dtype=np.int64)
  np.testing.assert_allclose(scatter_gen_zip(idx_ok, vals, n), np.array([8.0, 2.0, 100.0, 14.0]), rtol=1e-12)

  # Sibling fusion of two guarded keyed generators!
  gh, hh = dual_scatter_gen(idx, vals, hvals, mask, n)
  np.testing.assert_allclose(gh, expected_masked, rtol=1e-12)
  np.testing.assert_allclose(hh, np.array([4.5, 4.0, 0.0, 7.0]), rtol=1e-12)
  rep_dual = par.show_rewrite(dual_scatter_gen, idx, vals, hvals, mask, n)
  assert rep_dual.before_scatters == 2 and rep_dual.after_scatters == 1

  # Unguarded OOB index inside keyed generator still raises IndexError
  bad_mask = np.array([True, True, True, True, True, True])
  with pytest.raises(IndexError):
    scatter_gen_range(idx, vals, bad_mask, n)


def test_declarative_scatter_grad_and_jvp():
  idx = np.array([1, 3, -1, 1, 0, 5, 3], dtype=np.int64)
  vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
  mask = np.array([True, True, True, False, True, True, True])
  w = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
  dv = np.arange(len(vals), dtype=np.float64) + 0.5
  n = 4

  def f_comp(v):
    h = np.array([np.sum(v[(idx == b) & mask]) for b in range(n)])
    return np.sum(h * w)

  def f_gen_comp(v):
    h = np.array([sum(v[i] for i in range(len(idx)) if mask[i] and idx[i] == b) for b in range(n)])
    return np.sum(h * w)

  idx_ok = np.array([1, 3, 2, 1, 0, 2, 3], dtype=np.int64)

  def f_keyed_gen(v):
    h = par.scatter_add(n, ((idx_ok[i], v[i]) for i in range(len(idx_ok)) if mask[i]))
    return np.sum(h * w)

  # Expected gradient for f_comp / f_gen_comp: w[idx[i]] if (mask[i] and 0 <= idx[i] < n) else 0
  expected_g_oob = np.array([20.0, 40.0, 0.0, 0.0, 10.0, 0.0, 40.0])
  for fn in (f_comp, f_gen_comp):
    g = par.grad(fn)(vals)
    np.testing.assert_allclose(g, expected_g_oob, rtol=1e-12)
    _, df = par.jvp(fn, (vals,), (dv,))
    assert df == pytest.approx(float(np.sum(expected_g_oob * dv)), rel=1e-12)

  expected_g_ok = np.array([20.0, 40.0, 30.0, 0.0, 10.0, 30.0, 40.0])
  g_k = par.grad(f_keyed_gen)(vals)
  np.testing.assert_allclose(g_k, expected_g_ok, rtol=1e-12)
  _, df_k = par.jvp(f_keyed_gen, (vals,), (dv,))
  assert df_k == pytest.approx(float(np.sum(expected_g_ok * dv)), rel=1e-12)


def test_non_scatter_comprehensions_and_multicore():
  @par.jit
  def not_eq_comp(idx, vals, n):
    return np.array([np.sum(vals[idx != b]) for b in range(n)])

  @par.jit
  def b_dependent_val_comp(idx, vals, n):
    return np.array([np.sum(vals[idx == b] + float(b)) for b in range(n)])

  @par.jit
  def empty_range_comp(idx, vals, n):
    return np.array([np.sum(vals[idx == b]) for b in range(n)])

  idx = np.array([0, 1, 0, 2], dtype=np.int64)
  vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
  np.testing.assert_allclose(not_eq_comp(idx, vals, 3), np.array([6.0, 8.0, 6.0]), rtol=1e-12)
  np.testing.assert_allclose(b_dependent_val_comp(idx, vals, 3), np.array([4.0, 3.0, 6.0]), rtol=1e-12)
  assert empty_range_comp(idx, vals, -2).shape == (0,)

  # Check multicore backend agreement
  with par.options(backend="multicore"):
    out_mc = par.jit(lambda a, b_, m: np.array([np.sum(b_[a == k]) for k in range(m)]))(idx, vals, 3)
  np.testing.assert_allclose(out_mc, np.array([4.0, 2.0, 4.0]), rtol=1e-12)
