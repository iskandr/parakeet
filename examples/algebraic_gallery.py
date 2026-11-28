"""Gallery of algebraic kernels in Parakeet.

Every featured kernel is shown in Parakeet's standard recurring styles:

- ``*_numpy``   — NumPy broadcasting and array expressions
- ``*_comp``    — Python comprehensions
- ``*_loop``    — explicit Python ``for`` loops
- ``*_par``     — explicit data-parallel operators (``par.map``, ``par.allpairs``,
                  ``par.fold``, ``par.imap``, ``par.ireduce``, ``par.scatter_add``,
                  ``par.scan``)
- ``*_helpers`` — wrapped ``par.nn`` helpers where applicable

Run directly to verify exactness across every flavor and benchmark on CPU::

    python -m examples.algebraic_gallery
"""

from __future__ import annotations

import operator
import time
from typing import NamedTuple

import numpy as np

import parakeet as par
from parakeet import value_and_grad

# ---------------------------------------------------------------------------
# 1. Exact 1-NN Search (Pairwise Fusion + Tiling)
# ---------------------------------------------------------------------------


# --8<-- [start:squared_distance]
def squared_distance(query, reference):
  return np.sum((query - reference) ** 2)


# --8<-- [end:squared_distance]


# --8<-- [start:nearest_numpy]
@par.jit
def nearest_numpy(queries, references):
  delta = queries[:, None, :] - references[None, :, :]
  return np.argmin(np.sum(delta**2, axis=2), axis=1)


# --8<-- [end:nearest_numpy]


# --8<-- [start:nearest_comp]
@par.jit
def nearest_comp(queries, references):
  return np.array([np.argmin([squared_distance(q, r) for r in references]) for q in queries])


# --8<-- [end:nearest_comp]


# --8<-- [start:nearest_loop]
@par.jit
def nearest_loop(queries, references):
  out = np.empty(len(queries), dtype=np.int64)
  for i in range(len(queries)):
    best_idx = 0
    best_dist = squared_distance(queries[i], references[0])
    for j in range(1, len(references)):
      dist = squared_distance(queries[i], references[j])
      if (np.isnan(dist) or dist < best_dist) and not np.isnan(best_dist):
        best_dist = dist
        best_idx = j
    out[i] = best_idx
  return out


# --8<-- [end:nearest_loop]


# --8<-- [start:nearest_par]
class Winner(NamedTuple):
  distance: np.float32
  index: np.int64


def first_minimum(left, right):
  # Keep the first NaN; among finite values, replace only on strict improvement.
  right_wins = np.isnan(right.distance) or right.distance < left.distance
  return right if right_wins and not np.isnan(left.distance) else left


@par.jit
def nearest_par(queries, references):
  scores = par.allpairs(squared_distance, queries, references)
  return par.fold(Winner, first_minimum, scores, axis=1, indexed=True, emit=lambda winner: winner.index)


# --8<-- [end:nearest_par]


# --8<-- [start:exact_1nn]
@par.jit
def exact_1nn(x_query, x_ref):
  """Exact nearest-neighbor index per query row via nested ``par.map``."""
  return par.map(
    lambda query: np.argmin(par.map(lambda reference: squared_distance(query, reference), x_ref)),
    x_query,
  )


# --8<-- [end:exact_1nn]

nearest_fold = nearest_par


# ---------------------------------------------------------------------------
# 2. Inverse-Distance Smoothing (Weighted Pairwise Fold)
# ---------------------------------------------------------------------------


# --8<-- [start:smooth_numpy]
@par.jit
def smooth_numpy(queries, references, observations):
  values = observations.reshape(references.shape[0])
  delta = queries[:, None, :] - references[None, :, :]
  weights = 1.0 / (1.0 + np.sum(delta**2, axis=2))
  return np.sum(weights * values, axis=1) / np.sum(weights, axis=1)


# --8<-- [end:smooth_numpy]


# --8<-- [start:smooth_comp]
@par.jit
def smooth_comp(queries, references, observations):
  values = observations.reshape(references.shape[0])

  def smooth_one(q):
    w = np.array([1.0 / (1.0 + squared_distance(q, r)) for r in references])
    return np.sum(w * values) / np.sum(w)

  return np.array([smooth_one(q) for q in queries])


# --8<-- [end:smooth_comp]


# --8<-- [start:smooth_loop]
@par.jit
def smooth_loop(queries, references, observations):
  values = observations.reshape(references.shape[0])
  out = np.empty(len(queries), dtype=queries.dtype)
  for i in range(len(queries)):
    w_sum = 0.0
    wv_sum = 0.0
    for j in range(len(references)):
      w = 1.0 / (1.0 + squared_distance(queries[i], references[j]))
      w_sum += w
      wv_sum += w * values[j]
    out[i] = wv_sum / w_sum
  return out


# --8<-- [end:smooth_loop]


# --8<-- [start:smooth_par]
class WeightedSum(NamedTuple):
  weight: np.float32
  weighted_value: np.float32


def add_sums(left, right):
  return WeightedSum(left.weight + right.weight, left.weighted_value + right.weighted_value)


@par.jit
def smooth_par(queries, references, observations):
  values = observations.reshape(references.shape[0])
  scores = par.allpairs(squared_distance, queries, references)

  def contribution(distance_squared, ref_index):
    weight = np.float32(1) / (np.float32(1) + distance_squared)
    return WeightedSum(weight, weight * values[ref_index])

  return par.fold(contribution, add_sums, scores, axis=1, indexed=True, emit=lambda t: t.weighted_value / t.weight)


# --8<-- [end:smooth_par]

smooth_fold = smooth_par


# ---------------------------------------------------------------------------
# 3. Gaussian / RBF Kernel Regression (Nadaraya-Watson)
# ---------------------------------------------------------------------------


# --8<-- [start:rbf_regression_numpy]
@par.jit
def rbf_regression_numpy(x_query, x_train, y_train, neg_gamma):
  d2 = np.sum((x_query[:, None, :] - x_train[None, :, :]) ** np.float32(2.0), axis=-1)
  s = d2 * neg_gamma
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ y_train


# --8<-- [end:rbf_regression_numpy]


# --8<-- [start:rbf_regression_helpers]
@par.jit
def rbf_regression_helpers(x_query, x_train, y_train, neg_gamma):
  d2 = np.sum((x_query[:, None, :] - x_train[None, :, :]) ** np.float32(2.0), axis=-1)
  return par.nn.softmax(d2 * neg_gamma, axis=-1) @ y_train


# --8<-- [end:rbf_regression_helpers]


# --8<-- [start:rbf_regression_comp]
@par.jit
def rbf_regression_comp(x_query, x_train, y_train, neg_gamma):
  def predict_one(q):
    s = np.array([squared_distance(q, x) * neg_gamma for x in x_train])
    e = np.exp(s - np.max(s))
    return (e / np.sum(e)) @ y_train

  return np.array([predict_one(q) for q in x_query])


# --8<-- [end:rbf_regression_comp]


# --8<-- [start:rbf_regression_loop]
@par.jit
def rbf_regression_loop(x_query, x_train, y_train, neg_gamma):
  n_q = len(x_query)
  n_tr = len(x_train)
  out = np.zeros((n_q, y_train.shape[1]), dtype=x_query.dtype)
  for i in range(n_q):
    m = -np.inf
    for j in range(n_tr):
      s = squared_distance(x_query[i], x_train[j]) * neg_gamma
      if s > m:
        m = s
    denom = 0.0
    for j in range(n_tr):
      w = np.exp(squared_distance(x_query[i], x_train[j]) * neg_gamma - m)
      denom += w
      out[i] += w * y_train[j]
    out[i] /= denom
  return out


# --8<-- [end:rbf_regression_loop]


# --8<-- [start:rbf_regression_par]
@par.jit
def rbf_regression_par(x_query, x_train, y_train, neg_gamma):
  s = par.allpairs(squared_distance, x_query, x_train) * neg_gamma
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ y_train


# --8<-- [end:rbf_regression_par]

rbf_kernel_regression = rbf_regression_numpy


# ---------------------------------------------------------------------------
# 4. GBDT Histogram Building + Prefix-Sum Best-Split Finding
# ---------------------------------------------------------------------------


# --8<-- [start:best_split_helper]
def _best_split_from_histograms(g_hist, h_hist, nbins, reg_lambda):
  g_left = np.cumsum(g_hist)
  h_left = np.cumsum(h_hist)
  g_right = g_left[nbins - 1] - g_left
  h_right = h_left[nbins - 1] - h_left
  gain = (g_left * g_left) / (h_left + reg_lambda) + (g_right * g_right) / (h_right + reg_lambda)
  best_bin = np.argmax(gain)
  return best_bin, gain[best_bin]


# --8<-- [end:best_split_helper]


# --8<-- [start:gbdt_split_numpy]
@par.jit
def gbdt_split_numpy(bin_ids, grads, hess, nbins, reg_lambda):
  g_hist = np.zeros(nbins, dtype=np.float32)
  h_hist = np.zeros(nbins, dtype=np.float32)
  np.add.at(g_hist, bin_ids, grads)
  np.add.at(h_hist, bin_ids, hess)
  return _best_split_from_histograms(g_hist, h_hist, nbins, reg_lambda)


# --8<-- [end:gbdt_split_numpy]


# --8<-- [start:gbdt_split_comp]
@par.jit
def gbdt_split_comp(bin_ids, grads, hess, nbins, reg_lambda):
  g_hist = np.array([np.sum(grads[bin_ids == b]) for b in range(nbins)])
  h_hist = np.array([np.sum(hess[bin_ids == b]) for b in range(nbins)])
  return _best_split_from_histograms(g_hist, h_hist, nbins, reg_lambda)


# --8<-- [end:gbdt_split_comp]


# --8<-- [start:gbdt_split_loop]
@par.jit
def gbdt_split_loop(bin_ids, grads, hess, nbins, reg_lambda):
  g_hist = np.zeros(nbins, dtype=np.float32)
  h_hist = np.zeros(nbins, dtype=np.float32)
  for i in range(len(bin_ids)):
    b = bin_ids[i]
    g_hist[b] += grads[i]
    h_hist[b] += hess[i]
  return _best_split_from_histograms(g_hist, h_hist, nbins, reg_lambda)


# --8<-- [end:gbdt_split_loop]


# --8<-- [start:gbdt_split_par]
@par.jit
def gbdt_split_par(bin_ids, grads, hess, nbins, reg_lambda):
  g_hist = par.scatter_add(nbins, bin_ids, grads)
  h_hist = par.scatter_add(nbins, bin_ids, hess)
  return _best_split_from_histograms(g_hist, h_hist, nbins, reg_lambda)


# --8<-- [end:gbdt_split_par]

gbdt_best_split_numpy = gbdt_split_numpy
gbdt_best_split_comprehension = gbdt_split_comp
gbdt_best_split_loop = gbdt_split_loop
gbdt_best_split = gbdt_split_par


# ---------------------------------------------------------------------------
# 5. Tropical Min-Plus Matrix Product
# ---------------------------------------------------------------------------


# --8<-- [start:tropical_min_plus_numpy]
@par.jit
def tropical_min_plus_numpy(a, b):
  return np.min(a[:, :, None] + b[None, :, :], axis=1)


# --8<-- [end:tropical_min_plus_numpy]


# --8<-- [start:tropical_min_plus_comp]
@par.jit
def tropical_min_plus_comp(a, b):
  return np.array([[np.min(a[i] + b[:, j]) for j in range(b.shape[1])] for i in range(a.shape[0])])


# --8<-- [end:tropical_min_plus_comp]


# --8<-- [start:tropical_min_plus_loop]
@par.jit
def tropical_min_plus_loop(a, b):
  m, k = a.shape
  _, n = b.shape
  out = np.empty((m, n), dtype=a.dtype)
  for i in range(m):
    for j in range(n):
      best = a[i, 0] + b[0, j]
      for r in range(1, k):
        val = a[i, r] + b[r, j]
        if val < best:
          best = val
      out[i, j] = best
  return out


# --8<-- [end:tropical_min_plus_loop]


# --8<-- [start:tropical_min_plus_par]
@par.jit
def tropical_min_plus_par(a, b):
  """One min-plus matrix product: C[i, j] = min_r (A[i, r] + B[r, j])."""
  m, k = a.shape
  _, n = b.shape
  return par.imap(
    lambda i, j: par.ireduce(lambda r: a[i, r] + b[r, j], k, combine=np.minimum),
    (m, n),
  )


# --8<-- [end:tropical_min_plus_par]

tropical_min_plus = tropical_min_plus_par


# ---------------------------------------------------------------------------
# 6. Exponential IIR Filter & Mamba-2 State-Space Recurrence
# ---------------------------------------------------------------------------


# --8<-- [start:iir_filter_numpy]
@par.jit
def iir_filter_numpy(a, b):
  # Closed-form prefix formula for strictly positive decay coefficients a > 0.
  p = np.cumprod(a)
  return p * np.cumsum(b / p)


# --8<-- [end:iir_filter_numpy]


# --8<-- [start:iir_filter_comp]
@par.jit
def iir_filter_comp(a, b):
  p = np.cumprod(a)
  scaled = b / p
  return np.array([p[t] * np.sum(scaled[: t + 1]) for t in range(len(a))])


# --8<-- [end:iir_filter_comp]


# --8<-- [start:iir_filter_loop]
@par.jit
def iir_filter_loop(a, b):
  # Works for arbitrary a (including 0 or negative); Law 4 rewrites to TileScan.
  T = a.shape[0]
  out = np.empty_like(b)
  s = np.float32(0.0)
  for t in range(T):
    s = a[t] * s + b[t]
    out[t] = s
  return out


# --8<-- [end:iir_filter_loop]


# --8<-- [start:iir_filter_par]
@par.jit
def iir_filter_par(a, b):
  p = par.scan(operator.mul, a)
  return p * par.scan(operator.add, b / p)


# --8<-- [end:iir_filter_par]

iir_exponential_filter = iir_filter_loop


# --8<-- [start:mamba2_ssd]
@par.jit
def mamba2_ssd(q, k, v, a):
  T = q.shape[0]
  out = np.empty((T, v.shape[1]), dtype=np.float32)
  S = np.zeros((k.shape[1], v.shape[1]), dtype=np.float32)
  for t in range(T):
    S = a[t] * S + k[t][:, None] * v[t][None, :]
    out[t] = q[t] @ S
  return out


# --8<-- [end:mamba2_ssd]


# ---------------------------------------------------------------------------
# 7. Causal Attention, Sliding-Window Attention, MLA, and GQA
# ---------------------------------------------------------------------------


# --8<-- [start:causal_attention_numpy]
@par.jit
def causal_attention_numpy(q, k, v):
  T = q.shape[-2]
  idx = np.arange(T)
  mask = idx[:, None] >= idx[None, :]
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))
  s = np.where(mask, np.matmul(q, np.swapaxes(k, -1, -2)) * scale, np.float32(-1e9))
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return np.matmul(e / np.sum(e, axis=-1, keepdims=True), v)


# --8<-- [end:causal_attention_numpy]


# --8<-- [start:causal_attention_helpers]
@par.jit
def causal_attention_helpers(q, k, v):
  return par.nn.scaled_dot_product_attention(q, k, v, causal=True)


# --8<-- [end:causal_attention_helpers]


# --8<-- [start:causal_attention_comp]
@par.jit
def causal_attention_comp(q, k, v):
  T = q.shape[-2]
  idx = np.arange(T)
  mask = idx[:, None] >= idx[None, :]
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))

  def one_head(qh, kh, vh):
    s = np.where(mask, (qh @ kh.T) * scale, np.float32(-1e9))
    e = np.exp(s - np.max(s, axis=-1, keepdims=True))
    return (e / np.sum(e, axis=-1, keepdims=True)) @ vh

  return np.array([[one_head(q[b, h], k[b, h], v[b, h]) for h in range(q.shape[1])] for b in range(q.shape[0])])


# --8<-- [end:causal_attention_comp]


# --8<-- [start:causal_attention_loop]
@par.jit
def causal_attention_loop(q, k, v):
  B, H, T, D = q.shape
  scale = np.float32(1.0 / np.sqrt(np.float32(D)))
  out = np.zeros((B, H, T, v.shape[-1]), dtype=q.dtype)
  for b in range(B):
    for h in range(H):
      for i in range(T):
        m = -np.inf
        for j in range(i + 1):
          score = np.sum(q[b, h, i] * k[b, h, j]) * scale
          if score > m:
            m = score
        denom = 0.0
        for j in range(i + 1):
          w = np.exp(np.sum(q[b, h, i] * k[b, h, j]) * scale - m)
          denom += w
          out[b, h, i] += w * v[b, h, j]
        out[b, h, i] /= denom
  return out


# --8<-- [end:causal_attention_loop]


# --8<-- [start:causal_attention_par]
@par.jit
def causal_attention_par(q, k, v):
  T = q.shape[-2]
  idx = np.arange(T)
  mask = idx[:, None] >= idx[None, :]
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))

  def one_head(qh, kh, vh):
    s = np.where(mask, (qh @ kh.T) * scale, np.float32(-1e9))
    e = np.exp(s - np.max(s, axis=-1, keepdims=True))
    return (e / np.sum(e, axis=-1, keepdims=True)) @ vh

  return par.map(lambda qb, kb, vb: par.map(one_head, qb, kb, vb), q, k, v)


# --8<-- [end:causal_attention_par]

causal_flash_attention = causal_attention_numpy


@par.jit
def sliding_window_attention(q, k, v, window):
  T = q.shape[-2]
  idx = np.arange(T)
  mask = (idx[:, None] >= idx[None, :]) & ((idx[:, None] - idx[None, :]) <= window)
  return par.nn.scaled_dot_product_attention(q, k, v, mask=mask)


def causal_flash_loss(q, k, v):
  return np.sum(causal_flash_attention(q, k, v) ** np.float32(2.0))


# --8<-- [start:deepseek_mla]
@par.jit
def deepseek_mla(q, c_kv, w_uk, w_uv):
  k = c_kv @ w_uk
  v = c_kv @ w_uv
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))
  s = (q @ k.T) * scale
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v


# --8<-- [end:deepseek_mla]


# --8<-- [start:gqa_attention]
@par.jit
def gqa_attention(q, k, v):
  # q: (B, H_kv, G, T_q, D), k, v: (B, H_kv, 1, T_kv, D)
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))
  s = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return np.matmul(e / np.sum(e, axis=-1, keepdims=True), v)


# --8<-- [end:gqa_attention]


# ---------------------------------------------------------------------------
# 8. Linear Cross-Entropy (Cut Cross-Entropy)
# ---------------------------------------------------------------------------


# --8<-- [start:cross_entropy_numpy]
@par.jit
def cross_entropy_numpy(h, w, targets):
  logits = h @ w
  m = np.max(logits, axis=-1, keepdims=True)
  z = logits - m
  lse = np.log(np.sum(np.exp(z), axis=-1))
  picked = z[np.arange(h.shape[0]), targets]
  return -np.mean(picked - lse)


# --8<-- [end:cross_entropy_numpy]


# --8<-- [start:cross_entropy_helpers]
@par.jit
def cross_entropy_helpers(h, w, targets):
  return par.nn.cross_entropy(h @ w, targets)


# --8<-- [end:cross_entropy_helpers]


# --8<-- [start:cross_entropy_comp]
@par.jit
def cross_entropy_comp(h, w, targets):
  def row_loss(row, tgt):
    logits = row @ w
    z = logits - np.max(logits)
    return np.log(np.sum(np.exp(z))) - z[tgt]

  return np.mean(np.array([row_loss(row, tgt) for row, tgt in zip(h, targets)]))


# --8<-- [end:cross_entropy_comp]


# --8<-- [start:cross_entropy_loop]
@par.jit
def cross_entropy_loop(h, w, targets):
  total = 0.0
  for i in range(len(targets)):
    logits = h[i] @ w
    z = logits - np.max(logits)
    total += np.log(np.sum(np.exp(z))) - z[targets[i]]
  return total / len(targets)


# --8<-- [end:cross_entropy_loop]


# --8<-- [start:cross_entropy_par]
@par.jit
def cross_entropy_par(h, w, targets):
  def row_loss(row, tgt):
    logits = row @ w
    z = logits - np.max(logits)
    return np.log(np.sum(np.exp(z))) - z[tgt]

  return np.mean(par.map(row_loss, h, targets))


# --8<-- [end:cross_entropy_par]

cut_cross_entropy = cross_entropy_numpy


def _bench(fn, args, iters=10):
  fn(*args)
  t0 = time.perf_counter()
  for _ in range(iters):
    out = fn(*args)
  return (time.perf_counter() - t0) / iters * 1e3, out


def main():
  rng = np.random.default_rng(42)
  print("Parakeet Algebraic Gallery — Verified Benchmarks (CPU)\n" + "=" * 68)

  # 1. Exact 1-NN (all four flavors)
  q_nn = rng.standard_normal((64, 16)).astype(np.float32)
  r_nn = rng.standard_normal((256, 16)).astype(np.float32)
  ref_nn = np.argmin(np.sum((q_nn[:, None, :] - r_nn[None, :, :]) ** np.float32(2), axis=2), axis=1)
  for fn in (nearest_numpy, nearest_comp, nearest_loop, nearest_par, exact_1nn):
    np.testing.assert_array_equal(fn(q_nn, r_nn), ref_nn)

  # 2. Inverse-distance smoothing (all four flavors)
  obs = rng.standard_normal((256,)).astype(np.float32)
  ref_sm = smooth_numpy(q_nn, r_nn, obs, _backend="python")
  for fn in (smooth_numpy, smooth_comp, smooth_loop, smooth_par):
    np.testing.assert_allclose(fn(q_nn, r_nn, obs), ref_sm, rtol=1e-4, atol=1e-4)

  # 3. RBF kernel regression (all five flavors)
  xq = rng.standard_normal((128, 16)).astype(np.float32)
  xt = rng.standard_normal((256, 16)).astype(np.float32)
  yt = rng.standard_normal((256, 8)).astype(np.float32)
  ng = np.float32(-0.05)
  ref_rbf = rbf_regression_numpy(xq, xt, yt, ng, _backend="python")
  for fn in (
    rbf_regression_numpy,
    rbf_regression_helpers,
    rbf_regression_comp,
    rbf_regression_loop,
    rbf_regression_par,
  ):
    np.testing.assert_allclose(fn(xq, xt, yt, ng), ref_rbf, rtol=1e-4, atol=1e-4)

  # 4. GBDT histogram split (all four flavors)
  nbins = 64
  bin_ids = rng.integers(0, nbins, size=(10_000,), dtype=np.int64)
  grads = rng.standard_normal(10_000).astype(np.float32)
  hess = rng.uniform(0.1, 1.0, size=(10_000,)).astype(np.float32)
  reg = np.float32(1.0)
  ref_bin, ref_gain = gbdt_split_par(bin_ids, grads, hess, nbins, reg)
  for fn in (gbdt_split_numpy, gbdt_split_comp, gbdt_split_loop, gbdt_split_par):
    b_out, g_out = fn(bin_ids, grads, hess, nbins, reg)
    assert int(b_out) == int(ref_bin)
    np.testing.assert_allclose(g_out, ref_gain, rtol=1e-4, atol=1e-4)

  # 5. Tropical min-plus product (all four flavors)
  ta = rng.uniform(0.1, 5.0, size=(48, 48)).astype(np.float32)
  tb = rng.uniform(0.1, 5.0, size=(48, 48)).astype(np.float32)
  ref_tr = np.min(ta[:, :, None] + tb[None, :, :], axis=1)
  for fn in (tropical_min_plus_numpy, tropical_min_plus_comp, tropical_min_plus_loop, tropical_min_plus_par):
    np.testing.assert_allclose(fn(ta, tb), ref_tr, rtol=1e-5, atol=1e-5)

  # 6. Exponential IIR filter (all four flavors)
  a_iir = rng.uniform(0.95, 0.99, size=(128,)).astype(np.float32)
  b_iir = rng.standard_normal(128).astype(np.float32)
  ref_iir = iir_filter_loop(a_iir, b_iir)
  for fn in (iir_filter_numpy, iir_filter_comp, iir_filter_loop, iir_filter_par):
    np.testing.assert_allclose(fn(a_iir, b_iir), ref_iir, rtol=1e-3, atol=1e-3)

  # 7. Causal attention (all five flavors)
  q = rng.standard_normal((1, 4, 64, 32)).astype(np.float32) * 0.2
  k = rng.standard_normal((1, 4, 64, 32)).astype(np.float32) * 0.2
  v = rng.standard_normal((1, 4, 64, 32)).astype(np.float32) * 0.2
  ref_attn = causal_attention_numpy(q, k, v, _backend="python")
  for fn in (
    causal_attention_numpy,
    causal_attention_helpers,
    causal_attention_comp,
    causal_attention_loop,
    causal_attention_par,
  ):
    np.testing.assert_allclose(fn(q, k, v), ref_attn, rtol=1e-4, atol=1e-4)

  # 8. Linear cross-entropy (all five flavors + value_and_grad)
  h = rng.standard_normal((64, 32)).astype(np.float32) * 0.2
  w = rng.standard_normal((32, 256)).astype(np.float32) * 0.2
  tgt = rng.integers(0, 256, size=(64,), dtype=np.int64)
  ref_ce = float(cross_entropy_numpy(h, w, tgt, _backend="python"))
  for fn in (
    cross_entropy_numpy,
    cross_entropy_helpers,
    cross_entropy_comp,
    cross_entropy_loop,
    cross_entropy_par,
  ):
    np.testing.assert_allclose(float(fn(h, w, tgt)), ref_ce, rtol=1e-4, atol=1e-4)

  lv, (dh, dw) = value_and_grad(cut_cross_entropy, argnum=(0, 1))(h, w, tgt)
  np.testing.assert_allclose(float(lv), ref_ce, rtol=1e-4, atol=1e-4)
  assert dh.shape == h.shape and dw.shape == w.shape

  # 9. MLA, GQA, and Mamba-2 SSD
  qm = rng.standard_normal((4, 64)).astype(np.float32) * 0.2
  cm = rng.standard_normal((128, 32)).astype(np.float32) * 0.2
  wuk = rng.standard_normal((32, 64)).astype(np.float32) * 0.2
  wuv = rng.standard_normal((32, 64)).astype(np.float32) * 0.2
  np.testing.assert_allclose(
    deepseek_mla(qm, cm, wuk, wuv), deepseek_mla(qm, cm, wuk, wuv, _backend="python"), rtol=1e-4, atol=1e-4
  )

  qg = rng.standard_normal((1, 2, 4, 32, 16)).astype(np.float32) * 0.2
  kg = rng.standard_normal((1, 2, 1, 32, 16)).astype(np.float32) * 0.2
  vg = rng.standard_normal((1, 2, 1, 32, 16)).astype(np.float32) * 0.2
  np.testing.assert_allclose(
    gqa_attention(qg, kg, vg), gqa_attention(qg, kg, vg, _backend="python"), rtol=1e-4, atol=1e-4
  )

  qs = rng.standard_normal((128, 32)).astype(np.float32) * 0.1
  ks = rng.standard_normal((128, 32)).astype(np.float32) * 0.1
  vs = rng.standard_normal((128, 32)).astype(np.float32) * 0.1
  as_ = rng.uniform(0.9, 0.99, size=(128,)).astype(np.float32)
  np.testing.assert_allclose(
    mamba2_ssd(qs, ks, vs, as_), mamba2_ssd(qs, ks, vs, as_, _backend="python"), rtol=1e-4, atol=1e-4
  )

  t_fa, _ = _bench(causal_attention_numpy, (q, k, v), iters=5)
  t_nn, _ = _bench(nearest_par, (q_nn, r_nn), iters=5)
  t_gb, _ = _bench(gbdt_split_par, (bin_ids, grads, hess, nbins, reg), iters=5)
  print(f"Verified all flavors | CausalAttn={t_fa:.2f}ms | 1-NN={t_nn:.2f}ms | GBDT={t_gb:.2f}ms")


if __name__ == "__main__":
  main()
