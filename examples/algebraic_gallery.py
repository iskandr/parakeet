"""Enriched Gallery of Algebraic Kernels in Parakeet.

Run directly to verify exactness and benchmark every workload on CPU (and GPU
when available)::

    python examples/algebraic_gallery.py

The examples cover pairwise folds, normalized weighted sums, contraction
reassociation, affine scans, min-plus products and keyed histograms.
The documentation gallery imports its code snippets from this module."""

from __future__ import annotations

import time

import numpy as np

import parakeet
from parakeet import dsl, value_and_grad

# ---------------------------------------------------------------------------
# 1. Causal FlashAttention + Sliding-Window FlexAttention (Laws 1, 2, 5)
# ---------------------------------------------------------------------------


# --8<-- [start:causal_flash_attention]
@dsl.kernel
def causal_flash_attention(q, k, v):
  T = q.shape[-2]
  idx = np.arange(T)
  mask = idx[:, None] >= idx[None, :]
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))
  s = np.where(mask, np.matmul(q, np.swapaxes(k, -1, -2)) * scale, np.float32(-1e9))
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return np.matmul(e / np.sum(e, axis=-1, keepdims=True), v)


# --8<-- [end:causal_flash_attention]


@dsl.kernel
def sliding_window_attention(q, k, v, window):
  T = q.shape[-2]
  idx = np.arange(T)
  mask = (idx[:, None] >= idx[None, :]) & ((idx[:, None] - idx[None, :]) <= window)
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))
  s = np.where(mask, np.matmul(q, np.swapaxes(k, -1, -2)) * scale, np.float32(-1e9))
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return np.matmul(e / np.sum(e, axis=-1, keepdims=True), v)


def causal_flash_loss(q, k, v):
  return np.sum(causal_flash_attention(q, k, v) ** np.float32(2.0))


# ---------------------------------------------------------------------------
# 2. Apple Cut Cross-Entropy (zero M x V logit materialization, Laws 1 & 2)
# ---------------------------------------------------------------------------


# --8<-- [start:cut_cross_entropy]
def cut_cross_entropy(h, w, targets):
  logits = h @ w
  m = np.max(logits, axis=-1, keepdims=True)
  z = logits - m
  lse = np.log(np.sum(np.exp(z), axis=-1))
  picked = z[np.arange(h.shape[0]), targets]
  return -np.mean(picked - lse)


# --8<-- [end:cut_cross_entropy]


# ---------------------------------------------------------------------------
# 3. DeepSeek MLA Weight Absorption -> FlashMLA (Laws 3 + 1)
# ---------------------------------------------------------------------------


# --8<-- [start:deepseek_mla]
@dsl.kernel
def deepseek_mla(q, c_kv, w_uk, w_uv):
  k = c_kv @ w_uk
  v = c_kv @ w_uv
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))
  s = (q @ k.T) * scale
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v


# --8<-- [end:deepseek_mla]


# ---------------------------------------------------------------------------
# 4. Llama-3 GQA Shared-KV Head-Group Tiling (Laws 3 + 1)
# ---------------------------------------------------------------------------


# --8<-- [start:gqa_attention]
@dsl.kernel
def gqa_attention(q, k, v):
  # q: (B, H_kv, G, T_q, D), k, v: (B, H_kv, 1, T_kv, D)
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))
  s = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return np.matmul(e / np.sum(e, axis=-1, keepdims=True), v)


# --8<-- [end:gqa_attention]


# ---------------------------------------------------------------------------
# 5. Mamba-2 State-Space Duality (6-line recurrence -> 4 batched GEMMs, Law 4)
# ---------------------------------------------------------------------------


# --8<-- [start:mamba2_ssd]
@dsl.kernel
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
# 6. Parallel Prefix Exponential / IIR Filter (Law 4 TileScan)
# ---------------------------------------------------------------------------


# --8<-- [start:iir_exponential_filter]
@dsl.kernel
def iir_exponential_filter(a, b):
  T = a.shape[0]
  out = np.empty_like(b)
  s = np.float32(0.0)
  for t in range(T):
    s = a[t] * s + b[t]
    out[t] = s
  return out


# --8<-- [end:iir_exponential_filter]


# ---------------------------------------------------------------------------
# 7. Fused RBF / Gaussian Kernel Regression (Nadaraya-Watson, zero N x M!)
# ---------------------------------------------------------------------------


# --8<-- [start:rbf_kernel_regression]
@dsl.kernel
def rbf_kernel_regression(x_query, x_train, y_train, neg_gamma):
  d2 = np.sum((x_query[:, None, :] - x_train[None, :, :]) ** np.float32(2.0), axis=-1)
  s = d2 * neg_gamma
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ y_train


# --8<-- [end:rbf_kernel_regression]


# ---------------------------------------------------------------------------
# 8. Tropical Min-Plus Product & Exact 1-NN (Tiling + Carry)
# ---------------------------------------------------------------------------


# --8<-- [start:tropical_min_plus]
@dsl.kernel
def tropical_min_plus(a, b):
  """One min-plus matrix product: C[i,j] = min_k (A[i,k] + B[k,j])."""
  m, n = a.shape[0], b.shape[1]
  k = a.shape[1]
  return dsl.imap(
    lambda ij: dsl.ireduce(lambda r: a[ij[0], r] + b[r, ij[1]], k, combine=parakeet.minimum),
    (m, n),
  )


# --8<-- [end:tropical_min_plus]


# --8<-- [start:exact_1nn]
@dsl.kernel
def exact_1nn(x_query, x_ref):
  """Exact nearest-neighbor index per query row."""
  return dsl.map(
    lambda q: np.argmin(dsl.map(lambda r: np.sum((q - r) ** np.float32(2.0)), x_ref)),
    x_query,
  )


# --8<-- [end:exact_1nn]


# ---------------------------------------------------------------------------
# 9. GBDT Histogram Building + Prefix-Sum Best-Split Finding
# ---------------------------------------------------------------------------


# --8<-- [start:gbdt_best_split]
@dsl.kernel
def gbdt_best_split(bin_ids, grads, hess, nbins, reg_lambda):
  """Histogram scatter-add + prefix sums + argmax gain in 8 lines (XGBoost/LightGBM core)."""
  g_hist = dsl.iscatter(nbins, bin_ids, grads)
  h_hist = dsl.iscatter(nbins, bin_ids, hess)
  g_left = np.cumsum(g_hist)
  h_left = np.cumsum(h_hist)
  g_tot = g_left[nbins - 1]
  h_tot = h_left[nbins - 1]
  g_right = g_tot - g_left
  h_right = h_tot - h_left
  gain = (g_left * g_left) / (h_left + reg_lambda) + (g_right * g_right) / (h_right + reg_lambda)
  best_bin = np.argmax(gain)
  return best_bin, gain[best_bin]


# --8<-- [end:gbdt_best_split]


def _bench(fn, args, iters=10):
  fn(*args)
  t0 = time.perf_counter()
  for _ in range(iters):
    out = fn(*args)
  return (time.perf_counter() - t0) / iters * 1e3, out


def main():
  rng = np.random.default_rng(42)
  print("Parakeet Algebraic Gallery — Verified Benchmarks (CPU)\n" + "=" * 68)

  # 1. Causal FlashAttention
  q = rng.standard_normal((1, 4, 256, 64)).astype(np.float32) * 0.2
  k = rng.standard_normal((1, 4, 256, 64)).astype(np.float32) * 0.2
  v = rng.standard_normal((1, 4, 256, 64)).astype(np.float32) * 0.2
  rep_fa = dsl.show_rewrite(causal_flash_attention, q, k, v)
  with parakeet.options(blas=False):
    t_pk, o_pk = _bench(causal_flash_attention, (q, k, v), iters=5)

  def np_causal_attn(q, k, v):
    T = q.shape[-2]
    m = np.tril(np.ones((T, T), dtype=bool))
    s = np.where(m, np.matmul(q, np.swapaxes(k, -1, -2)) * np.float32(1.0 / np.sqrt(q.shape[-1])), np.float32(-1e9))
    e = np.exp(s - np.max(s, axis=-1, keepdims=True))
    return np.matmul(e / np.sum(e, axis=-1, keepdims=True), v)

  t_np, o_np = _bench(np_causal_attn, (q, k, v), iters=5)
  np.testing.assert_allclose(o_pk, o_np, rtol=1e-4, atol=1e-4)
  print(
    f"1. Causal FlashAttention (B=1,H=4,T=256,D=64) : PK={t_pk:6.2f}ms vs NumPy={t_np:6.2f}ms ({t_np / t_pk:4.1f}x) | {rep_fa.laws_triggered}"
  )

  # 2. Cut Cross-Entropy fwd+bwd
  h = rng.standard_normal((256, 64)).astype(np.float32) * 0.2
  w = rng.standard_normal((64, 4096)).astype(np.float32) * 0.2
  tgt = rng.integers(0, 4096, size=(256,), dtype=np.int64)
  cce_vg = value_and_grad(cut_cross_entropy, argnum=(0, 1))
  with parakeet.options(blas=False):
    t_pk, (lv, (dh, dw)) = _bench(cce_vg, (h, w, tgt), iters=5)

  def np_cce_vg(h, w, tgt):
    logits = h @ w
    z = logits - np.max(logits, axis=-1, keepdims=True)
    ez = np.exp(z)
    p = ez / np.sum(ez, axis=-1, keepdims=True)
    loss = -np.mean(z[np.arange(h.shape[0]), tgt] - np.log(np.sum(ez, axis=-1)))
    dlog = p.copy()
    dlog[np.arange(h.shape[0]), tgt] -= 1.0
    dlog /= np.float32(h.shape[0])
    return loss, (dlog @ w.T, h.T @ dlog)

  t_np, (lv_np, (dh_np, dw_np)) = _bench(np_cce_vg, (h, w, tgt), iters=5)
  np.testing.assert_allclose(lv, lv_np, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dh, dh_np, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dw, dw_np, rtol=1e-4, atol=1e-4)
  print(
    f"2. Cut Cross-Entropy fwd+bwd (M=256,V=4096)   : PK={t_pk:6.2f}ms vs NumPy={t_np:6.2f}ms ({t_np / t_pk:4.1f}x) | Laws 1 + 2"
  )

  # 3. DeepSeek MLA Decode
  qm = rng.standard_normal((4, 256)).astype(np.float32) * 0.2
  cm = rng.standard_normal((2048, 64)).astype(np.float32) * 0.2
  wuk = rng.standard_normal((64, 256)).astype(np.float32) * 0.2
  wuv = rng.standard_normal((64, 256)).astype(np.float32) * 0.2
  with parakeet.options(blas=False):
    t_pk, om_pk = _bench(deepseek_mla, (qm, cm, wuk, wuv), iters=10)

  def np_mla(q, c, wuk, wuv):
    k, v = c @ wuk, c @ wuv
    s = (q @ k.T) * np.float32(1.0 / np.sqrt(q.shape[-1]))
    e = np.exp(s - np.max(s, axis=-1, keepdims=True))
    return (e / np.sum(e, axis=-1, keepdims=True)) @ v

  t_np, om_np = _bench(np_mla, (qm, cm, wuk, wuv), iters=10)
  np.testing.assert_allclose(om_pk, om_np, rtol=1e-4, atol=1e-4)
  print(
    f"3. DeepSeek MLA Decode (Tq=4,Tkv=2048,dc=64)  : PK={t_pk:6.2f}ms vs NumPy={t_np:6.2f}ms ({t_np / t_pk:4.1f}x) | Laws 3 + 1"
  )

  # 4. Mamba-2 SSD
  qs = rng.standard_normal((2048, 64)).astype(np.float32) * 0.1
  ks = rng.standard_normal((2048, 64)).astype(np.float32) * 0.1
  vs = rng.standard_normal((2048, 64)).astype(np.float32) * 0.1
  as_ = rng.uniform(0.9, 0.99, size=(2048,)).astype(np.float32)
  with parakeet.options(blas=False):
    t_pk, os_pk = _bench(mamba2_ssd, (qs, ks, vs, as_), iters=5)

  def np_ssd(q, k, v, a):
    out = np.empty((q.shape[0], v.shape[1]), dtype=np.float32)
    S = np.zeros((k.shape[1], v.shape[1]), dtype=np.float32)
    for t in range(q.shape[0]):
      S = a[t] * S + np.outer(k[t], v[t])
      out[t] = q[t] @ S
    return out

  t_np, os_np = _bench(np_ssd, (qs, ks, vs, as_), iters=2)
  np.testing.assert_allclose(os_pk, os_np, rtol=1e-4, atol=1e-4)
  print(
    f"4. Mamba-2 SSD (T=2048, D=64, V=64)           : PK={t_pk:6.2f}ms vs NumPy={t_np:6.2f}ms ({t_np / t_pk:4.1f}x) | Law 4 SSD"
  )

  # 5. Fused RBF Kernel Regression (Nadaraya-Watson)
  xq = rng.standard_normal((512, 32)).astype(np.float32)
  xt = rng.standard_normal((2048, 32)).astype(np.float32)
  yt = rng.standard_normal((2048, 8)).astype(np.float32)
  ng = np.float32(-0.05)
  with parakeet.options(blas=False):
    t_pk, or_pk = _bench(rbf_kernel_regression, (xq, xt, yt, ng), iters=5)

  def np_rbf(xq, xt, yt, ng):
    d2 = np.sum((xq[:, None, :] - xt[None, :, :]) ** np.float32(2.0), axis=-1)
    s = d2 * ng
    e = np.exp(s - np.max(s, axis=-1, keepdims=True))
    return (e / np.sum(e, axis=-1, keepdims=True)) @ yt

  t_np, or_np = _bench(np_rbf, (xq, xt, yt, ng), iters=3)
  np.testing.assert_allclose(or_pk, or_np, rtol=1e-4, atol=1e-4)
  print(
    f"5. RBF Kernel Regression (512x2048x32)        : PK={t_pk:6.2f}ms vs NumPy={t_np:6.2f}ms ({t_np / t_pk:4.1f}x) | Law 1 (0 N*M!)"
  )

  # 6. Tropical Min-Plus Shortest Path
  ta = rng.uniform(0.1, 5.0, size=(256, 256)).astype(np.float32)
  tb = rng.uniform(0.1, 5.0, size=(256, 256)).astype(np.float32)
  with parakeet.options(blas=False):
    t_pk, ot_pk = _bench(tropical_min_plus, (ta, tb), iters=5)
  t_np, ot_np = _bench(lambda a, b: np.min(a[:, :, None] + b[None, :, :], axis=1), (ta, tb), iters=3)
  np.testing.assert_allclose(ot_pk, ot_np, rtol=1e-5, atol=1e-5)
  print(
    f"6. One Min-Plus Matrix Product (256x256)      : PK={t_pk:6.2f}ms vs NumPy={t_np:6.2f}ms ({t_np / t_pk:4.1f}x) | Tiled Semiring"
  )

  # 7. Exact 1-NN (zero N x M matrix!)
  with parakeet.options(blas=False):
    t_pk, nn_pk = _bench(exact_1nn, (xq, xt), iters=5)
  t_np, nn_np = _bench(
    lambda a, b: np.argmin(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1), axis=-1), (xq, xt), iters=3
  )
  np.testing.assert_array_equal(nn_pk, nn_np)
  print(
    f"7. Exact 1-NN Search (512x2048x32)            : PK={t_pk:6.2f}ms vs NumPy={t_np:6.2f}ms ({t_np / t_pk:4.1f}x) | Carry + Zero N*M"
  )

  # 8. GBDT Histogram + Prefix-Sum Best Split
  bins = rng.integers(0, 256, size=(200_000,), dtype=np.int64)
  grads = rng.standard_normal((200_000,)).astype(np.float32)
  hess = rng.uniform(0.1, 1.0, size=(200_000,)).astype(np.float32)
  lam = np.float32(1.0)
  with parakeet.options(blas=False):
    t_pk, (b_pk, g_pk) = _bench(gbdt_best_split, (bins, grads, hess, 256, lam), iters=10)

  def np_gbdt(bins, grads, hess, nbins, lam):
    gh = np.bincount(bins, weights=grads, minlength=nbins).astype(np.float32)
    hh = np.bincount(bins, weights=hess, minlength=nbins).astype(np.float32)
    gl, hl = np.cumsum(gh), np.cumsum(hh)
    gr, hr = gl[-1] - gl, hl[-1] - hl
    gain = (gl * gl) / (hl + lam) + (gr * gr) / (hr + lam)
    idx = int(np.argmax(gain))
    return idx, gain[idx]

  t_np, (b_np, g_np) = _bench(np_gbdt, (bins, grads, hess, 256, lam), iters=10)
  assert int(b_pk) == b_np
  np.testing.assert_allclose(g_pk, g_np, rtol=1e-4, atol=1e-4)
  print(
    f"8. GBDT Hist + Best-Split (N=200k, B=256)     : PK={t_pk:6.2f}ms vs NumPy={t_np:6.2f}ms ({t_np / t_pk:4.1f}x) | Scatter+Scan+Carry"
  )


if __name__ == "__main__":
  main()
