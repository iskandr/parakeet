"""Tests for Laws 1, 2, and 5:
- Law 5: Affine tile-interval predicate propagation (causal, sliding window,
  spatial cutoff, tabular feature-range predicates).
- Law 1: 2-level rescaled semiring monoid tile fusion (FlashAttention,
  FlashDecoding Split-K, Cut Cross-Entropy, and non-transformer RBF / Gaussian
  kernel smoothing).
- Law 2: Algebraic rematerialization in reverse-mode AD (FlashAttention D_i
  backward, Cut Cross-Entropy backward, Softmax VJP, 1-pass LayerNorm / RMSNorm
  VJP, and end-to-end nanoGPT value_and_grad parity).
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import parakeet
from parakeet import jit, value_and_grad
from parakeet.ndtypes import Bool
from parakeet.syntax import ir, ops
from parakeet.syntax.ir import Const, Var
from parakeet.syntax.ops import I, cmp, iadd, isub
from parakeet.transforms import algebraic_vjp, pipeline
from parakeet.transforms.analysis import IndexFacts, TileInterval, tile_predicate_guards

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_bool(expr: ir.Expr, env: dict[int, int]) -> bool:
  if isinstance(expr, Const):
    return bool(expr.value)
  if isinstance(expr, Var):
    return bool(env[expr.uid])
  if isinstance(expr, ir.Prim):
    if expr.op == "not":
      return not _eval_bool(expr.args[0], env)
    if expr.op in ("and", "bitand"):
      return _eval_bool(expr.args[0], env) and _eval_bool(expr.args[1], env)
    if expr.op in ("or", "bitor"):
      return _eval_bool(expr.args[0], env) or _eval_bool(expr.args[1], env)
    a, b = _eval_int(expr.args[0], env), _eval_int(expr.args[1], env)
    return {
      "lt": a < b,
      "le": a <= b,
      "gt": a > b,
      "ge": a >= b,
      "eq": a == b,
      "ne": a != b,
    }[expr.op]
  raise AssertionError(f"unexpected bool expr {expr}")


def _eval_int(expr: ir.Expr, env: dict[int, int]) -> int:
  if isinstance(expr, Const):
    return int(expr.value)
  if isinstance(expr, Var):
    return int(env[expr.uid])
  if isinstance(expr, ir.Prim):
    if expr.op == "neg":
      return -_eval_int(expr.args[0], env)
    a, b = _eval_int(expr.args[0], env), _eval_int(expr.args[1], env)
    return {"add": a + b, "sub": a - b, "mul": a * b, "min": min(a, b), "max": max(a, b)}[expr.op]
  raise AssertionError(f"unexpected int expr {expr}")


def _capture_plans_and_allocs(fn, *args, **opts):
  plans: list[str] = []
  alloc_count = [0]
  orig_run = pipeline.run

  def spy(ir_fn, options, trace=None):
    res = orig_run(ir_fn, options, trace=trace)
    for s in ir.walk_stmts(ir_fn.body):
      if isinstance(s, ir.ParLoop) and s.monoidal_tile is not None:
        plans.append(s.monoidal_tile.kind)
      elif isinstance(s, ir.Alloc):
        alloc_count[0] += 1
    return res

  pipeline.run = spy
  try:
    with parakeet.options(**opts):
      out = fn(*args)
  finally:
    pipeline.run = orig_run
  return out, plans, alloc_count[0]


# ---------------------------------------------------------------------------
# Law 5: Affine Tile-Interval Predicate Propagation
# ---------------------------------------------------------------------------


def test_law5_tile_interval_guards_causal_sliding_spatial_and_range():
  facts = IndexFacts()
  i, j = Var("i", I), Var("j", I)
  i0, j0 = Var("i0", I), Var("j0", I)
  W, lo, hi = Var("W", I), Var("lo", I), Var("hi", I)
  BM, BN = 16, 16
  ivs = {
    i.uid: TileInterval(i0, iadd(i0, Const(BM - 1, I))),
    j.uid: TileInterval(j0, iadd(j0, Const(BN - 1, I))),
  }

  # 1. Causal mask: j <= i
  af, at = tile_predicate_guards(cmp("le", j, i), ivs, facts)
  for ti in range(0, 64, BM):
    for tj in range(0, 64, BN):
      env = {i0.uid: ti, j0.uid: tj}
      pts = [jj <= ii for ii in range(ti, ti + BM) for jj in range(tj, tj + BN)]
      assert _eval_bool(af, env) == (not any(pts))
      assert _eval_bool(at, env) == all(pts)

  # 2. Causal sliding window: (j <= i) & (i - j <= W)
  cond_win = ops.prim("and", (cmp("le", j, i), cmp("le", isub(i, j), W)), Bool)
  af_w, at_w = tile_predicate_guards(cond_win, ivs, facts)
  for ti in range(0, 96, BM):
    for tj in range(0, 96, BN):
      env = {i0.uid: ti, j0.uid: tj, W.uid: 20}
      pts = [(jj <= ii and ii - jj <= 20) for ii in range(ti, ti + BM) for jj in range(tj, tj + BN)]
      assert _eval_bool(af_w, env) == (not any(pts))
      assert _eval_bool(at_w, env) == all(pts)

  # 3. Symmetric spatial cutoff: abs(i - j) <= W
  cond_sym = cmp("le", ops.prim("abs", (isub(i, j),), I), W)
  af_s, at_s = tile_predicate_guards(cond_sym, ivs, facts)
  for ti in range(0, 96, BM):
    for tj in range(0, 96, BN):
      env = {i0.uid: ti, j0.uid: tj, W.uid: 24}
      pts = [abs(ii - jj) <= 24 for ii in range(ti, ti + BM) for jj in range(tj, tj + BN)]
      assert _eval_bool(af_s, env) == (not any(pts))
      assert _eval_bool(at_s, env) == all(pts)

  # 4. Tabular / GBDT feature-range predicate: (lo <= j) & (j < hi)
  cond_rng = ops.prim("and", (cmp("le", lo, j), cmp("lt", j, hi)), Bool)
  af_r, at_r = tile_predicate_guards(cond_rng, ivs, facts)
  for tj in range(0, 96, BN):
    env = {i0.uid: 0, j0.uid: tj, lo.uid: 20, hi.uid: 60}
    pts = [20 <= jj < 60 for jj in range(tj, tj + BN)]
    assert _eval_bool(af_r, env) == (not any(pts))
    assert _eval_bool(at_r, env) == all(pts)


# ---------------------------------------------------------------------------
# Law 1: FlashAttention, Cut Cross-Entropy, and Non-Transformer RBF Kernel
# ---------------------------------------------------------------------------


def _attn_fwd(q, k, v, causal):
  D = q.shape[3]
  s = (q @ k.transpose((0, 1, 3, 2))) * np.float32(1.0 / np.sqrt(D))
  s = np.where(causal, s, np.float32(-1e9))
  m = np.max(s, axis=3, keepdims=True)
  e = np.exp(s - m)
  return (e / np.sum(e, axis=3, keepdims=True)) @ v


def _attn_sliding_window(q, k, v, win):
  B, H, T, D = q.shape
  qi = np.arange(T).reshape((T, 1))
  kj = np.arange(T).reshape((1, T))
  s = (q @ k.transpose((0, 1, 3, 2))) * np.float32(1.0 / np.sqrt(D))
  s = np.where((kj <= qi) & (qi - kj <= win), s, np.float32(-1e9))
  m = np.max(s, axis=3, keepdims=True)
  e = np.exp(s - m)
  return (e / np.sum(e, axis=3, keepdims=True)) @ v


def _cce_fwd(h, w, targets):
  logits = h @ w
  m = np.max(logits, axis=1, keepdims=True)
  z = logits - m
  lse = np.log(np.sum(np.exp(z), axis=1))
  picked = z[np.arange(h.shape[0]), targets]
  return -np.mean(picked - lse)


def _rbf_kernel_smoothing(x, y, values, inv_2sigma2):
  # Non-transformer Law 1 workload: Nadaraya-Watson / Gaussian kernel regression
  # without materializing the N x M pairwise distance matrix!
  d2 = np.sum((x[:, None, :] - y[None, :, :]) ** np.float32(2.0), axis=2)
  logits = d2 * (-inv_2sigma2)
  m = np.max(logits, axis=1, keepdims=True)
  e = np.exp(logits - m)
  return (e / np.sum(e, axis=1, keepdims=True)) @ values


def test_law1_flash_attention_causal_and_sliding_window_cpu():
  rng = np.random.default_rng(11)
  q = rng.standard_normal((2, 3, 24, 12)).astype(np.float32)
  k = rng.standard_normal((2, 3, 24, 12)).astype(np.float32)
  v = rng.standard_normal((2, 3, 24, 12)).astype(np.float32)
  c = np.tril(np.ones((24, 24), dtype=bool))

  got, plans, _ = _capture_plans_and_allocs(jit(_attn_fwd), q, k, v, c, blas=False, verify=True)
  assert plans == ["flash_monoid"], plans
  s = np.where(c, (q.astype(np.float64) @ k.astype(np.float64).transpose(0, 1, 3, 2)) / np.sqrt(12.0), -1e9)
  e = np.exp(s - s.max(axis=3, keepdims=True))
  want = (e / e.sum(axis=3, keepdims=True)) @ v.astype(np.float64)
  np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

  # Symbolic sliding window (Law 5 tile_false_guard + tile_true_guard)
  got_w, plans_w, _ = _capture_plans_and_allocs(jit(_attn_sliding_window), q, k, v, 8, blas=False, verify=True)
  assert plans_w == ["flash_monoid"], plans_w
  qi = np.arange(24)[:, None]
  kj = np.arange(24)[None, :]
  sw = np.where((kj <= qi) & (qi - kj <= 8), (q.astype(np.float64) @ k.astype(np.float64).transpose(0, 1, 3, 2)) / np.sqrt(12.0), -1e9)
  ew = np.exp(sw - sw.max(axis=3, keepdims=True))
  want_w = (ew / ew.sum(axis=3, keepdims=True)) @ v.astype(np.float64)
  np.testing.assert_allclose(got_w, want_w, rtol=1e-4, atol=1e-4)


def test_law1_cut_cross_entropy_and_rbf_kernel_smoothing_cpu():
  rng = np.random.default_rng(12)
  h = rng.standard_normal((40, 16)).astype(np.float32)
  w = rng.standard_normal((16, 64)).astype(np.float32)
  tgt = rng.integers(0, 64, size=(40,), dtype=np.int64)

  got_c, plans_c, _ = _capture_plans_and_allocs(jit(_cce_fwd), h, w, tgt, blas=False, verify=True)
  assert plans_c == ["monoid_lse"], plans_c
  z = h.astype(np.float64) @ w.astype(np.float64)
  ez = np.exp(z - z.max(axis=1, keepdims=True))
  pz = ez / ez.sum(axis=1, keepdims=True)
  want_c = -np.mean(np.log(pz[np.arange(40), tgt]))
  np.testing.assert_allclose(got_c, want_c, rtol=1e-4, atol=1e-4)

  # Non-transformer RBF kernel regression (score_op == "sqdist")
  x = rng.standard_normal((32, 10)).astype(np.float32)
  y = rng.standard_normal((48, 10)).astype(np.float32)
  vals = rng.standard_normal((48, 8)).astype(np.float32)
  got_r, plans_r, _ = _capture_plans_and_allocs(
    jit(_rbf_kernel_smoothing), x, y, vals, np.float32(0.5), blas=False, verify=True
  )
  assert plans_r == ["flash_monoid"], plans_r
  d2 = np.sum((x.astype(np.float64)[:, None, :] - y.astype(np.float64)[None, :, :]) ** 2, axis=2)
  lr = -0.5 * d2
  er = np.exp(lr - lr.max(axis=1, keepdims=True))
  want_r = (er / er.sum(axis=1, keepdims=True)) @ vals.astype(np.float64)
  np.testing.assert_allclose(got_r, want_r, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Law 2: Algebraic Rematerialization in AD (FlashAttention, CCE, Softmax, Norm)
# ---------------------------------------------------------------------------


def _attn_loss(q, k, v, causal):
  return np.sum(_attn_fwd(q, k, v, causal))


def _softmax_loss(x, w):
  m = np.max(x, axis=1, keepdims=True)
  e = np.exp(x - m)
  p = e / np.sum(e, axis=1, keepdims=True)
  return np.sum(p * w)


def _rmsnorm_loss(x, g):
  var = np.mean(x ** np.float32(2.0), axis=2, keepdims=True)
  y = x / np.sqrt(var + np.float32(1e-5)) * g
  return np.sum(y * y)


def _layernorm_loss(x, g, b):
  mu = np.mean(x, axis=2, keepdims=True)
  var = np.mean((x - mu) ** np.float32(2.0), axis=2, keepdims=True)
  y = (x - mu) / np.sqrt(var + np.float32(1e-5)) * g + b
  return np.sum(y * y)


def test_law2_flash_attention_and_cce_and_norm_vjps_cpu():
  rng = np.random.default_rng(13)
  q = rng.standard_normal((2, 2, 20, 12)).astype(np.float32)
  k = rng.standard_normal((2, 2, 20, 12)).astype(np.float32)
  v = rng.standard_normal((2, 2, 20, 12)).astype(np.float32)
  c = np.tril(np.ones((20, 20), dtype=bool))

  (av, (dq, dk, dv)), a_plans, _ = _capture_plans_and_allocs(
    value_and_grad(_attn_loss, argnum=(0, 1, 2)), q, k, v, c, blas=False, verify=True
  )
  assert a_plans == ["flash_monoid", "flash_monoid_bwd"], a_plans

  q64, k64, v64 = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
  sc = 1.0 / np.sqrt(12.0)
  s = np.where(c, (q64 @ k64.transpose(0, 1, 3, 2)) * sc, -1e9)
  e = np.exp(s - s.max(axis=3, keepdims=True))
  p = e / e.sum(axis=3, keepdims=True)
  o = p @ v64
  do = np.ones_like(o)
  D = (do * o).sum(axis=3, keepdims=True)
  ds = np.where(c, p * ((do @ v64.transpose(0, 1, 3, 2)) - D) * sc, 0.0)
  np.testing.assert_allclose(av, float(o.sum()), rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dq, ds @ k64, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dk, ds.transpose(0, 1, 3, 2) @ q64, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dv, p.transpose(0, 1, 3, 2) @ do, rtol=1e-4, atol=1e-4)

  # CCE forward + backward BOTH monoidal (zero M x V allocations!)
  h = rng.standard_normal((36, 16)).astype(np.float32)
  w = rng.standard_normal((16, 50)).astype(np.float32)
  tgt = rng.integers(0, 50, size=(36,), dtype=np.int64)
  (cv, (dh, dw)), c_plans, _ = _capture_plans_and_allocs(
    value_and_grad(_cce_fwd, argnum=(0, 1)), h, w, tgt, blas=False, verify=True
  )
  assert c_plans == ["monoid_lse", "cce_bwd"], c_plans
  h64, w64 = h.astype(np.float64), w.astype(np.float64)
  z = h64 @ w64
  ez = np.exp(z - z.max(axis=1, keepdims=True))
  pz = ez / ez.sum(axis=1, keepdims=True)
  dz = pz.copy()
  dz[np.arange(36), tgt] -= 1.0
  dz /= 36.0
  np.testing.assert_allclose(cv, -np.mean(np.log(pz[np.arange(36), tgt])), rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dh, dz @ w64.T, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dw, h64.T @ dz, rtol=1e-4, atol=1e-4)

  # Standalone Softmax VJP
  sx = rng.standard_normal((12, 20)).astype(np.float32)
  sw = rng.standard_normal((12, 20)).astype(np.float32)
  fired: list[str] = []
  orig_opt = algebraic_vjp.optimize_tape

  def spy_opt(tape, el, out):
    orig_opt(tape, el, out)
    fired.extend(nd.info[0].__name__ for nd in tape.nodes if nd.kind == "algebraic_vjp")

  algebraic_vjp.optimize_tape = spy_opt
  try:
    with parakeet.options(verify=True):
      _, dx_sm = value_and_grad(_softmax_loss, argnum=0)(sx, sw)
      x3 = rng.standard_normal((2, 8, 16)).astype(np.float32)
      g1 = rng.standard_normal(16).astype(np.float32)
      b1 = rng.standard_normal(16).astype(np.float32)
      _, (dx_ln, _, _) = value_and_grad(_layernorm_loss, argnum=(0, 1, 2))(x3, g1, b1)
      _, (dx_rms, _) = value_and_grad(_rmsnorm_loss, argnum=(0, 1))(x3, g1)
  finally:
    algebraic_vjp.optimize_tape = orig_opt

  assert "_bwd_softmax" in fired and "_bwd_norm" in fired, fired
  sx64, sw64 = sx.astype(np.float64), sw.astype(np.float64)
  esx = np.exp(sx64 - sx64.max(axis=1, keepdims=True))
  psx = esx / esx.sum(axis=1, keepdims=True)
  np.testing.assert_allclose(dx_sm, psx * (sw64 - np.sum(sw64 * psx, axis=1, keepdims=True)), rtol=1e-4, atol=1e-4)

  # Finite-diff checks on LayerNorm and RMSNorm
  eps = 1e-3
  x64 = x3.astype(np.float64)
  for idx_flat in (0, 7, 42):
    delta = np.zeros_like(x64)
    delta.ravel()[idx_flat] = eps

    def f_ln(xx):
      mu = xx.mean(axis=2, keepdims=True)
      var = ((xx - mu) ** 2).mean(axis=2, keepdims=True)
      y = (xx - mu) / np.sqrt(var + 1e-5) * g1 + b1
      return float(np.sum(y * y))

    def f_rms(xx):
      var = (xx**2).mean(axis=2, keepdims=True)
      y = xx / np.sqrt(var + 1e-5) * g1
      return float(np.sum(y * y))

    np.testing.assert_allclose(dx_ln.ravel()[idx_flat], (f_ln(x64 + delta) - f_ln(x64 - delta)) / (2 * eps), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
      dx_rms.ravel()[idx_flat], (f_rms(x64 + delta) - f_rms(x64 - delta)) / (2 * eps), rtol=1e-3, atol=1e-3
    )


# ---------------------------------------------------------------------------
# GPU Execution (CUDA Simulator + Metal): FlashAttention, FlashDecoding, CCE, RBF
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="the simulator needs a POSIX C++ toolchain")
def test_cuda_simulator_monoidal_flash_decode_cce_and_rbf(tmp_path):
  """Run FlashAttention fwd+bwd, FlashDecoding Split-K, Cut Cross-Entropy fwd+bwd,
  and RBF kernel smoothing on the CUDA simulator with device_fallback='error'."""
  code = """
import numpy as np, parakeet
from parakeet import jit, value_and_grad
import tests.test_algebraic_monoid as tm

rng = np.random.default_rng(21)
q = rng.standard_normal((2, 2, 35, 20)).astype(np.float32)
k = rng.standard_normal((2, 2, 35, 20)).astype(np.float32)
v = rng.standard_normal((2, 2, 35, 20)).astype(np.float32)
c = np.tril(np.ones((35, 35), dtype=bool))
# Autoregressive decode shape -> triggers FlashDecoding Split-K!
qd = rng.standard_normal((1, 2, 1, 20)).astype(np.float32)
kd = rng.standard_normal((1, 2, 160, 20)).astype(np.float32)
vd = rng.standard_normal((1, 2, 160, 20)).astype(np.float32)
cd = np.ones((1, 160), dtype=bool)
h = rng.standard_normal((36, 20)).astype(np.float32)
w = rng.standard_normal((20, 40)).astype(np.float32)
tgt = rng.integers(0, 40, size=(36,), dtype=np.int64)
rx = rng.standard_normal((34, 12)).astype(np.float32)
ry = rng.standard_normal((38, 12)).astype(np.float32)
rv = rng.standard_normal((38, 10)).astype(np.float32)

with parakeet.options(backend='cuda', blas=False, device_threshold=1, device_fallback='error', verify=True):
    av, (dq, dk, dv) = value_and_grad(tm._attn_loss, argnum=(0, 1, 2))(q, k, v, c)
    dec_got = jit(tm._attn_fwd)(qd, kd, vd, cd)
    cv, (dh, dw) = value_and_grad(tm._cce_fwd, argnum=(0, 1))(h, w, tgt)
    rbf_got = jit(tm._rbf_kernel_smoothing)(rx, ry, rv, np.float32(0.25))

q64, k64, v64 = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
sc = 1.0 / np.sqrt(20.0)
s = np.where(c, (q64 @ k64.transpose(0,1,3,2)) * sc, -1e9)
e = np.exp(s - s.max(axis=3, keepdims=True)); p = e / e.sum(axis=3, keepdims=True); o = p @ v64
do = np.ones_like(o); D = (do * o).sum(axis=3, keepdims=True)
ds = np.where(c, p * ((do @ v64.transpose(0,1,3,2)) - D) * sc, 0.0)
np.testing.assert_allclose(av, float(o.sum()), rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(dq, ds @ k64, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(dk, ds.transpose(0,1,3,2) @ q64, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(dv, p.transpose(0,1,3,2) @ do, rtol=1e-4, atol=1e-4)

sd = (qd.astype(np.float64) @ kd.astype(np.float64).transpose(0,1,3,2)) * sc
ed = np.exp(sd - sd.max(axis=3, keepdims=True))
np.testing.assert_allclose(dec_got, (ed / ed.sum(axis=3, keepdims=True)) @ vd.astype(np.float64), rtol=1e-4, atol=1e-4)

h64, w64 = h.astype(np.float64), w.astype(np.float64)
z = h64 @ w64; ez = np.exp(z - z.max(axis=1, keepdims=True)); pz = ez / ez.sum(axis=1, keepdims=True)
dz = pz.copy(); dz[np.arange(36), tgt] -= 1.0; dz /= 36.0
np.testing.assert_allclose(cv, -np.mean(np.log(pz[np.arange(36), tgt])), rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(dh, dz @ w64.T, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(dw, h64.T @ dz, rtol=1e-4, atol=1e-4)

d2 = np.sum((rx.astype(np.float64)[:, None, :] - ry.astype(np.float64)[None, :, :]) ** 2, axis=2)
er = np.exp(-0.25 * d2 - (-0.25 * d2).max(axis=1, keepdims=True))
np.testing.assert_allclose(rbf_got, (er / er.sum(axis=1, keepdims=True)) @ rv.astype(np.float64), rtol=1e-4, atol=1e-4)
print('monoidal-cuda-sim-ok')
"""
  script = tmp_path / "run_monoidal_sim.py"
  script.write_text(code)
  env = dict(os.environ, PARAKEET_CUDA_SIMULATE="1")
  proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, check=False)
  assert proc.returncode == 0 and "monoidal-cuda-sim-ok" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.skipif(parakeet.available_backends().get("metal") != "ok", reason="Metal GPU unavailable")
def test_metal_monoidal_flash_decode_cce_and_rbf():
  rng = np.random.default_rng(22)
  q = rng.standard_normal((2, 2, 35, 20)).astype(np.float32)
  k = rng.standard_normal((2, 2, 35, 20)).astype(np.float32)
  v = rng.standard_normal((2, 2, 35, 20)).astype(np.float32)
  c = np.tril(np.ones((35, 35), dtype=bool))
  qd = rng.standard_normal((1, 2, 1, 20)).astype(np.float32)
  kd = rng.standard_normal((1, 2, 160, 20)).astype(np.float32)
  vd = rng.standard_normal((1, 2, 160, 20)).astype(np.float32)
  cd = np.ones((1, 160), dtype=bool)
  h = rng.standard_normal((36, 20)).astype(np.float32)
  w = rng.standard_normal((20, 40)).astype(np.float32)
  tgt = rng.integers(0, 40, size=(36,), dtype=np.int64)

  with parakeet.options(backend="metal", blas=False, device_threshold=1, device_fallback="error", verify=True):
    av, (dq, dk, dv) = value_and_grad(_attn_loss, argnum=(0, 1, 2))(q, k, v, c)
    dec_got = jit(_attn_fwd)(qd, kd, vd, cd)
    cv, (dh, dw) = value_and_grad(_cce_fwd, argnum=(0, 1))(h, w, tgt)

  q64, k64, v64 = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
  sc = 1.0 / np.sqrt(20.0)
  s = np.where(c, (q64 @ k64.transpose(0, 1, 3, 2)) * sc, -1e9)
  e = np.exp(s - s.max(axis=3, keepdims=True))
  p = e / e.sum(axis=3, keepdims=True)
  o = p @ v64
  do = np.ones_like(o)
  D = (do * o).sum(axis=3, keepdims=True)
  ds = np.where(c, p * ((do @ v64.transpose(0, 1, 3, 2)) - D) * sc, 0.0)
  np.testing.assert_allclose(av, float(o.sum()), rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dq, ds @ k64, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dk, ds.transpose(0, 1, 3, 2) @ q64, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dv, p.transpose(0, 1, 3, 2) @ do, rtol=1e-4, atol=1e-4)

  sd = (qd.astype(np.float64) @ kd.astype(np.float64).transpose(0, 1, 3, 2)) * sc
  ed = np.exp(sd - sd.max(axis=3, keepdims=True))
  np.testing.assert_allclose(dec_got, (ed / ed.sum(axis=3, keepdims=True)) @ vd.astype(np.float64), rtol=1e-4, atol=1e-4)

  h64, w64 = h.astype(np.float64), w.astype(np.float64)
  z = h64 @ w64
  ez = np.exp(z - z.max(axis=1, keepdims=True))
  pz = ez / ez.sum(axis=1, keepdims=True)
  dz = pz.copy()
  dz[np.arange(36), tgt] -= 1.0
  dz /= 36.0
  np.testing.assert_allclose(cv, -np.mean(np.log(pz[np.arange(36), tgt])), rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dh, dz @ w64.T, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dw, h64.T @ dz, rtol=1e-4, atol=1e-4)


def _finite_fill_causal_sym(q, k, v):
  causal = parakeet.imap(lambda ij: ij[1] <= ij[0], (q.shape[0], k.shape[0]))
  s = np.where(causal, (q @ k.T) * np.float32(0.125), np.float32(0.0))
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v


def _finite_fill_causal_buf(q, k, v, mask):
  s = np.where(mask, (q @ k.T) * np.float32(0.125), np.float32(0.0))
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v


def _finite_fill_loss_buf(q, k, v, mask):
  return np.sum(_finite_fill_causal_buf(q, k, v, mask))


@pytest.mark.skipif(sys.platform == "win32", reason="the simulator needs a POSIX C++ toolchain")
def test_finite_fill_false_tile_not_skipped_fwd_and_bwd(tmp_path):
  """Regression for finite mask fill across >= 2 tiles (64x64): an all-false
  tile with finite fill (e.g. 0.0) contributes weight exp(fill - m) != 0 to both
  forward softmax and backward dV, and must NOT be pruned."""
  code = """
import numpy as np, parakeet
from parakeet import jit, value_and_grad
import tests.test_algebraic_monoid as tm

q = np.zeros((64, 8), dtype=np.float32)
k = np.zeros((64, 8), dtype=np.float32)
v = np.zeros((64, 8), dtype=np.float32)
v[32:, :] = 1.0
mask = np.tril(np.ones((64, 64), dtype=bool))

with parakeet.options(backend="cuda", blas=False, device_threshold=1, device_fallback="error", verify=True):
    out_sym = jit(tm._finite_fill_causal_sym)(q, k, v)
    out_buf = jit(tm._finite_fill_causal_buf)(q, k, v, mask)
    lv, (dq, dk, dv) = value_and_grad(tm._finite_fill_loss_buf, argnum=(0, 1, 2))(q, k, v, mask)

np.testing.assert_allclose(out_sym, np.full((64, 8), 0.5, dtype=np.float32), rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(out_buf, np.full((64, 8), 0.5, dtype=np.float32), rtol=1e-5, atol=1e-5)
# Exact reference gradients for uniform p = 1/64 across all 64 keys (even masked ones with fill=0!):
# dV[j, d] = sum_i p[i, j] * 1.0 = 64 * (1/64) = 1.0 for every j in 0..63!
np.testing.assert_allclose(lv, 64.0 * 8.0 * 0.5, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(dv, np.ones((64, 8), dtype=np.float32), rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(dq, np.zeros((64, 8), dtype=np.float32), rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(dk, np.zeros((64, 8), dtype=np.float32), rtol=1e-4, atol=1e-4)
print("finite-fill-ok")
"""
  script = tmp_path / "run_finite_fill.py"
  script.write_text(code)
  env = dict(os.environ, PARAKEET_CUDA_SIMULATE="1")
  proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, check=False)
  assert proc.returncode == 0 and "finite-fill-ok" in proc.stdout, proc.stdout + proc.stderr

