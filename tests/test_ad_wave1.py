"""Tests for AD Wave 1 (#75 + #77): stop_gradient, custom_vjp, checkpoint,
IR-level checkpointed reverse for/while loops, checkpointed fold, and scan AD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pytest

import parakeet as par


def _fd_grad(fn, *args, argnum=0, eps=1e-5):
  args = list(args)
  x = args[argnum]
  if np.isscalar(x):
    a1, a2 = list(args), list(args)
    a1[argnum] = x + eps
    a2[argnum] = x - eps
    return (fn(*a1) - fn(*a2)) / (2 * eps)
  x = np.asarray(x, dtype=np.float64)
  g = np.zeros_like(x)
  if x.size == 0:
    return g
  it = np.nditer(x, flags=["multi_index"])
  while not it.finished:
    idx = it.multi_index
    xp, xm = x.copy(), x.copy()
    xp[idx] += eps
    xm[idx] -= eps
    a1, a2 = list(args), list(args)
    a1[argnum], a2[argnum] = xp, xm
    g[idx] = (fn(*a1) - fn(*a2)) / (2 * eps)
    it.iternext()
  return g


# ---------------------------------------------------------------------------
# 1. stop_gradient
# ---------------------------------------------------------------------------


class _State(NamedTuple):
  a: np.float64
  b: np.float64


@dataclass(frozen=True)
class _Box:
  w: np.ndarray
  tag: int


def test_stop_gradient_scalar_array_and_ste():
  def loss_scalar(x):
    return x * par.stop_gradient(x * x)

  # d/dx [x * const(x^2)] = x^2 (not 3*x^2)
  assert np.isclose(par.grad(loss_scalar)(3.0), 9.0)

  def ste_loss(x):
    # Straight-through estimator around rounding: forward uses round(x), backward is identity!
    y = x + par.stop_gradient(np.round(x) - x)
    return np.sum(y * y)

  x = np.array([1.2, 2.7, -0.4], dtype=np.float64)
  val, g = par.value_and_grad(ste_loss)(x)
  np.testing.assert_allclose(val, np.sum(np.round(x) ** 2))
  # d/dx sum(y^2) with dy/dx = 1 is 2 * y = 2 * round(x)
  np.testing.assert_allclose(g, 2.0 * np.round(x), rtol=1e-6)

  # Outside grad, stop_gradient is a zero-copy identity
  @par.jit
  def plain(arr):
    return par.stop_gradient(arr) + 1.0

  np.testing.assert_allclose(plain(x), x + 1.0)


def test_stop_gradient_records_and_ragged():
  def loss_rec(x):
    st = par.stop_gradient(_State(x, x * 2.0))
    return x * st.a + x * st.b

  assert np.isclose(par.grad(loss_rec)(2.5), 2.5 + 5.0)

  def loss_dc(box):
    sb = par.stop_gradient(box)
    return np.sum(box.w * sb.w)

  b = _Box(w=np.array([1.0, 2.0, 3.0]), tag=7)
  gb = par.grad(loss_dc)(b)
  np.testing.assert_allclose(gb.w, b.w)

  def loss_ragged(r):
    sr = par.stop_gradient(r)
    return np.sum(r.values * sr.values)

  r = par.RaggedArray.from_lengths(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64), np.array([2, 3]))
  gr = par.grad(loss_ragged)(r)
  np.testing.assert_allclose(gr.values, r.values)
  np.testing.assert_array_equal(gr.offsets, r.offsets)


# ---------------------------------------------------------------------------
# 2. @par.custom_vjp
# ---------------------------------------------------------------------------


def test_custom_vjp_single_multi_and_higher_order():
  @par.custom_vjp
  def scaled_square(x, c):
    return c * x * x

  def ss_fwd(x, c):
    return c * x * x, (x, c)

  def ss_bwd(res, g):
    x, c = res
    # Custom rule intentionally scales x's grad by 3x so we verify the custom rule fired!
    return g * 3.0 * c * x, np.sum(g * x * x)

  scaled_square.defvjp(ss_fwd, ss_bwd)

  def f(x, c):
    return np.sum(scaled_square(x, c))

  x = np.array([1.0, 2.0, -3.0], dtype=np.float64)
  c = 2.5
  gx, gc = par.grad(f, (0, 1))(x, c)
  np.testing.assert_allclose(gx, 3.0 * c * x)
  assert np.isclose(gc, np.sum(x * x))

  # Higher-order AD through custom_vjp!
  @par.custom_vjp
  def smooth_cube(t):
    return t * t * t

  smooth_cube.defvjp(lambda t: (t * t * t, t), lambda t, g: g * 3.0 * t * t)

  d1 = par.grad(smooth_cube)
  d2 = par.grad(lambda t: d1(t))
  assert np.isclose(d1(2.0), 12.0)
  assert np.isclose(d2(2.0), 12.0)


def test_custom_vjp_implicit_while_fixed_point():
  """Implicit Function Theorem differentiation through a while-loop fixed-point solver:
  solve y = 0.5 * cos(a * y) + b, storing ZERO iteration history."""

  @par.custom_vjp
  def fixed_point(a, b):
    y = b
    i = 0
    while i < 60:
      y = 0.5 * np.cos(a * y) + b
      i = i + 1
    return y

  def fp_fwd(a, b):
    y = fixed_point(a, b)
    return y, (a, y)

  def fp_bwd(res, g):
    a, y = res
    # F(y, a, b) = y - 0.5 * cos(a * y) - b = 0
    # dF/dy = 1 + 0.5 * a * sin(a * y)
    # dy/da = (-0.5 * y * sin(a * y)) / dF/dy
    # dy/db = 1 / dF/dy
    df_dy = 1.0 + 0.5 * a * np.sin(a * y)
    v = g / df_dy
    return v * (-0.5 * y * np.sin(a * y)), v

  fixed_point.defvjp(fp_fwd, fp_bwd)

  def objective(a, b):
    y = fixed_point(a, b)
    return y * y

  a, b = 0.8, 0.3
  ga, gb = par.grad(objective, (0, 1))(a, b)
  assert np.isclose(ga, _fd_grad(objective, a, b, argnum=0), rtol=1e-4)
  assert np.isclose(gb, _fd_grad(objective, a, b, argnum=1), rtol=1e-4)


def test_custom_vjp_namedtuple_multi_output_and_map():
  class _Pair(NamedTuple):
    u: np.ndarray
    v: np.ndarray

  @par.custom_vjp
  def split_op(x):
    return _Pair(x * 2.0, x * x)

  split_op.defvjp(
    lambda x: (_Pair(x * 2.0, x * x), x),
    lambda x, g: g.u * 2.0 + g.v * (2.0 * x),
  )

  def loss(x):
    p = split_op(x)
    return np.sum(p.u * 3.0 + p.v)

  x = np.array([1.0, -2.0, 4.0], dtype=np.float64)
  np.testing.assert_allclose(par.grad(loss)(x), 6.0 + 2.0 * x)

  # Inside par.map
  def mapped_loss(mat):
    return np.sum(par.map(lambda row: split_op(row).v, mat))

  mat = np.array([[1.0, 2.0], [-3.0, 4.0]], dtype=np.float64)
  np.testing.assert_allclose(par.grad(mapped_loss)(mat), 2.0 * mat)


# ---------------------------------------------------------------------------
# 3. par.checkpoint
# ---------------------------------------------------------------------------


def test_checkpoint_mlp_and_attention():
  @par.checkpoint
  def block(h, w):
    return h + np.tanh(h @ w) * 0.25

  def loss_ckpt(x, w1, w2):
    h = block(x, w1)
    h = block(h, w2)
    return np.sum(h * h)

  def loss_plain(x, w1, w2):
    h = x + np.tanh(x @ w1) * 0.25
    h = h + np.tanh(h @ w2) * 0.25
    return np.sum(h * h)

  rng = np.random.default_rng(0)
  x = rng.normal(size=(8, 4))
  w1 = rng.normal(size=(4, 4)) * 0.3
  w2 = rng.normal(size=(4, 4)) * 0.3
  g_ckpt = par.grad(loss_ckpt, (0, 1, 2))(x, w1, w2)
  g_plain = par.grad(loss_plain, (0, 1, 2))(x, w1, w2)
  for gc, gp in zip(g_ckpt, g_plain):
    np.testing.assert_allclose(gc, gp, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# 4. IR-level checkpointed reverse `for` loops (#75)
# ---------------------------------------------------------------------------


def test_runtime_for_loop_scalar_and_iir():
  def iir_loss(a, b, s0):
    s = s0
    for t in range(a.shape[0]):
      s = a[t] * s + b[t]
    return s * s

  rng = np.random.default_rng(42)
  for T in (0, 1, 7, 64, 250):
    a = rng.uniform(0.5, 0.95, size=(T,))
    b = rng.normal(size=(T,))
    s0 = 1.25
    ga, gb, gs0 = par.grad(iir_loss, (0, 1, 2))(a, b, s0)
    if T <= 64:
      np.testing.assert_allclose(ga, _fd_grad(iir_loss, a, b, s0, argnum=0), rtol=1e-4, atol=1e-6)
      np.testing.assert_allclose(gb, _fd_grad(iir_loss, a, b, s0, argnum=1), rtol=1e-4, atol=1e-6)
      assert np.isclose(gs0, _fd_grad(iir_loss, a, b, s0, argnum=2), rtol=1e-4, atol=1e-6)
    else:
      # Exact analytical check for T = 250
      # s_T = s0 * prod(a) + sum_k b_k * prod_{j>k} a_j
      s = s0
      for t in range(T):
        s = a[t] * s + b[t]
      # Reverse analytical
      ds = 2.0 * s
      exp_gb = np.empty(T)
      for t in range(T - 1, -1, -1):
        exp_gb[t] = ds
        ds = ds * a[t]
      np.testing.assert_allclose(gb, exp_gb, rtol=1e-5, atol=1e-6)
      assert np.isclose(gs0, ds, rtol=1e-5, atol=1e-6)


def test_runtime_for_loop_over_array_and_array_state():
  def sum_weighted(w, x):
    s = 0.0
    for v in x.ravel():
      s = s + np.sum(w * v)
    return s

  w = np.array([[1.0, 2.0], [3.0, 4.0]])
  x = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]])
  gw, gx = par.grad(sum_weighted, (0, 1))(w, x)
  np.testing.assert_allclose(gw, np.full_like(w, np.sum(x)))
  np.testing.assert_allclose(gx, np.full_like(x, np.sum(w)))

  # Array-valued loop-carried state (RNN / ODE rollout with runtime horizon > UNROLL_THRESHOLD)
  def rollout(h0, W, n_steps):
    h = h0
    for _ in range(n_steps):
      h = h + 0.05 * np.tanh(h @ W)
    return np.sum(h * h)

  rng = np.random.default_rng(7)
  h0 = rng.normal(size=(4, 3))
  W = rng.normal(size=(3, 3)) * 0.4
  gh0, gW = par.grad(rollout, (0, 1))(h0, W, 45)
  np.testing.assert_allclose(gh0, _fd_grad(rollout, h0, W, 45, argnum=0), rtol=1e-4, atol=1e-6)
  np.testing.assert_allclose(gW, _fd_grad(rollout, h0, W, 45, argnum=1), rtol=1e-4, atol=1e-6)


def test_long_static_for_loop_beyond_max_unroll():
  """Previously crashed at MAX_UNROLL = 4096; now compiles cleanly with O(sqrt(N)) checkpoints."""

  def long_decay(w):
    s = np.sum(w)
    for _ in range(10000):
      s = s * 0.9999 + 0.0001
    return s

  w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
  g = par.grad(long_decay)(w)
  np.testing.assert_allclose(g, np.full_like(w, 0.9999**10000), rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. IR-level checkpointed reverse `while` loops (#75)
# ---------------------------------------------------------------------------


def test_while_loop_short_zero_and_stride_doubling():
  def scale_while(w, limit):
    s = np.sum(w)
    i = 0
    while i < limit:
      s = s * 1.01 + 0.5
      i = i + 1
    return s

  w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
  for limit in (0, 3, 25, 160):  # 160 > WHILE_CHECKPOINT_SLOTS=64 -> exercises stride doubling!
    g = par.grad(scale_while)(w, limit)
    np.testing.assert_allclose(g, np.full_like(w, 1.01**limit), rtol=1e-5)


def test_while_loop_data_dependent_newton_sqrt():
  def newton_sqrt(x):
    y = x * 0.5 + 0.5
    i = 0
    while np.abs(y * y - x) > 1e-10 and i < 100:
      y = 0.5 * (y + x / y)
      i = i + 1
    return y

  for x_val in (2.0, 9.0, 25.0):
    val, g = par.value_and_grad(newton_sqrt)(x_val)
    assert np.isclose(val, np.sqrt(x_val), rtol=1e-8)
    assert np.isclose(g, 0.5 / np.sqrt(x_val), rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. Checkpointed fold & general scan AD (#75)
# ---------------------------------------------------------------------------


def test_checkpointed_fold_and_general_scan_ad():
  def fold_loss(x, init):
    return par.reduce(lambda acc, xi: 0.85 * acc + xi * xi, x, init=init)

  rng = np.random.default_rng(11)
  x = rng.normal(size=(120,)) * 0.3
  gx, gi = par.grad(fold_loss, (0, 1))(x, 0.4)
  # Analytical: out = init * 0.85^N + sum_k x_k^2 * 0.85^(N-1-k)
  n = len(x)
  powers = 0.85 ** np.arange(n - 1, -1, -1)
  np.testing.assert_allclose(gx, 2.0 * x * powers, rtol=1e-5, atol=1e-7)
  assert np.isclose(gi, 0.85**n, rtol=1e-5)

  def scan_loss(x, init):
    y = par.scan(lambda acc, xi: 0.75 * acc + xi, x, init=init)
    return np.sum(y * y)

  x_s = rng.normal(size=(30,)) * 0.5
  gx_s, gi_s = par.grad(scan_loss, (0, 1))(x_s, 0.2)
  np.testing.assert_allclose(gx_s, _fd_grad(scan_loss, x_s, 0.2, argnum=0), rtol=1e-4, atol=1e-6)
  assert np.isclose(gi_s, _fd_grad(scan_loss, x_s, 0.2, argnum=1), rtol=1e-4, atol=1e-6)

  # Scan without init (e.g. cumprod)
  def cumprod_loss(x):
    return np.sum(par.scan(par.multiply, x))

  x_p = np.array([1.2, 0.8, 1.5, 0.9, 1.1])
  np.testing.assert_allclose(par.grad(cumprod_loss)(x_p), _fd_grad(cumprod_loss, x_p), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("backend", ["c", "multicore"])
def test_ad_wave1_backends(backend):
  def f(a, b):
    s = 0.0
    for t in range(a.shape[0]):
      s = a[t] * s + b[t]
    return s * s

  a = np.full(50, 0.9, dtype=np.float64)
  b = np.linspace(0.1, 1.0, 50, dtype=np.float64)
  with par.options(backend=backend):
    ga, gb = par.grad(f, (0, 1))(a, b)
  np.testing.assert_allclose(ga, _fd_grad(f, a, b, argnum=0), rtol=1e-4, atol=1e-6)
  np.testing.assert_allclose(gb, _fd_grad(f, a, b, argnum=1), rtol=1e-4, atol=1e-6)


def test_runtime_multilayer_stack_and_show_rewrite():
  """Differentiate through a runtime-length stack of layers indexing a 3D weight tensor
  and verify par.show_rewrite reports Checkpointed Reverse Loop."""

  def stack_loss(h0, weights):
    h = h0
    for t in range(weights.shape[0]):
      h = h + 0.1 * np.tanh(h @ weights[t])
    return np.sum(h * h)

  rng = np.random.default_rng(19)
  h0 = rng.normal(size=(4, 3))
  weights = rng.normal(size=(5, 3, 3)) * 0.3
  gh0, gw = par.grad(stack_loss, (0, 1))(h0, weights)
  np.testing.assert_allclose(gh0, _fd_grad(stack_loss, h0, weights, argnum=0), rtol=1e-4, atol=1e-6)
  np.testing.assert_allclose(gw, _fd_grad(stack_loss, h0, weights, argnum=1), rtol=1e-4, atol=1e-6)

  rep = par.show_rewrite(par.grad(stack_loss, (0, 1)), h0, weights)
  assert any("Checkpointed Reverse Loop" in law for law in rep.laws_triggered)


def test_law2_flash_attention_inside_checkpoint():
  """Verify Law 2 FlashAttention rematerialization composes cleanly inside par.checkpoint."""

  @par.checkpoint
  def attn_block(q, k, v):
    scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))
    s = (q @ k.T) * scale
    e = np.exp(s - np.max(s, axis=-1, keepdims=True))
    return (e / np.sum(e, axis=-1, keepdims=True)) @ v

  def loss(q, k, v):
    o = attn_block(q, k, v)
    return np.sum(o * o)

  rng = np.random.default_rng(23)
  q = rng.normal(size=(16, 8)).astype(np.float32)
  k = rng.normal(size=(16, 8)).astype(np.float32)
  v = rng.normal(size=(16, 8)).astype(np.float32)
  gq, gk, gv = par.grad(loss, (0, 1, 2))(q, k, v)

  def ref_loss(q_, k_, v_):
    scale = np.float32(1.0 / np.sqrt(np.float32(q_.shape[-1])))
    s = (q_ @ k_.T) * scale
    e = np.exp(s - np.max(s, axis=-1, keepdims=True))
    o = (e / np.sum(e, axis=-1, keepdims=True)) @ v_
    return np.sum(o * o)

  rq, rk, rv = par.grad(ref_loss, (0, 1, 2))(q, k, v)
  np.testing.assert_allclose(gq, rq, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(gk, rk, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(gv, rv, rtol=1e-5, atol=1e-5)
