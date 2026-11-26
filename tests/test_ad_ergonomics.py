"""Tests for v1.10 AD ergonomics: @par.jit(jvp=..., vjp=...), @par.jvp_for/@par.vjp_for,
parameter-name differentiation (wrt=), and differentiable compiled np.linalg.*."""

from __future__ import annotations

import numpy as np
import pytest

import parakeet as par


def _fd_grad(fn, x, eps=1e-6):
  x = np.asarray(x, dtype=np.float64)
  g = np.zeros_like(x)
  flat_x = x.reshape(-1)
  flat_g = g.reshape(-1)
  for i in range(flat_x.size):
    xp = flat_x.copy()
    xm = flat_x.copy()
    xp[i] += eps
    xm[i] -= eps
    flat_g[i] = (float(fn(xp.reshape(x.shape))) - float(fn(xm.reshape(x.shape)))) / (2.0 * eps)
  return g


def _fd_jvp(fn, x, dx, eps=1e-6):
  return (np.asarray(fn(x + eps * dx)) - np.asarray(fn(x - eps * dx))) / (2.0 * eps)


# ---------------------------------------------------------------------------
# 1. @par.jit(jvp=..., vjp=...) and @par.jvp_for / @par.vjp_for
# ---------------------------------------------------------------------------


def test_jit_jvp_flat_args_single_and_multi():
  def smooth_relu_jvp(x, dx):
    return np.maximum(x, 0.0), dx / (1.0 + np.exp(-x))

  @par.jit(jvp=smooth_relu_jvp)
  def smooth_relu(x):
    return np.maximum(x, 0.0)

  x = np.array([-1.5, 0.0, 2.0])
  dx = np.array([1.0, 2.0, -0.5])
  y, dy = par.jvp(smooth_relu, (x,), (dx,))
  np.testing.assert_allclose(y, np.maximum(x, 0.0))
  np.testing.assert_allclose(dy, dx / (1.0 + np.exp(-x)), rtol=1e-12)

  # 2-argument flat JVP: (a, b, da, db)
  def weird_mul_jvp(a, b, da, db):
    return a * b, 2.0 * b * da + 3.0 * a * db

  @par.jit(jvp=weird_mul_jvp)
  def weird_mul(a, b):
    return a * b

  a = np.array([1.0, 2.0, 3.0])
  b = np.array([4.0, 5.0, 6.0])
  da = np.array([0.5, -1.0, 2.0])
  db = np.array([1.0, 0.25, -0.5])
  out, dout = par.jvp(weird_mul, (a, b), (da, db))
  np.testing.assert_allclose(out, a * b)
  np.testing.assert_allclose(dout, 2.0 * b * da + 3.0 * a * db, rtol=1e-12)


def test_jit_vjp_closure_single_and_multi():
  def stable_softplus_vjp(x):
    y = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def pullback(dy):
      return dy / (1.0 + np.exp(-x))

    return y, pullback

  @par.jit(vjp=stable_softplus_vjp)
  def stable_softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

  x = np.array([-50.0, -1.0, 0.0, 2.0, 50.0])
  val, g = par.value_and_grad(lambda v: np.sum(stable_softplus(v)))(x)
  np.testing.assert_allclose(val, np.sum(np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)), rtol=1e-12)
  np.testing.assert_allclose(g, 1.0 / (1.0 + np.exp(-x)), rtol=1e-12)
  assert np.all(np.isfinite(g))

  # 2-argument closure VJP returning a tuple of cotangents
  def custom_bilinear_vjp(a, b):
    y = a * b

    def pullback(dy):
      return 2.0 * dy * b, 3.0 * dy * a

    return y, pullback

  @par.jit(vjp=custom_bilinear_vjp)
  def custom_bilinear(a, b):
    return a * b

  a = np.array([1.0, 2.0, 3.0])
  b = np.array([4.0, 5.0, 6.0])
  ga, gb = par.grad(lambda u, v: np.sum(custom_bilinear(u, v)), (0, 1))(a, b)
  np.testing.assert_allclose(ga, 2.0 * b, rtol=1e-12)
  np.testing.assert_allclose(gb, 3.0 * a, rtol=1e-12)


def test_jit_vjp_tuple_and_combined_jvp_vjp():
  def fwd(x):
    return x * x, (x,)

  def bwd(res, g):
    (x,) = res
    return 10.0 * x * g

  @par.jit(vjp=(fwd, bwd), jvp=lambda x, dx: (x * x, 10.0 * x * dx))
  def scaled_sq(x):
    return x * x

  x = np.array([1.0, -2.0, 3.0])
  dx = np.array([0.5, 1.0, -1.0])
  np.testing.assert_allclose(scaled_sq(x), x * x)
  np.testing.assert_allclose(par.grad(lambda v: np.sum(scaled_sq(v)))(x), 10.0 * x)
  _, dy = par.jvp(scaled_sq, (x,), (dx,))
  np.testing.assert_allclose(dy, 10.0 * x * dx)


def test_jvp_for_and_vjp_for_decorators():
  def step_clip(x):
    return np.clip(x, -1.0, 1.0)

  @par.vjp_for(step_clip)
  def _step_clip_vjp(x):
    # Straight-through estimator
    return np.clip(x, -1.0, 1.0), lambda g: g

  @par.jvp_for(step_clip)
  def _step_clip_jvp(x, dx):
    return np.clip(x, -1.0, 1.0), dx

  x = np.array([-3.0, 0.2, 4.0])
  dx = np.array([1.0, -2.0, 0.5])
  g = par.grad(lambda v: np.sum(2.0 * step_clip(v)))(x)
  np.testing.assert_allclose(g, np.full_like(x, 2.0))
  y, dy = par.jvp(step_clip, (x,), (dx,))
  np.testing.assert_allclose(y, np.clip(x, -1.0, 1.0))
  np.testing.assert_allclose(dy, dx)

  # Works on @par.jit-wrapped functions too
  @par.jit
  def my_act(x):
    return np.maximum(x, 0.0)

  @par.vjp_for(my_act)
  def _my_act_vjp(x):
    return np.maximum(x, 0.0), lambda g: 0.5 * g

  np.testing.assert_allclose(par.grad(lambda v: np.sum(my_act(v)))(x), np.full_like(x, 0.5))


# ---------------------------------------------------------------------------
# 2. Parameter-name differentiation (wrt=) and keyword-argument calls
# ---------------------------------------------------------------------------


def test_grad_by_parameter_name_and_kwargs():
  def mse_loss(x, y, w, b):
    pred = x @ w + b
    diff = pred - y
    return np.mean(diff * diff)

  rng = np.random.default_rng(7)
  x = rng.normal(size=(16, 4))
  y = rng.normal(size=(16,))
  w = rng.normal(size=(4,))
  b = 0.3

  expected_gw, expected_gb = par.grad(mse_loss, (2, 3))(x, y, w, b)
  expected_val = float(par.jit(mse_loss)(x, y, w, b))

  # Positional str spec
  gw1 = par.grad(mse_loss, "w")(x, y, w, b)
  np.testing.assert_allclose(gw1, expected_gw, rtol=1e-12)

  # Keyword wrt= spec (single & tuple)
  gw2 = par.grad(mse_loss, wrt="w")(x, y, w, b)
  np.testing.assert_allclose(gw2, expected_gw, rtol=1e-12)

  gw3, gb3 = par.grad(mse_loss, wrt=("w", "b"))(x, y, w, b)
  np.testing.assert_allclose(gw3, expected_gw, rtol=1e-12)
  np.testing.assert_allclose(gb3, expected_gb, rtol=1e-12)

  # value_and_grad with wrt= and keyword call arguments
  val, (gw4, gb4) = par.value_and_grad(mse_loss, wrt=("w", "b"))(x=x, y=y, w=w, b=b)
  np.testing.assert_allclose(val, expected_val, rtol=1e-12)
  np.testing.assert_allclose(gw4, expected_gw, rtol=1e-12)
  np.testing.assert_allclose(gb4, expected_gb, rtol=1e-12)

  # Mixed positional + keyword arguments
  gw5 = par.grad(mse_loss, wrt="w")(x, y, b=b, w=w)
  np.testing.assert_allclose(gw5, expected_gw, rtol=1e-12)

  # Inside @par.jit
  @par.jit
  def train_step(x, y, w, b, lr):
    loss, (gw, gb) = par.value_and_grad(mse_loss, wrt=("w", "b"))(x, y, w, b)
    return loss, w - lr * gw, b - lr * gb

  loss_s, w_next, b_next = train_step(x, y, w, b, 0.1)
  np.testing.assert_allclose(loss_s, expected_val, rtol=1e-12)
  np.testing.assert_allclose(w_next, w - 0.1 * expected_gw, rtol=1e-12)
  np.testing.assert_allclose(b_next, b - 0.1 * expected_gb, rtol=1e-12)

  # Error handling: unknown parameter name & duplicate target
  with pytest.raises(par.CompileError, match="unknown parameter name"):
    par.grad(mse_loss, wrt="nope")(x, y, w, b)
  with pytest.raises(par.CompileError, match="distinct"):
    par.grad(mse_loss, wrt=(2, "w"))(x, y, w, b)


# ---------------------------------------------------------------------------
# 3. Differentiable compiled np.linalg.* (grad + jvp) & Kabsch RMSD
# ---------------------------------------------------------------------------


def test_linalg_det_inv_solve_cholesky_grad_and_jvp():
  rng = np.random.default_rng(123)
  A = rng.normal(size=(4, 4)) + 3.0 * np.eye(4)
  dA = rng.normal(size=(4, 4))
  W = rng.normal(size=(4, 4))
  b1 = rng.normal(size=(4,))
  db1 = rng.normal(size=(4,))
  b2 = rng.normal(size=(4, 3))
  db2 = rng.normal(size=(4, 3))
  W2 = rng.normal(size=(4, 3))

  # det: grad & jvp
  f_det = lambda X: np.linalg.det(X)
  np.testing.assert_allclose(par.grad(f_det)(A), _fd_grad(f_det, A), rtol=1e-5, atol=1e-6)
  y, dy = par.jvp(f_det, (A,), (dA,))
  np.testing.assert_allclose(y, np.linalg.det(A), rtol=1e-12)
  np.testing.assert_allclose(dy, _fd_jvp(f_det, A, dA), rtol=1e-5, atol=1e-6)

  # inv: grad & jvp
  f_inv = lambda X: np.sum(W * np.linalg.inv(X))
  np.testing.assert_allclose(par.grad(f_inv)(A), _fd_grad(f_inv, A), rtol=1e-5, atol=1e-6)
  yi, dyi = par.jvp(lambda X: np.linalg.inv(X), (A,), (dA,))
  np.testing.assert_allclose(yi, np.linalg.inv(A), rtol=1e-11)
  np.testing.assert_allclose(dyi, _fd_jvp(np.linalg.inv, A, dA), rtol=1e-5, atol=1e-6)

  # solve (1D RHS): grad & jvp
  w1 = rng.normal(size=(4,))
  f_solve1 = lambda X, y_rhs: np.sum(w1 * np.linalg.solve(X, y_rhs))
  gA1, gb1 = par.grad(f_solve1, (0, 1))(A, b1)
  np.testing.assert_allclose(gA1, _fd_grad(lambda X: f_solve1(X, b1), A), rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(gb1, _fd_grad(lambda y_rhs: f_solve1(A, y_rhs), b1), rtol=1e-5, atol=1e-6)
  xs1, dxs1 = par.jvp(lambda X, y_rhs: np.linalg.solve(X, y_rhs), (A, b1), (dA, db1))
  np.testing.assert_allclose(xs1, np.linalg.solve(A, b1), rtol=1e-11)
  fd_xs1 = (np.linalg.solve(A + 1e-6 * dA, b1 + 1e-6 * db1) - np.linalg.solve(A - 1e-6 * dA, b1 - 1e-6 * db1)) / 2e-6
  np.testing.assert_allclose(dxs1, fd_xs1, rtol=1e-5, atol=1e-6)

  # solve (2D RHS): grad & jvp
  f_solve2 = lambda X, B: np.sum(W2 * np.linalg.solve(X, B))
  gA2, gb2 = par.grad(f_solve2, (0, 1))(A, b2)
  np.testing.assert_allclose(gA2, _fd_grad(lambda X: f_solve2(X, b2), A), rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(gb2, _fd_grad(lambda B: f_solve2(A, B), b2), rtol=1e-5, atol=1e-6)
  xs2, dxs2 = par.jvp(lambda X, B: np.linalg.solve(X, B), (A, b2), (dA, db2))
  fd_xs2 = (np.linalg.solve(A + 1e-6 * dA, b2 + 1e-6 * db2) - np.linalg.solve(A - 1e-6 * dA, b2 - 1e-6 * db2)) / 2e-6
  np.testing.assert_allclose(dxs2, fd_xs2, rtol=1e-5, atol=1e-6)

  # cholesky on SPD matrix A = M @ M.T + 2 I: grad & jvp
  M = rng.normal(size=(4, 4))
  dM = rng.normal(size=(4, 4))
  f_chol = lambda X: np.sum(W * np.linalg.cholesky(X @ X.T + 2.0 * np.eye(4)))
  np.testing.assert_allclose(par.grad(f_chol)(M), _fd_grad(f_chol, M), rtol=1e-5, atol=1e-6)
  yc, dyc = par.jvp(lambda X: np.linalg.cholesky(X @ X.T + 2.0 * np.eye(4)), (M,), (dM,))
  fd_yc = _fd_jvp(lambda X: np.linalg.cholesky(X @ X.T + 2.0 * np.eye(4)), M, dM)
  np.testing.assert_allclose(yc, np.linalg.cholesky(M @ M.T + 2.0 * np.eye(4)), rtol=1e-11)
  np.testing.assert_allclose(dyc, fd_yc, rtol=1e-5, atol=1e-6)


def test_linalg_eigh_eigvalsh_svd_grad():
  rng = np.random.default_rng(42)
  M = rng.normal(size=(4, 4))
  W_val = np.array([1.0, -2.0, 0.5, 3.0])

  # eigvalsh
  f_eigvalsh = lambda X: np.sum(W_val * np.linalg.eigvalsh(0.5 * (X + X.T)))
  np.testing.assert_allclose(par.grad(f_eigvalsh)(M), _fd_grad(f_eigvalsh, M), rtol=1e-5, atol=1e-6)

  # eigh (sign-invariant quadratic function of eigenvectors + eigenvalues)
  Q = rng.normal(size=(4, 4))

  def f_eigh(X):
    w, V = np.linalg.eigh(0.5 * (X + X.T))
    # V * V is invariant to column sign flips of eigenvectors
    return np.sum(W_val * w) + np.sum(Q * (V * V))

  np.testing.assert_allclose(par.grad(f_eigh)(M), _fd_grad(f_eigh, M), rtol=1e-5, atol=1e-6)

  # SVD: square (3x3), tall (5x3), and wide (3x5)
  for shape in [(3, 3), (5, 3), (3, 5)]:
    A = rng.normal(size=shape)
    k = min(shape)
    Ws = rng.normal(size=(k,))
    Wuv = rng.normal(size=shape)

    def f_svd(X):
      U, S, Vt = np.linalg.svd(X, full_matrices=False)
      # Sign-invariant in (U[:, i], Vt[i, :]) pairs: sum_i (U[:, i] @ Vt[i, :]) weighted by S
      recon = (U * S[None, :]) @ Vt
      return np.sum(Ws * S) + np.sum(Wuv * (recon * recon))

    np.testing.assert_allclose(par.grad(f_svd)(A), _fd_grad(f_svd, A), rtol=1e-5, atol=1e-5)


def test_gp_log_marginal_likelihood_and_differentiable_kabsch():
  # 1. Gaussian process log-marginal likelihood through cholesky + solve
  rng = np.random.default_rng(99)
  x = np.linspace(0.0, 4.0, 12)
  y = np.sin(x) + 0.1 * rng.normal(size=x.shape)
  d2 = (x[:, None] - x[None, :]) ** 2

  def gp_nll(log_amp, log_ls, log_noise):
    amp = np.exp(log_amp)
    ls = np.exp(log_ls)
    noise = np.exp(log_noise)
    K = amp * np.exp(-0.5 * d2 / (ls * ls)) + noise * np.eye(len(x))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return 0.5 * (np.dot(y, alpha) + logdet)

  params = (0.2, -0.3, -2.0)
  g_par = par.grad(gp_nll, wrt=("log_amp", "log_ls", "log_noise"))(*params)
  eps = 1e-6
  for i in range(3):
    p_plus = list(params)
    p_minus = list(params)
    p_plus[i] += eps
    p_minus[i] -= eps
    fd = (gp_nll(*p_plus) - gp_nll(*p_minus)) / (2.0 * eps)
    np.testing.assert_allclose(g_par[i], fd, rtol=1e-5, atol=1e-6)

  # 2. Differentiable Kabsch 3D protein RMSD wrt coordinates P
  import importlib.util
  import pathlib

  spec = importlib.util.spec_from_file_location(
    "bio_algorithms", pathlib.Path(__file__).resolve().parents[1] / "examples" / "bio_algorithms.py"
  )
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)

  Q = rng.normal(size=(15, 3))
  # Non-reflected and reflected initial conformations
  for flip in (1.0, -1.0):
    P = Q.copy()
    P[:, 2] *= flip
    P = P + 0.25 * rng.normal(size=(15, 3))

    def rmsd_only(coords):
      rmsd, _, _ = mod.kabsch_rmsd(coords, Q)
      return rmsd

    gP = par.grad(rmsd_only)(P)
    fd_gP = _fd_grad(lambda c: float(mod.kabsch_rmsd(c, Q)[0]), P)
    np.testing.assert_allclose(gP, fd_gP, rtol=1e-4, atol=1e-5)
