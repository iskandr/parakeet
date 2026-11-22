"""grad: gradients are checked against central differences, and everything
that cannot be differentiated has to say so instead of returning a number."""

import numpy as np
import pytest

import parakeet
from parakeet import allpairs, each, grad, jit, value_and_grad

RNG = np.random.default_rng(0)
X = RNG.random((6, 4)) + 0.5
Y = RNG.random((6, 4)) + 0.5
W = RNG.random((4, 3))
V = RNG.random(4) + 0.5
T = RNG.random((6, 3))


def numeric(f, args, k, eps=1e-6):
  """Central-difference gradient of ``f`` with respect to argument ``k``."""
  args = list(args)
  if not isinstance(args[k], np.ndarray):
    hi = f(*args[:k], args[k] + eps, *args[k + 1 :])
    lo = f(*args[:k], args[k] - eps, *args[k + 1 :])
    return (hi - lo) / (2 * eps)
  a = args[k] = args[k].astype(float)
  g = np.zeros_like(a)
  for i in np.ndindex(*a.shape):
    old = a[i]
    a[i] = old + eps
    hi = f(*args)
    a[i] = old - eps
    lo = f(*args)
    a[i] = old
    g[i] = (hi - lo) / (2 * eps)
  return g


def check(f, *args, wrt=(0,)):
  got = grad(f, wrt if len(wrt) > 1 else wrt[0])(*args)
  got = got if len(wrt) > 1 else (got,)
  value, _ = value_and_grad(f, wrt[0])(*args)
  np.testing.assert_allclose(value, f(*args), rtol=1e-10)
  for k, g in zip(wrt, got):
    want = numeric(f, args, k)
    assert np.shape(g) == np.shape(want)
    np.testing.assert_allclose(g, want, rtol=1e-4, atol=1e-6, err_msg=f"argument {k}")


# ---- element-wise operators and broadcasting ------------------------------------


def test_elementwise():
  check(lambda x: np.sum(x * x), X)
  check(lambda x, y: np.sum(np.tanh(x * y) + x / y - np.exp(-x) * np.log(y)), X, Y, wrt=(0, 1))
  check(lambda x: np.sum(np.sin(x) * np.cos(x) + np.arctan(x) - np.sqrt(x) + np.log1p(x)), X)
  check(lambda x: np.sum(np.sinh(x) / np.cosh(x) + np.expm1(x) * 0.1 + np.arcsinh(x)), X)


def test_powers():
  check(lambda x, p: np.sum(x**3) + np.sum(x**p) + np.sum(1.0 / x), X, 1.5, wrt=(0, 1))
  check(lambda x, y: np.sum(x**y), X, Y, wrt=(0, 1))


def test_broadcasting_sums_the_adjoint_back():
  check(lambda x, b: np.sum((x + b) * b), X, V, wrt=(0, 1))
  check(lambda x, c: np.sum(x * c[:, None]), X, RNG.random(6), wrt=(0, 1))
  check(lambda x, s: np.sum(x * s) + s * s, X, 0.7, wrt=(0, 1))


def test_piecewise():
  check(lambda x: np.sum(np.where(x > 1.0, x, 0.1 * x)), X)
  check(lambda x: np.sum(np.maximum(x - 1.0, 0.0)) + np.sum(np.abs(x - 1.0)), X)
  check(lambda x: np.sum(np.clip(x, 0.8, 1.2) * x), X)
  check(lambda x, y: np.sum(np.minimum(x, y) * np.maximum(x, y) ** 2), X, Y, wrt=(0, 1))


def test_scalars_need_no_tape():
  check(lambda a, b: np.sin(a) * b + a / b, 0.3, 1.7, wrt=(0, 1))
  check(lambda a: (a * a if a > 0 else -a) + np.exp(a), 0.4)


# ---- reductions --------------------------------------------------------------------


def test_sums_and_means():
  check(lambda x: np.mean(x) + np.var(x) + np.std(x), X)
  check(lambda x: np.sum(np.sum(x, axis=0) ** 2) + np.sum(np.mean(x, axis=1, keepdims=True) * x), X)
  check(lambda x: np.linalg.norm(x) + np.sum(np.linalg.norm(x, axis=1)), X)
  check(lambda x: parakeet.reduce(parakeet.add, x.ravel()) ** 2, X)


def test_min_max_send_the_adjoint_to_the_winner():
  check(lambda x: np.max(x) + np.sum(np.max(x, axis=1)) - np.min(x) + np.sum(np.min(x, axis=0)), X)


def test_ties_share_the_adjoint():
  x = np.array([1.0, 3.0, 3.0, 2.0])
  np.testing.assert_allclose(grad(lambda x: np.max(x))(x), [0.0, 0.5, 0.5, 0.0])


def test_prod():
  check(lambda x: np.prod(x[0]) + np.sum(np.prod(x, axis=1)), X)


# ---- linear algebra ---------------------------------------------------------------


def test_matmul():
  check(lambda x, w: np.sum(np.tanh(x @ w)), X, W, wrt=(0, 1))
  check(lambda x, b: np.sum((x @ b) ** 2) + np.sum(b @ x.T), X, V, wrt=(0, 1))
  check(lambda a, b: a @ b + np.dot(a, a), V, V + 1, wrt=(0, 1))
  check(lambda x: np.sum(x.T @ x), X)


# ---- views --------------------------------------------------------------------------


def test_views_are_views_of_the_adjoint():
  check(lambda x: np.sum(x[1:, :2] * x[:-1, 2:]) + x[0, 0] * x[2, 3] + np.sum(x[:, 1] ** 2), X)
  check(lambda x: np.sum(x.reshape(4, 6)[1] * 2.0) + np.sum(x.T[::2] ** 2) + np.sum(x.ravel()[3:7]), X)
  check(lambda x: np.sum(np.exp(x)[::2, ::-1] * 3.0), X)


def test_gather_scatters_the_adjoint():
  idx = np.array([0, 2, 2, 5])
  check(lambda x, i: np.sum(x[i] ** 2) + np.sum(x[i, 1]), X, idx)
  rows, cols = np.array([0, 1, 1, 5]), np.array([3, 0, 0, 2])
  check(lambda x, r, c: np.sum(x[r, c] ** 3), X, rows, cols)


def test_assembled_arrays():
  check(
    lambda x, y: np.sum(np.concatenate((x, y * 2.0), axis=1) ** 2) + np.sum(np.stack((x, y)) * 3.0), X, Y, wrt=(0, 1)
  )
  check(lambda a, b: np.sum(np.array([a, b, a * b]) ** 2), 0.3, 1.2, wrt=(0, 1))


def test_reused_values():
  def f(x):
    y = np.exp(x)
    return np.sum(y * y) + np.sum(y) + np.sum(y[0])

  check(f, X)


# ---- adverbs ---------------------------------------------------------------------------


def test_map_differentiates_its_function_per_element():
  check(lambda x: np.sum(each(lambda row: np.sum(row * row) * row[0], x)), X)
  check(lambda x: np.sum(np.array([np.sum(r) ** 2 for r in x])), X)
  check(lambda x: np.sum(each(lambda row: np.tanh(row) * 2.0, x) ** 2), X)


def test_allpairs():
  check(lambda x, y: np.sum(allpairs(lambda a, b: np.sqrt(np.sum((a - b) ** 2) + 1.0), x, y)), X, Y, wrt=(0, 1))


def test_values_captured_by_a_mapped_function():
  def f(w, x):
    return np.sum(each(lambda row: np.sum(np.tanh(row @ w)), x))

  check(f, W, X)
  check(lambda s, x: np.sum(each(lambda row: np.sum(row) * s, x)), 0.7, X)


def test_softmax_cross_entropy():
  def loss(x):
    z = x - np.max(x, axis=1, keepdims=True)
    logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    return -np.mean(logp[:, 0])

  check(loss, X)


# ---- structure of the differentiated arguments ---------------------------------------------


def mlp(params, x, y):
  (w1, b1), (w2, b2) = params
  h = np.tanh(x @ w1 + b1)
  return np.mean((h @ w2 + b2 - y) ** 2)


def test_tuples_of_parameters_give_tuples_of_gradients():
  params = ((RNG.random((4, 5)), RNG.random(5)), (RNG.random((5, 3)), RNG.random(3)))
  grads = grad(mlp)(params, X, T)
  flat = [params[0][0], params[0][1], params[1][0], params[1][1]]
  got = [grads[0][0], grads[0][1], grads[1][0], grads[1][1]]
  for k, (p, g) in enumerate(zip(flat, got)):

    def f(a, k=k):
      q = list(flat)
      q[k] = a
      return mlp(((q[0], q[1]), (q[2], q[3])), X, T)

    np.testing.assert_allclose(g, numeric(f, [p], 0), rtol=1e-4, atol=1e-7)


def test_value_and_grad_with_several_arguments():
  value, (ga, gb) = value_and_grad(lambda a, b: np.sum(a) * np.sum(b), (0, 1))(X, W)
  assert np.isclose(value, X.sum() * W.sum())
  np.testing.assert_allclose(ga, np.full(X.shape, W.sum()))
  np.testing.assert_allclose(gb, np.full(W.shape, X.sum()))


def test_unused_argument_has_a_zero_gradient():
  g = grad(lambda a, b: np.sum(b * b), 0)(X, W)
  assert g.shape == X.shape and not g.any()


def test_float32_stays_float32():
  w, x, t = W.astype(np.float32), X.astype(np.float32), T.astype(np.float32)
  g = grad(lambda w, x, t: np.mean((np.tanh(x @ w) - t) ** 2))(w, x, t)
  assert g.dtype == np.float32
  np.testing.assert_allclose(g, grad(lambda w, x, t: np.mean((np.tanh(x @ w) - t) ** 2))(W, X, T), rtol=1e-3)


def test_strided_and_repeated_arguments():
  f = lambda a, b: np.sum(a * b * b)  # noqa: E731
  np.testing.assert_allclose(grad(f)(X[::2, ::-1], Y[::2]), (Y[::2] ** 2))
  ga, gb = grad(f, (0, 1))(X, X)
  np.testing.assert_allclose(ga, X * X)
  np.testing.assert_allclose(gb, 2 * X * X)


# ---- inside compiled code -------------------------------------------------------------------------


def loss(w, x, t):
  return np.mean((np.tanh(x @ w) - t) ** 2)


@jit
def descend(w, x, t, rate, steps):
  for _ in range(steps):
    w = w - rate * grad(loss)(w, x, t)
  return w


@jit
def step(w, x, t, rate):
  value, g = value_and_grad(loss)(w, x, t)
  return w - rate * g, value


def test_grad_inside_compiled_code():
  w, value = step(W, X, T, 0.1)
  np.testing.assert_allclose(w, W - 0.1 * grad(loss)(W, X, T))
  assert np.isclose(value, loss(W, X, T))


def test_training_loop_in_compiled_code():
  want = W.copy()
  for _ in range(20):
    want = want - 0.5 * grad(loss)(want, X, T)
  np.testing.assert_allclose(descend(W, X, T, 0.5, 20), want)
  assert loss(want, X, T) < loss(W, X, T)


def test_a_gradient_object_can_be_a_global():
  dloss = grad(loss)

  @jit
  def f(w, x, t):
    return dloss(w, x, t) * 2.0

  np.testing.assert_allclose(f(W, X, T), 2 * dloss(W, X, T))


# ---- loops ------------------------------------------------------------------------------------------


def test_loops_over_constants_are_unrolled():
  def f(x):
    h = x
    for i in range(3):
      h = np.tanh(h) * (i + 1.0)
    return np.sum(h)

  check(f, X)


def test_loop_length_from_an_argument_or_a_shape():
  def layers(w, x, n):
    h = x
    for _ in range(n):
      h = np.tanh(h @ w @ w.T)
    return np.sum(h)

  np.testing.assert_allclose(grad(layers)(W, X, 2), numeric(lambda w: layers(w, X, 2), [W], 0), rtol=1e-4)

  def correlate(x, w):
    out = x[: x.shape[0] - w.shape[0] + 1] * 0.0
    for i in range(w.shape[0]):
      out = out + x[i : i + out.shape[0]] * w[i]
    return np.sum(out * out)

  check(correlate, np.arange(10.0), np.array([1.0, -2.0, 0.5]), wrt=(0, 1))
  # a different kernel length is a different specialization
  check(correlate, np.arange(9.0), np.array([1.0, -2.0]), wrt=(0, 1))


def test_loops_over_tuples():
  def f(params, x):
    h = x
    for w in params:
      h = np.tanh(h @ w)
    return np.sum(h)

  params = (RNG.random((4, 4)), RNG.random((4, 2)))
  g = grad(f)(params, X)
  np.testing.assert_allclose(g[1], numeric(lambda w: f((params[0], w), X), [params[1]], 0), rtol=1e-4)


def test_code_after_a_guard_is_still_straight_line():
  def f(w, x):
    if x.shape[0] == 0:
      raise ValueError("empty")
    return np.sum(x @ w)

  np.testing.assert_allclose(grad(f)(W, X), np.repeat(X.sum(0)[:, None], 3, axis=1))


def test_things_computed_from_but_not_differentiated():
  def f(w, x, t):
    p = x @ w
    hits = np.sum(np.argmax(p, axis=1) == np.argmax(t, axis=1))
    return np.sum(p * p) + 0.0 * hits

  np.testing.assert_allclose(grad(f)(W, X, T), grad(lambda w, x: np.sum((x @ w) ** 2))(W, X))


# ---- what is refused -----------------------------------------------------------------------------------


def refused(f, *args, match, argnum=0, error=parakeet.UnsupportedError):
  with pytest.raises(error, match=match):
    grad(f, argnum)(*args)


def test_if_statements_run_both_branches_and_choose():
  def assigned(w, x):
    s = np.sum(w)
    if x[0, 0] > 0.5:
      s = s * 2.0
    return s * s

  def computed(w, x):
    h = x @ w
    if np.sum(x) > 10.0:
      h, k = np.tanh(h), 2.0
    else:
      h, k = h * h, 3.0
    return np.sum(h) * k

  def nested(w, x):
    s = np.sum(w * w)
    if x[0, 0] > 0.2:
      if x[1, 1] > 0.5:
        s = s * 3.0
      else:
        s = s + np.sum(w)
    return s

  def in_a_map(w, x):
    def f(row):
      t = np.sum(row @ w)
      if t > 1.5:
        t = t * t
      return t

    return np.sum(each(f, x))

  for f in (assigned, computed, nested, in_a_map):
    for x in (X, X * 0.1, X * 3.0):  # both sides of every condition
      np.testing.assert_allclose(grad(f)(W, x), numeric(f, [W, x], 0), rtol=1e-4, atol=1e-6, err_msg=f.__name__)


def test_branches_that_cannot_be_chosen_between():
  def returns(w, x):
    if x[0, 0] > 0.5:
      return np.sum(w)
    return np.sum(w * w)

  def stores(w, x):
    out = np.zeros(3)
    if x[0, 0] > 0.5:
      out[0] = np.sum(w)
    return np.sum(out)

  def updates_in_place(w, x):
    h = w * 2.0
    if x[0, 0] > 0.5:
      h = h + 1.0
      h += w
    return np.sum(h)

  refused(returns, W, X, match="only runs conditionally")
  refused(stores, W, X, match="only runs conditionally")
  refused(updates_in_place, W, X, match="in-place update inside an if statement")


def test_loops_that_cannot_be_unrolled():
  def runtime(w, x):
    s = 0.0
    for v in x.ravel():
      s = s + np.sum(w * v)
    return s

  def forever(w, x):
    s, i = np.sum(w), 0
    while i < 3:
      s, i = s * 1.5, i + 1
    return s

  def long(w):
    s = np.sum(w)
    for _ in range(100000):
      s = s * 1.0
    return s

  refused(runtime, W, X, match="length is only known at run time")
  refused(forever, W, X, match="through this loop")
  refused(long, W, match="100000 iterations")


def test_in_place_updates():
  def elements(w, x):
    out = np.zeros(3)
    for i in range(3):
      out[i] = np.sum(w[:, i]) * (i + 1.0)
    return np.sum(out * out)

  def slices(w, x):
    out = np.zeros((6, 3))
    out[:3] = x[:3] @ w
    out[3:] = np.tanh(x[3:] @ w)
    return np.sum(out**2)

  def accumulated(w, x):
    out = np.zeros((6, 3))
    for i in range(2):
      out += x @ w * (i + 1.0)
    out[0] += 1.0
    return np.sum(out * out)

  def overwritten(w, x):
    h = w * 2.0
    y = np.sum(h * 3.0)
    h[0, 0] = 5.0  # the earlier use needs the adjoint of h, not its old value
    return y + np.sum(h * h)

  def argument_overwritten(w, x):
    w[1] = x[0, :3]
    return np.sum(w * w)

  def read_and_written(w, x):
    out = w * 1.0
    out[0] = out[1] * out[2]
    out[:, 0] = out[:, 1] + 1.0
    return np.sum(out**3)

  def scaled_in_place(w, x):
    h = w * 1.0
    h *= x[0, :3]
    return np.sum(h * h)

  for f in (elements, slices, accumulated, overwritten, argument_overwritten, read_and_written, scaled_in_place):
    want = numeric(lambda w, f=f: f(w.copy(), X), [W + 0.5], 0)
    np.testing.assert_allclose(grad(f)(W + 0.5, X), want, rtol=1e-4, atol=1e-6, err_msg=f.__name__)


def test_values_overwritten_before_their_derivative_needed_them():
  def modified_afterwards(w, x):
    h = w * 2.0
    y = np.sum(np.tanh(h))
    h[0, 0] = 5.0
    return y

  def input_modified(w, x):
    y = np.sum(np.tanh(w * x[:4, :3]))
    x[0, 0] = 9.0
    return y

  def stored_through_an_index_array(w, x):
    out = np.zeros(4)
    out[np.array([0, 2])] = w[:2, 0]
    return np.sum(out)

  refused(modified_afterwards, W, X, match="modified in place")
  refused(input_modified, W, X.copy(), match="modified in place")
  refused(stored_through_an_index_array, W, X, match="assigned one by one")


def test_library_functions_without_a_derivative_are_named():
  refused(lambda w: np.sum(np.cumprod(w)), W, match=r"cumprod\(\)")
  refused(lambda w: np.sum(np.convolve(w.ravel(), w.ravel())), W, match=r"convolve\(\)")
  refused(lambda w: np.sum(parakeet.scan(parakeet.multiply, w.ravel())), W, match=r"scan\(\)")


def test_cumsum_sort_and_folds():
  v = RNG.random(7) + 0.5
  check(lambda a: np.sum(np.cumsum(a) ** 2), v)
  check(lambda a: np.sum(np.cumsum(a, axis=1) * np.cumsum(a, axis=0)) + np.sum(np.cumsum(a) ** 2), X)
  check(lambda a: np.sum(parakeet.scan(parakeet.add, a) ** 3), v)
  check(lambda a: np.sum(np.sort(a) * np.arange(7.0)), v)
  check(lambda a: np.sum(np.sort(a, axis=0) * a[::-1]) + np.sum(np.sort(a)[:, :2]), X)
  check(lambda a: np.sort(a.ravel())[3] * 2.0, X)
  # a fold with any function: the running values are kept and replayed backwards
  check(lambda a: parakeet.reduce(lambda acc, x: acc * 0.9 + x * x, a), v)
  check(lambda a, s: parakeet.reduce(lambda acc, x: acc * x + s, a, init=s), v, 0.7, wrt=(0, 1))
  check(lambda a, w: parakeet.reduce(lambda acc, x: np.tanh(acc + x * np.sum(w)), a, init=0.1), v, X, wrt=(0, 1))
  check(lambda c, x: parakeet.reduce(lambda acc, ci: acc * x + ci, c), v, 0.8, wrt=(0, 1))  # Horner


def test_stacked_products_and_einsum():
  a, b, m = RNG.random((3, 4, 5)), RNG.random((3, 5, 6)), RNG.random((5, 6))
  check(lambda a, b: np.sum(np.tanh(a @ b)), a, b, wrt=(0, 1))
  check(lambda a, b: np.sum((a @ b) ** 2), a, m, wrt=(0, 1))
  check(lambda a, b: np.sum(np.einsum("bij,bjk->bik", a, b) ** 2), a, b, wrt=(0, 1))
  check(lambda a, b: np.sum(np.einsum("ij,jk->i", a, b) ** 3), m.T.copy(), m, wrt=(0, 1))
  check(lambda a: np.sum(np.sum(a, axis=(0, 2)) ** 2) + np.sum(np.max(a, axis=(1, 2))), a)


def test_wrong_uses():
  T_ = parakeet.TypeInferenceError
  refused(lambda w, x: x @ w, W, X, match="returns a floating point scalar", error=T_)
  refused(lambda a: np.sum(a), np.arange(4), match="floating point scalars and arrays", error=T_)
  refused(loss, W, X, T, argnum=5, match="no positional argument 5", error=T_)
  refused(lambda w: np.sum(grad(lambda a: np.sum(a * a))(w)), W, match="itself being differentiated")
  with pytest.raises(TypeError, match="expects a function"):
    grad(3)
  with pytest.raises(TypeError, match="argnum"):
    grad(loss, -1)


def test_unknown_derivative_of_a_primitive():
  import math

  refused(lambda a: math.lgamma(a) * 1.0, 2.5, match="lgamma")


# ---- other backends ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["c", "auto"])
def test_backends_agree(backend):
  params = ((RNG.random((4, 5)), RNG.random(5)), (RNG.random((5, 3)), RNG.random(3)))
  want = grad(mlp)(params, X, T)
  with parakeet.options(backend=backend, fusion=backend == "auto", tiling=backend == "auto"):
    got = grad(mlp)(params, X, T)
  for a, b in zip(got, want):
    for u, v in zip(a, b):
      np.testing.assert_allclose(u, v, rtol=1e-10)
