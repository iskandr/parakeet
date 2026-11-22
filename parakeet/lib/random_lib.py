"""``np.random`` inside compiled code.

Compiled code uses a counter-based generator: every draw is a pure function
of a per-call key and the element's position, so random arrays can be filled
in parallel (and on GPUs) and still be reproducible after ``parakeet.seed``.
The streams are *not* the same as NumPy's.
"""

from __future__ import annotations

import math

import numpy as np

from ..errors import UnsupportedError
from ..ndtypes import Float64, Int64, WeakFloat, WeakInt
from ..syntax import ops
from ..syntax.ir import Const, Expr, RtCall, Var
from ..syntax.ops import I, cast, cmp, iadd, imul
from .core import register
from .pure import source

TWO_PI = 2.0 * math.pi


def _key(el) -> Expr:
  return el.tmp(RtCall("rng_key", (), I), "key")


def _flat(el, idx, strides) -> Expr:
  pos: Expr = ops.ZERO
  for i, s in zip(idx, strides):
    pos = iadd(pos, imul(i, s))
  return pos


def _size(el, size, dims=()) -> tuple | None:
  """Shape requested through ``size=`` or positional dims; None for a scalar."""
  if dims:
    return el.check_dims(tuple(el.as_index(d, "dimension") for d in dims))
  if size is None:
    return None
  return el.check_dims(el.shape_tuple(size, "size"))


def _draw(el, shape, sample, elt, weak):
  """Scalar draw or delayed array of draws; ``sample(key, i)`` builds one."""
  key = _key(el)
  if shape is None:
    return cast(sample(key, ops.ZERO), weak)
  strides = el.c_strides(shape)
  return el.delayed(elt, shape, lambda idx: sample(key, _flat(el, idx, strides)), cost=4)


def _uniform(key, i) -> Expr:
  return RtCall("rand_f64", (key, i), Float64, pure=True)


def _normal(key, i) -> Expr:
  # Box-Muller on two independent uniforms; 1 - u is in (0, 1] so the log is finite
  u1 = RtCall("rand_f64", (key, imul(i, 2)), Float64, pure=True)
  u2 = RtCall("rand_f64", (key, iadd(imul(i, 2), 1)), Float64, pure=True)
  one = Const(1.0, Float64)
  radius = ops.prim(
    "sqrt",
    (
      ops.prim(
        "mul", (Const(-2.0, Float64), ops.prim("log", (ops.prim("sub", (one, u1), Float64),), Float64)), Float64
      ),
    ),
    Float64,
  )
  angle = ops.prim("cos", (ops.prim("mul", (Const(TWO_PI, Float64), u2), Float64),), Float64)
  return ops.prim("mul", (radius, angle), Float64)


def _affine(el, value, loc, scale):
  if scale is not None:
    value = el.binop("mul", value, scale)
  if loc is not None:
    value = el.binop("add", value, loc)
  return value


@register(np.random.rand)
def rand(el, *dims):
  return _draw(el, _size(el, None, dims) if dims else None, _uniform, Float64, WeakFloat)


@register(np.random.random, np.random.random_sample, np.random.ranf, np.random.sample)
def random(el, size=None):
  return _draw(el, _size(el, size), _uniform, Float64, WeakFloat)


@register(np.random.uniform)
def uniform(el, low=0.0, high=1.0, size=None):
  low_s = el.as_scalar(low, "low")
  span = el.binop("sub", el.as_scalar(high, "high"), low_s)
  return _affine(el, _draw(el, _size(el, size), _uniform, Float64, WeakFloat), low_s, span)


@register(np.random.randn)
def randn(el, *dims):
  return _draw(el, _size(el, None, dims) if dims else None, _normal, Float64, WeakFloat)


@register(np.random.standard_normal)
def standard_normal(el, size=None):
  return _draw(el, _size(el, size), _normal, Float64, WeakFloat)


@register(np.random.normal)
def normal(el, loc=0.0, scale=1.0, size=None):
  return _affine(
    el,
    _draw(el, _size(el, size), _normal, Float64, WeakFloat),
    el.as_scalar(loc, "loc"),
    el.as_scalar(scale, "scale"),
  )


@register(np.random.randint)
def randint(el, low, high=None, size=None, dtype=None):
  if high is None:
    low, high = Const(0, I), low
  lo = el.tmp(el.as_index(low, "low"))
  hi = el.tmp(el.as_index(high, "high"))
  el.check(cmp("lt", lo, hi), "ValueError", "low >= high in randint ({0} >= {1})", lo, hi)
  elt = Int64 if dtype is None else el.as_dtype(dtype)

  def sample(key, i):
    return cast(RtCall("rand_int", (key, i, lo, hi), I, pure=True), elt)

  key = _key(el)
  shape = _size(el, size)
  if shape is None:
    return cast(sample(key, ops.ZERO), WeakInt if dtype is None else elt)
  strides = el.c_strides(shape)
  return el.delayed(elt, shape, lambda idx: sample(key, _flat(el, idx, strides)), cost=4)


@register(np.random.seed)
def _seed(el, *args):
  raise el.error(
    UnsupportedError,
    "np.random.seed() cannot be called from compiled code",
    hint="call parakeet.seed(n) from Python before calling the compiled function",
  )


@source(np.random.permutation)
def permutation(n):
  """Random permutation of ``range(n)`` (Fisher-Yates)."""
  x = np.arange(n)
  for i in range(n - 1, 0, -1):
    j = np.random.randint(0, i + 1)
    t = x[i]
    x[i] = x[j]
    x[j] = t
  return x


@source(np.random.shuffle)
def shuffle(x):
  """Shuffle a 1-d array in place (Fisher-Yates)."""
  for i in range(len(x) - 1, 0, -1):
    j = np.random.randint(0, i + 1)
    t = x[i]
    x[i] = x[j]
    x[j] = t


__parakeet_lib__ = True
_ = Var
