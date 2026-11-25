"""Tests for Wave 3 (v1.4.0, #79 — Python Semantics 1):
- Non-frozen @dataclass with pure methods, @property, __call__, staticmethod/classmethod,
  local functional field rebinding (obj.x = v, obj.a.b = v, obj.x += v) including inside loops
  and under grad / jvp with mixed float + discrete metadata fields
- Structural match/case destructuring (non-frozen dataclasses, NamedTuple, nested tuples,
  starred array patterns, MatchMapping dicts, RaggedArray/ndarray class patterns, guards, OR)
- Compile-time isinstance (including UnionType A | B, dataclasses, RaggedArray), hasattr, getattr,
  and None specialization with dead-branch pruning
- Python call ergonomics: *args / **kwargs forwarding, keyword-only args, functools.partial,
  operator.* callables (including itemgetter, attrgetter, associative reduce/scan combiners),
  and bound methods on records passed into @par.jit / par.map
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pytest

import parakeet as par
from parakeet import RaggedArray
from parakeet.errors import TypeInferenceError


@dataclass
class LinearLayer:
  weight: np.ndarray
  bias: np.ndarray
  scale: float = 1.0
  name: str = "dense"

  @property
  def out_features(self):
    return self.weight.shape[1]

  def forward(self, x):
    return np.tanh(x @ self.weight + self.bias) * self.scale

  def __call__(self, x):
    return self.forward(x)

  @staticmethod
  def norm2(x):
    return np.sum(x * x)

  @classmethod
  def scaled_zero(cls, weight, bias, scale):
    return cls(weight=weight, bias=bias, scale=scale)


@dataclass
class Particle:
  pos: np.ndarray
  vel: np.ndarray
  mass: float
  steps: int = 0

  @property
  def kinetic_energy(self):
    return 0.5 * self.mass * np.sum(self.vel * self.vel)

  def advance(self, force, dt):
    self.vel = self.vel + (dt / self.mass) * force
    self.pos = self.pos + dt * self.vel
    self.steps += 1
    return self


@dataclass
class Wrapper:
  inner: Particle
  tag: str = "sim"


@dataclass(frozen=True)
class FrozenPoint:
  x: float
  y: float

  @property
  def r2(self):
    return self.x * self.x + self.y * self.y


class Pair(NamedTuple):
  a: float
  b: float

  @property
  def total(self):
    return self.a + self.b

  def weighted(self, w):
    return w * self.a + (1.0 - w) * self.b


# ---------------------------------------------------------------------------
# 1. Non-frozen @dataclass, pure methods, @property, __call__, field rebinding
# ---------------------------------------------------------------------------


def test_non_frozen_dataclass_methods_property_and_call():
  rng = np.random.default_rng(0)
  w = rng.normal(size=(4, 3))
  b = rng.normal(size=(3,))
  x = rng.normal(size=(5, 4))
  layer = LinearLayer(w, b, scale=0.75, name="fc1")

  @par.jit
  def run_layer(lyr, inp):
    y1 = lyr(inp)
    y2 = lyr.forward(inp)
    built = LinearLayer.scaled_zero(lyr.weight, lyr.bias, lyr.scale)
    return y1 + y2 + built(inp), lyr.out_features, LinearLayer.norm2(y1)

  out, n_out, n2 = run_layer(layer, x)
  expected_y = np.tanh(x @ w + b) * 0.75
  np.testing.assert_allclose(out, 3.0 * expected_y, rtol=1e-12)
  assert int(n_out) == 3
  assert float(n2) == pytest.approx(float(np.sum(expected_y * expected_y)), rel=1e-12)

  # Bound method on record passed directly into par.map from Python!
  mapped = par.map(layer.forward, x)
  np.testing.assert_allclose(mapped, expected_y, rtol=1e-12)


def test_non_frozen_dataclass_field_rebinding_loops_and_ad():
  @par.jit
  def simulate(p, force, dt, n_steps):
    for _ in range(n_steps):
      p.vel = p.vel + (dt / p.mass) * force
      p.pos += dt * p.vel
      p.steps += 1
    return p, p.kinetic_energy

  pos0 = np.array([1.0, -2.0, 0.5])
  vel0 = np.array([0.2, 0.1, -0.3])
  force = np.array([0.5, -0.4, 0.2])
  p0 = Particle(pos0.copy(), vel0.copy(), mass=2.0, steps=0)

  p_out, ke = simulate(p0, force, 0.1, 6)
  # Pure functional value semantics: caller's p0 is NOT mutated
  np.testing.assert_array_equal(p0.pos, pos0)
  assert p0.steps == 0

  # Compare against Python reference
  v_ref, x_ref = vel0.copy(), pos0.copy()
  for _ in range(6):
    v_ref = v_ref + (0.1 / 2.0) * force
    x_ref = x_ref + 0.1 * v_ref
  np.testing.assert_allclose(p_out.pos, x_ref, rtol=1e-12)
  np.testing.assert_allclose(p_out.vel, v_ref, rtol=1e-12)
  assert p_out.steps == 6
  assert float(ke) == pytest.approx(float(0.5 * 2.0 * np.sum(v_ref * v_ref)), rel=1e-12)

  # Nested field assignment `w.inner.mass = ...` & `dataclasses.replace`
  @par.jit
  def bump_nested(wrap):
    wrap.inner.mass *= 2.0
    p2 = dataclasses.replace(wrap.inner, steps=99)
    return wrap.inner.mass, p2.steps

  m2, s2 = bump_nested(Wrapper(p0))
  assert float(m2) == pytest.approx(4.0)
  assert int(s2) == 99

  # Frozen dataclass field assignment is refused loudly
  @par.jit
  def bad_frozen(pt):
    pt.x = 2.0
    return pt

  with pytest.raises(TypeInferenceError, match="frozen dataclass"):
    bad_frozen(FrozenPoint(1.0, 2.0))

  # grad and jvp through non-frozen dataclass with methods + mixed float & discrete fields!
  def loss(lyr, inp):
    return np.sum(lyr(inp) ** 2)

  rng = np.random.default_rng(7)
  lyr = LinearLayer(rng.normal(size=(4, 3)), rng.normal(size=(3,)), scale=1.2, name="mlp")
  inp = rng.normal(size=(5, 4))
  g_lyr = par.grad(loss, 0)(lyr, inp)
  assert g_lyr.name == "mlp"

  # Verify against finite differences on weight, bias, and scale
  eps = 1e-6
  g_scale_fd = (
    loss(LinearLayer(lyr.weight, lyr.bias, lyr.scale + eps, "mlp"), inp)
    - loss(LinearLayer(lyr.weight, lyr.bias, lyr.scale - eps, "mlp"), inp)
  ) / (2 * eps)
  assert float(g_lyr.scale) == pytest.approx(float(g_scale_fd), rel=1e-5, abs=1e-6)

  d_lyr = LinearLayer(rng.normal(size=(4, 3)), rng.normal(size=(3,)), scale=0.3, name="mlp")
  _, d_out = par.jvp(lambda l_: loss(l_, inp), (lyr,), (d_lyr,))
  expected_jvp = float(
    np.sum(g_lyr.weight * d_lyr.weight) + np.sum(g_lyr.bias * d_lyr.bias) + g_lyr.scale * d_lyr.scale
  )
  assert float(d_out) == pytest.approx(expected_jvp, rel=1e-10)


def test_namedtuple_methods_and_property():
  @par.jit
  def eval_pair(p, w):
    return p.total + p.weighted(w)

  p = Pair(3.0, 7.0)
  assert float(eval_pair(p, 0.25)) == pytest.approx(10.0 + 0.25 * 3.0 + 0.75 * 7.0)


# ---------------------------------------------------------------------------
# 2. Structural match/case destructuring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["c", "multicore"])
def test_structural_match_case_destructuring(backend):
  @par.jit(backend=backend)
  def dispatch(obj):
    match obj:
      case FrozenPoint(x, y) if x > 0.0:
        return x * y + obj.r2
      case FrozenPoint(x=x, y=y):
        return y - x
      case Pair(a, b):
        return a * 2.0 + b
      case {"kind": "scale", "val": v, **rest}:
        return v * rest["factor"]
      case (a, (b, c)) | (a, b, c):
        return a + 2.0 * b + 3.0 * c
      case RaggedArray(values=vals, offsets=offs):
        return np.sum(vals) + float(len(offs))
      case [first, *mid, last]:
        return first + np.sum(mid) + last
      case _:
        return -1.0

  assert float(dispatch(FrozenPoint(2.0, 3.0))) == pytest.approx(6.0 + 13.0)
  assert float(dispatch(FrozenPoint(-2.0, 3.0))) == pytest.approx(5.0)
  assert float(dispatch(Pair(4.0, 5.0))) == pytest.approx(13.0)
  assert float(dispatch({"kind": "scale", "val": 3.0, "factor": 4.0})) == pytest.approx(12.0)
  assert float(dispatch((1.0, (2.0, 3.0)))) == pytest.approx(14.0)
  assert float(dispatch((1.0, 2.0, 3.0))) == pytest.approx(14.0)
  rag = RaggedArray.from_lengths(np.array([1.0, 2.0, 3.0]), np.array([1, 2], dtype=np.int64))
  assert float(dispatch(rag)) == pytest.approx(6.0 + 3.0)
  assert float(dispatch(np.array([10.0, 1.0, 2.0, 20.0]))) == pytest.approx(33.0)


# ---------------------------------------------------------------------------
# 3. Compile-time isinstance, hasattr, getattr, and None dead-branch pruning
# ---------------------------------------------------------------------------


def test_compile_time_isinstance_hasattr_getattr_and_none_pruning():
  @par.jit
  def smart_reduce(x, mask=None):
    # Dead branch accesses invalid attributes for the inactive type -> proves compile-time pruning!
    if isinstance(x, RaggedArray):
      vals = x.values
      base = np.sum(vals)
    elif isinstance(x, LinearLayer):
      base = np.sum(x.weight) + np.sum(x.bias)
    elif isinstance(x, int | float):
      base = float(x) * 2.0
    else:
      base = np.sum(x)
    if mask is not None:
      return base + np.sum(mask)
    return base

  rag = RaggedArray.from_lengths(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2, 2], dtype=np.int64))
  lyr = LinearLayer(np.ones((2, 2)), np.full((2,), 0.5))
  arr = np.array([1.0, 2.0, 3.0])
  mask = np.array([10.0, 20.0])

  assert float(smart_reduce(rag)) == pytest.approx(10.0)
  assert float(smart_reduce(lyr)) == pytest.approx(5.0)
  assert float(smart_reduce(3.5)) == pytest.approx(7.0)
  assert float(smart_reduce(arr, mask)) == pytest.approx(36.0)

  @par.jit
  def inspect_attrs(obj):
    s = 0.0
    if hasattr(obj, "r2"):
      s = s + obj.r2
    if hasattr(obj, "nonexistent_field"):
      s = s + obj.nonexistent_field  # dead branch must be pruned!
    return s + getattr(obj, "x", 100.0) + getattr(obj, "missing", 5.0)

  assert float(inspect_attrs(FrozenPoint(3.0, 4.0))) == pytest.approx(25.0 + 3.0 + 5.0)
  assert float(inspect_attrs(Pair(1.0, 2.0))) == pytest.approx(0.0 + 100.0 + 5.0)


# ---------------------------------------------------------------------------
# 4. Call ergonomics: *args / **kwargs forwarding, kw-only, partial, operator.*
# ---------------------------------------------------------------------------


def test_call_ergonomics_args_kwargs_partial_and_operator():
  def affine(x, weight, *, bias=0.0, scale=1.0):
    return (x * weight + bias) * scale

  def forward_wrapper(fn, *args, extra=0.0, **kwargs):
    return fn(*args, **kwargs) + extra

  @par.jit
  def run_calls(x):
    p_affine = functools.partial(affine, bias=1.5, scale=2.0)
    a = forward_wrapper(affine, x, 3.0, bias=1.0, scale=0.5, extra=2.0)
    b = p_affine(x, 2.0)
    # operator.* in direct calls and as associative reduce/scan combiners!
    s_add = par.reduce(operator.add, x)
    s_mul = par.reduce(operator.mul, x)
    c_add = par.scan(operator.add, x)
    get02 = operator.itemgetter(0, 2)
    t0, t2 = get02(x)
    return a, b, s_add, s_mul, c_add, t0 + t2

  x = np.array([1.0, 2.0, 3.0, 4.0])
  a, b, s_add, s_mul, c_add, t02 = run_calls(x)
  np.testing.assert_allclose(a, (x * 3.0 + 1.0) * 0.5 + 2.0, rtol=1e-12)
  np.testing.assert_allclose(b, (x * 2.0 + 1.5) * 2.0, rtol=1e-12)
  assert float(s_add) == pytest.approx(10.0)
  assert float(s_mul) == pytest.approx(24.0)
  np.testing.assert_allclose(c_add, np.cumsum(x), rtol=1e-12)
  assert float(t02) == pytest.approx(4.0)

  # Passing functools.partial and operator.attrgetter INTO @par.jit from Python!
  @par.jit
  def apply_fn(fn, arg):
    return fn(arg)

  ext_partial = functools.partial(affine, weight=4.0, bias=1.0, scale=0.5)
  np.testing.assert_allclose(apply_fn(ext_partial, x), (x * 4.0 + 1.0) * 0.5, rtol=1e-12)

  get_r2 = operator.attrgetter("r2")
  assert float(apply_fn(get_r2, FrozenPoint(3.0, 4.0))) == pytest.approx(25.0)
  get_nested = operator.attrgetter("inner.mass")
  assert float(apply_fn(get_nested, Wrapper(Particle(x, x, mass=7.5)))) == pytest.approx(7.5)

  # Passing functools.partial into par.map directly!
  mapped = par.map(functools.partial(operator.add, 10.0), x)
  np.testing.assert_allclose(mapped, x + 10.0, rtol=1e-12)
