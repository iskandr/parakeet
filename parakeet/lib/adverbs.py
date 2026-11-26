"""Parakeet's higher-order array operators: map, reduce, scan, allpairs, ...

Inside compiled code these are fused with their function arguments.  Called
directly from Python they compile a small wrapper on the fly, so
``parakeet.map(f, x)`` works without decorating anything.
"""

from __future__ import annotations

import builtins
import functools

import numpy as np

from .. import prims
from ..errors import StaticValueError, TypeInferenceError, UnsupportedError
from ..ndtypes import Int64
from ..syntax import ops
from ..syntax.ir import Const, Expr, Var
from ..syntax.ops import ZERO, I, cast, cmp
from ..syntax.values import CHEAP_COST, ArrayVal, RangeVal, describe
from . import core
from .core import FUNCTIONS, NAMES, along_axis, build_map, norm_axis, reduce_arrays, reduce_values, scan_values


class LibFunction:
  """A library function usable inside compiled code *and* callable from Python."""

  def __init__(self, name: str, impl, doc: str | None = None):
    self.__name__ = name
    self.__qualname__ = name
    self.__doc__ = doc or impl.__doc__
    self.__module__ = "parakeet"
    self.impl = impl
    FUNCTIONS[self] = impl
    NAMES[self] = name

  def __repr__(self) -> str:
    return f"<parakeet function {self.__name__}>"

  def __call__(self, *args, **kwargs):
    from ..api import call_staged

    return call_staged(self, args, kwargs)


def libfunction(name: str | None = None):
  def deco(impl):
    return LibFunction(name or impl.__name__, impl)

  return deco


def _arrayish(el, v):
  """Arrays stay arrays; tuples/ranges become arrays; scalars pass through."""
  if isinstance(v, (tuple, RangeVal)):
    return el.as_array(v)
  return v


def _prepare(el, a: ArrayVal) -> ArrayVal:
  """Make ``a`` safe to slice once per iteration of a data-parallel loop."""
  if a.buf is None and a.cost > CHEAP_COST:
    a = el.as_concrete(a)
  el.consume(a)
  return a


def _axis_arg(el, axis, what: str = "axis"):
  if axis is None:
    return None
  if isinstance(axis, tuple):
    return tuple(el.static_int(a, what) for a in axis)
  return el.static_int(axis, what)


@libfunction("map")
def map(el, f, *args, axis=0):  # noqa: A001
  """Apply ``f`` to every element along ``axis`` of the argument arrays.

  With ``axis=None`` the function is applied to every scalar element (with
  broadcasting).  Non-array arguments are passed to every call unchanged."""
  from ..ragged import RaggedArray

  if not args:
    raise el.error(TypeInferenceError, "map() needs at least one array argument")
  vals = [_arrayish(el, a) for a in args]
  arrays = [a for a in vals if (isinstance(a, ArrayVal) and a.rank > 0) or isinstance(a, RaggedArray)]
  if not arrays:
    return el.call(f, [el.as_scalar(v) if isinstance(v, ArrayVal) else v for v in vals])
  ax = _axis_arg(el, axis)
  if ax is None:
    if any(isinstance(a, RaggedArray) for a in arrays):
      raise el.error(UnsupportedError, "map(..., axis=None) is not supported for RaggedArray; use axis=0")
    shape = el.broadcast_shapes([a.shape for a in arrays], "arguments of map")
    adapted = []
    for v in vals:
      if isinstance(v, ArrayVal):
        if v.rank == 0:
          adapted.append(el.as_scalar(v))
        else:
          v = _prepare(el, v)
          adapted.append(el.broadcast_to(v, shape, consumed=True))
      else:
        adapted.append(v)

    def at(idx):
      return el.call(f, [el.tmp(el.elem(a, idx), "x") if isinstance(a, ArrayVal) else a for a in adapted])

    return build_map(el, shape, at, "map")
  if isinstance(ax, tuple):
    raise el.error(StaticValueError, "map() takes a single axis (or None)")
  n = None
  prepared = []
  for v in vals:
    if isinstance(v, RaggedArray):
      if ax != 0:
        raise el.error(UnsupportedError, "map() over a RaggedArray only supports axis=0")
      dim = el._ragged_leading_len(v)
      if n is None:
        n = dim
      elif not el.dim_equal(n, dim):
        el.check(cmp("eq", n, dim), "ValueError", "arguments of map have different lengths ({0} and {1})", n, dim)
      prepared.append((v, "ragged"))
    elif isinstance(v, ArrayVal) and v.rank > 0:
      k = norm_axis(el, ax, v.rank)
      v = _prepare(el, v)
      dim = v.shape[k]
      if n is None:
        n = dim
      elif not el.dim_equal(n, dim):
        el.check(cmp("eq", n, dim), "ValueError", "arguments of map have different lengths ({0} and {1})", n, dim)
      prepared.append((v, k))
    elif isinstance(v, ArrayVal):
      prepared.append((el.as_scalar(v), None))
    else:
      prepared.append((v, None))

  def at(idx):
    return el.call(
      f,
      [
        v
        if k is None
        else (el._ragged_getitem(v, idx[0]) if k == "ragged" else el.axis_view(v, k, idx[0], consumed=True))
        for v, k in prepared
      ],
    )

  return build_map(el, (n,), at, "map")


each = map
_MAP = map  # the adverbs below take a keyword argument called ``map``


@libfunction("allpairs")
def allpairs(el, f, x, y, axis=0):
  """``result[i, j] = f(x[i], y[j])`` for all pairs of elements along ``axis``."""
  return outer_map.impl(el, f, x, y, axis=axis)


@libfunction("outer_map")
def outer_map(el, f, *args, axis=0, axes=None):
  """Apply ``f`` to every combination of elements of the argument arrays."""
  if axes is not None:
    axis = axes
  ax = _axis_arg(el, axis)
  vals = [el.as_array(_arrayish(el, a)) for a in args]
  if not vals:
    raise el.error(TypeInferenceError, "outer_map() needs at least one array argument")
  if not isinstance(ax, tuple):
    ax = (ax,) * len(vals)
  if len(ax) != len(vals):
    raise el.error(StaticValueError, f"outer_map() got {len(vals)} arrays but {len(ax)} axes")
  prepared = []
  shape = []
  for v, a in zip(vals, ax):
    if v.rank == 0:
      raise el.error(TypeInferenceError, "outer_map() arguments must be at least 1-dimensional")
    if a is None:
      v = _prepare(el, el.flat_view(v))
      k = 0
    else:
      k = norm_axis(el, a, v.rank)
      v = _prepare(el, v)
    prepared.append((v, k))
    shape.append(v.shape[k])

  def at(idx):
    return el.call(f, [el.axis_view(v, k, i, consumed=True) for (v, k), i in zip(prepared, idx)])

  return build_map(el, tuple(shape), at, "allpairs")


def _as_combiner_ufunc(f):
  """Map polymorphic DSL combiners (e.g. ``dsl.min`` / ``dsl.max``) to their binary UFunc."""
  if f is _DSL_MIN:
    return prims.minimum
  if f is _DSL_MAX:
    return prims.maximum
  return f


def _reduce(el, f, x, init, axis, what: str):
  f = _as_combiner_ufunc(f)
  x = _arrayish(el, x)
  if not isinstance(x, ArrayVal):
    raise el.error(TypeInferenceError, f"{what}() expects an array, got {describe(x)}")
  if isinstance(init, ArrayVal) and init.rank == 0:
    init = el.as_scalar(init)

  def combine(a, b):
    return el.call(f, [a, b])

  ax = _axis_arg(el, axis)
  if isinstance(ax, tuple):
    raise el.error(StaticValueError, f"{what}() takes a single axis (or None)")
  tape = el.tape
  if tape is not None and not tape.muted and isinstance(f, prims.UFunc) and f.name in ("add", "mul", "min", "max"):
    if (init is None or isinstance(init, Expr)) and (
      tape.is_active(el, x) or (init is not None and tape.is_active(el, init))
    ):
      # the reductions that have a derivative rule
      from .numpy_lib import _reduce_prim

      return _reduce_prim(el, f.name, x, ax, init, what)
  if tape is not None and not tape.muted and (init is None or isinstance(init, Expr)):
    if (ax is None or x.rank == 1) and x.rank > 0 and (tape.is_active(el, x) or tape.is_active(el, init)):
      # any other function: keep the running values and replay the fold backwards
      return tape.fold(el, combine, x, init, what)
  if ax is None and not isinstance(init, ArrayVal):
    if x.rank == 0:
      v = el.as_scalar(x)
      return v if init is None else combine(init, v)
    x = _prepare(el, x)
    return reduce_values(el, x.shape, lambda idx: el.tmp(el.elem(x, idx), "x"), combine, init, what)
  k = norm_axis(el, 0 if ax is None else ax, x.rank)
  if x.rank == 1 and not isinstance(init, ArrayVal):
    x = _prepare(el, x)
    return reduce_values(el, x.shape, lambda idx: el.tmp(el.elem(x, idx), "x"), combine, init, what)
  if isinstance(f, prims.UFunc) and (init is None or isinstance(init, Expr)):
    # element-wise combiner: reduce each 1-d lane independently (and in parallel)
    return along_axis(
            el, x, k,
            lambda lane: reduce_values(
                el, lane.shape, lambda idx: el.tmp(el.elem(lane, idx), "x"), combine, init, what
            ),
            what,
        )  # fmt: skip
  x = _prepare(el, x)
  return reduce_arrays(el, x.shape[k], lambda i: el.axis_view(x, k, i, consumed=True), combine, init, what)


def _zipped(el, xs: tuple, init, fn, axis, what: str):
  """The sequence an adverb folds over: one array, or ``fn`` mapped over
  several (never built: the map fuses into the fold).  A trailing scalar
  after a single array is the old positional ``init``.

  Returns ``(seq, init, fold_axis)``: when ``fn`` is mapped along an axis,
  :func:`map` stacks its per-slice results onto leading axis ``0``, so the
  subsequent fold must consume axis ``0`` rather than the original ``axis``."""
  xs = list(xs)
  if init is None and len(xs) == 2 and fn is None and isinstance(xs[1], Expr):
    init = xs.pop()
  if not xs:
    raise el.error(TypeInferenceError, f"{what}() needs at least one array argument")
  if fn is None:
    if len(xs) > 1:
      raise el.error(
        TypeInferenceError,
        f"{what}() over {len(xs)} arrays needs map=: a function that turns one element of each into the value to fold",
      )
    return xs[0], init, axis
  map_axis = 0 if axis is None and what != "reduce" else axis
  mapped = _MAP.impl(el, fn, *xs, axis=map_axis)
  fold_axis = None if map_axis is None else 0
  return mapped, init, fold_axis


MAX_FOLD_CARRIES = 64


def _select_tree(vals: list[Expr], idx: Expr, lo: int, hi: int) -> Expr:
  if hi - lo == 1:
    return vals[lo]
  mid = (lo + hi) // 2
  return ops.select(
    cmp("lt", idx, Const(mid, I)),
    _select_tree(vals, idx, lo, mid),
    _select_tree(vals, idx, mid, hi),
  )


def _packed_const_array(el, elt, cshape: tuple[int, ...], scalars: list[Expr]) -> ArrayVal:
  """Build a zero-Alloc delayed fixed-size ArrayVal backed by ``scalars``.

  Constant-index reads fold straight to the underlying register scalar; dynamic
  reads lower to a pure balanced ``select`` tree (no heap allocation).
  """
  shape = tuple(Const(d, I) for d in cshape)
  strides: list[int] = [1] * len(cshape)
  for k in range(len(cshape) - 2, -1, -1):
    strides[k] = strides[k + 1] * cshape[k + 1]

  def fn(idx):
    if all(isinstance(i, Const) for i in idx):
      flat = sum(int(i.value) * s for i, s in zip(idx, strides))
      if 0 <= flat < len(scalars):
        return scalars[flat]
    flat_e: Expr = ZERO
    for i, s in zip(idx, strides):
      term = i if s == 1 else ops.imul(i, Const(s, I))
      flat_e = term if flat_e is ZERO else ops.iadd(flat_e, term)
    return _select_tree(scalars, flat_e, 0, len(scalars))

  return el.delayed(elt, shape, fn, cost=0)


def _flatten_fold_state(el, v, what: str = "fold", expected_tree=None):
  """Flatten a tuple/NamedTuple of scalars and small fixed-size arrays into
  ``(schema_tree, flat_scalars)``.  When ``expected_tree`` is supplied (e.g.
  from ``init``), dynamic slices whose runtime shape equals the expected
  compile-time shape are runtime-checked and unpacked cleanly."""
  from ..syntax.records import record_type

  def walk(node, exp, out: list[Expr]):
    if isinstance(node, Expr):
      out.append(node)
      return ("scalar", node.type.strong)
    if isinstance(node, ArrayVal):
      if node.rank == 0:
        s = el.as_scalar(node)
        out.append(s)
        return ("scalar", s.type.strong)
      if all(isinstance(d, Const) for d in node.shape):
        cshape = tuple(int(d.value) for d in node.shape)
      elif exp is not None and exp[0] == "array" and len(exp[2]) == node.rank:
        cshape = exp[2]
        for d, cd in zip(node.shape, cshape):
          el.check(
            cmp("eq", d, Const(cd, I)),
            "ValueError",
            f"{what} array state dimension mismatch (expected {cd}, got {{0}})",
            d,
          )
      else:
        raise el.error(TypeInferenceError, f"{what} array state fields must have compile-time constant shapes")
      size = 1
      for d in cshape:
        size *= d
      if size <= 0 or len(out) + size > MAX_FOLD_CARRIES:
        raise el.error(
          TypeInferenceError,
          f"{what} state has too many scalar fields (max {MAX_FOLD_CARRIES})",
        )
      coords: list[tuple[int, ...]] = [()]
      for d in cshape:
        coords = [(*c, i) for c in coords for i in range(d)]
      for c in coords:
        out.append(el.tmp(el.elem(node, tuple(Const(i, I) for i in c)), "fv"))
      return ("array", node.elt.strong, cshape)
    if isinstance(node, tuple):
      rec = record_type(node)
      exp_children = (
        exp[3] if (exp is not None and exp[0] == "tuple" and len(exp[3]) == len(node)) else (None,) * len(node)
      )
      children = tuple(walk(item, ec, out) for item, ec in zip(node, exp_children))
      return ("tuple", rec, type(node), children)
    raise el.error(TypeInferenceError, f"{what} needs a nonempty tuple of scalar or fixed-size array fields")

  if not isinstance(v, tuple) or not v:
    raise el.error(TypeInferenceError, f"{what} needs a nonempty tuple of scalar fields")
  flat: list[Expr] = []
  tree = walk(v, expected_tree, flat)
  if not flat:
    raise el.error(TypeInferenceError, f"{what} needs a nonempty tuple of scalar fields")
  if len(flat) > MAX_FOLD_CARRIES:
    raise el.error(TypeInferenceError, f"{what} state exceeds maximum of {MAX_FOLD_CARRIES} scalar fields")
  return tree, flat


def _unflatten_fold_state(el, tree, flat: list[Expr]):
  cursor = [0]

  def build(node):
    kind = node[0]
    if kind == "scalar":
      v = flat[cursor[0]]
      cursor[0] += 1
      return v
    if kind == "array":
      _, elt, cshape = node
      size = 1
      for d in cshape:
        size *= d
      chunk = flat[cursor[0] : cursor[0] + size]
      cursor[0] += size
      return _packed_const_array(el, elt, cshape, chunk)
    _, rec, _cls, children = node
    items = [build(ch) for ch in children]
    return rec.build(items) if rec else tuple(items)

  return build(tree)


def _same_fold_schema(a, b) -> bool:
  if a[0] != b[0]:
    return False
  kind = a[0]
  if kind == "scalar":
    return a[1] is b[1]
  if kind == "array":
    return a[1] is b[1] and a[2] == b[2]
  return a[2] is b[2] and len(a[3]) == len(b[3]) and all(_same_fold_schema(x, y) for x, y in zip(a[3], b[3]))


def _flatten_emit_output(el, v):
  """Flatten an ``emit`` return value (scalar, fixed-size array, or tuple/NamedTuple thereof)
  into ``(out_tree, leaves)`` where each leaf is ``(elt, cshape, scalars_list)``."""
  from ..syntax.records import record_type

  leaves: list[tuple] = []

  def walk(node):
    if isinstance(node, Expr):
      idx = len(leaves)
      leaves.append((node.type.strong, (), [node]))
      return ("leaf", idx)
    if isinstance(node, ArrayVal):
      if node.rank == 0:
        s = el.as_scalar(node)
        idx = len(leaves)
        leaves.append((s.type.strong, (), [s]))
        return ("leaf", idx)
      if not all(isinstance(d, Const) for d in node.shape):
        raise el.error(TypeInferenceError, "fold projection arrays must have compile-time constant shapes")
      cshape = tuple(int(d.value) for d in node.shape)
      coords: list[tuple[int, ...]] = [()]
      for d in cshape:
        coords = [(*c, i) for c in coords for i in range(d)]
      vals = [el.elem(node, tuple(Const(i, I) for i in c)) for c in coords]
      idx = len(leaves)
      leaves.append((node.elt.strong, cshape, vals))
      return ("leaf", idx)
    if isinstance(node, tuple) and node:
      rec = record_type(node)
      return ("tuple", rec, tuple(walk(item) for item in node))
    raise el.error(
      TypeInferenceError, f"fold projection must return scalars or fixed-size arrays, got {describe(node)}"
    )

  tree = walk(v)
  return tree, leaves


def _rebuild_emit_output(tree, arrays: list[ArrayVal]):
  if tree[0] == "leaf":
    return arrays[tree[1]]
  _, rec, children = tree
  items = [_rebuild_emit_output(ch, arrays) for ch in children]
  return rec.build(items) if rec else tuple(items)


def _fold_impl(el, lift_cb, merge_cb, x, init=None, *, axis=None, indexed=False, emit_cb=None):
  """Internal implementation of :func:`fold` accepting host IR-builder callbacks."""
  from ..syntax import ir

  if el.tape is not None and not el.tape.muted:
    raise el.no_derivative("differentiation through a custom fold is not implemented")
  x = el.as_array(_arrayish(el, x))
  if x.rank == 0:
    raise el.error(TypeInferenceError, "fold needs an array with at least one dimension")
  indexed = indexed if isinstance(indexed, bool) else el.static_bool(indexed, "indexed")

  init_tree = None
  init_flat: list[Expr] | None = None
  if init is not None:
    init_tree, init_flat = _flatten_fold_state(el, init, "fold")

  def build_lane_state(arr):
    n = arr.shape[0]
    idx = ir.Var("i", I)
    el.nonneg.add(idx.uid)
    el.upper[idx.uid] = n
    home = el.scope
    scope = el.new_scope("loop")
    parallel = el.can_parallel((n,))
    body, combine = [], []
    with el.in_block(body, scope, loop=True, par=parallel):
      element = el.axis_view(arr, 0, idx, consumed=True)
      val_raw = lift_cb(element, idx) if indexed else lift_cb(element)
      val_tree, values = _flatten_fold_state(el, val_raw, "fold", expected_tree=init_tree)

    schema_tree = init_tree if init_tree is not None else val_tree
    schema_flat = init_flat if init_flat is not None else values
    if not _same_fold_schema(val_tree, schema_tree):
      raise el.error(TypeInferenceError, "fold lift and merge must preserve the state structure and dtypes")

    left_flat = [ir.Var("left", v.type.strong) for v in schema_flat]
    right_flat = [ir.Var("right", v.type.strong) for v in schema_flat]
    with el.in_block(combine, loop=True):
      left_obj = _unflatten_fold_state(el, schema_tree, left_flat)
      right_obj = _unflatten_fold_state(el, schema_tree, right_flat)
      merged_raw = merge_cb(left_obj, right_obj)
      merged_tree, merged = _flatten_fold_state(el, merged_raw, "fold", expected_tree=schema_tree)
    if not _same_fold_schema(merged_tree, schema_tree):
      raise el.error(TypeInferenceError, "fold lift and merge must preserve the state structure and dtypes")
    if any(isinstance(s, ir.Alloc) for s in ir.walk_stmts(combine)) or any(
      isinstance(s, (ir.Store, ir.IndexScatter))
      or any(isinstance(e, ir.RtCall) and not e.pure for expr in ir.stmt_exprs(s) for e in ir.walk_expr(expr))
      for s in ir.walk_stmts([*body, *combine])
    ):
      raise el.error(UnsupportedError, "fold lift and merge must be pure scalar computations")

    res_flat = [ir.Var("field", v.type.strong) for v in schema_flat]
    carries = [ir.Carry(*parts) for parts in list(zip(res_flat, values, left_flat, right_flat, merged))[1:]]
    empty = ir.ErrorInfo("ValueError", "cannot fold an empty lane without an initial state", el.loc())
    reduction = ir.Fold(
      res_flat[0],
      values[0],
      left_flat[0],
      right_flat[0],
      combine,
      merged[0],
      op="structured",
      empty=empty,
      carries=carries,
    )
    el.emit(ir.IndexReduce((n,), (idx,), body, [reduction], scope, home, parallel, el.loc()))
    results = _unflatten_fold_state(el, schema_tree, res_flat)
    if init is None:
      return results
    seeded_raw = merge_cb(_unflatten_fold_state(el, schema_tree, init_flat), results)
    seeded_tree, seeded_flat = _flatten_fold_state(el, seeded_raw, "fold", expected_tree=schema_tree)
    if not _same_fold_schema(seeded_tree, schema_tree):
      raise el.error(TypeInferenceError, "fold lift and merge must preserve the state structure and dtypes")
    return _unflatten_fold_state(el, schema_tree, seeded_flat)

  def project_raw(summary):
    if emit_cb is None:
      return summary
    return emit_cb(summary)

  k = None if axis is None else norm_axis(el, axis, x.rank)
  if k is not None and x.rank > 1 and emit_cb is None:
    raise el.error(TypeInferenceError, "fold along an axis needs emit= to project each state to an output")

  if k is None or x.rank == 1:
    # Single-lane fold
    def single_lane():
      st = build_lane_state(_prepare(el, x))
      return project_raw(st) if emit_cb is not None else st

    if init is None:
      return single_lane()
    nonempty, empty = [], []
    with el.in_block(nonempty):
      a = single_lane()
    with el.in_block(empty):
      b = project_raw(init) if emit_cb is not None else init
    result = el.merge_values("fold", a, nonempty, b, empty)
    el.emit(ir.If(cmp("gt", x.shape[0], ZERO), nonempty, empty))
    return result

  # Multi-lane fold along axis k: stream delayed `x` directly (like along_axis(stream=True))
  # so pairwise score producers land inside the lane IndexReduce with zero O(N*M) allocation!
  lead = x.shape[:k] + x.shape[k + 1 :]
  with el.trial():
    probe_idx = tuple(ir.Var("p", I) for _ in lead)
    lane_view = el.remap(
      x,
      (x.shape[k],),
      [(None, i, ZERO) for i in probe_idx[:k]] + [(0, ZERO, ops.ONE)] + [(None, i, ZERO) for i in probe_idx[k:]],
      consumed=True,
    )
    probe_st = build_lane_state(lane_view)
    probe_out = project_raw(probe_st)
    is_scalar_emit = isinstance(probe_out, Expr) or (isinstance(probe_out, ArrayVal) and probe_out.rank == 0)
    if not is_scalar_emit:
      out_tree, probe_leaves = _flatten_emit_output(el, probe_out)

  if is_scalar_emit:

    def scalar_lane(arr):
      return el.as_scalar(project_raw(build_lane_state(arr)), "fold projection")

    if init is None:
      el.check(cmp("gt", x.shape[k], ZERO), "ValueError", "cannot fold an empty lane without an initial state")
      return along_axis(el, x, k, scalar_lane, "fold", stream=True)
    nonempty, empty = [], []
    with el.in_block(nonempty):
      a = along_axis(el, x, k, scalar_lane, "fold", stream=True)
    with el.in_block(empty):
      b = build_map(el, lead, lambda _: el.as_scalar(project_raw(init), "fold projection"), "fold")
    result = el.merge_values("fold", a, nonempty, b, empty)
    el.emit(ir.If(cmp("gt", x.shape[k], ZERO), nonempty, empty))
    return result

  # Multi-output or fixed-size-array emit: allocate all output arrays once and fill in ONE IndexMap
  # so pairwise_fold.py can tile all outputs simultaneously on CPU & GPU!
  out_arrays: list[ArrayVal] = []
  for elt, cshape, _ in probe_leaves:
    full_shape = lead + tuple(Const(d, I) for d in cshape)
    out_arrays.append(el.alloc_array(elt, full_shape, label="fold"))

  def emit_stores(idx, proj_val):
    _, leaves = _flatten_emit_output(el, proj_val)
    lead_flat: Expr = ZERO
    for coord, dim in zip(idx, lead):
      lead_flat = coord if lead_flat is ZERO else ops.iadd(ops.imul(lead_flat, dim), coord)
    for arr, (elt, _cshape, scalars) in zip(out_arrays, leaves):
      n_elem = len(scalars)
      base = lead_flat if n_elem == 1 else el.tmp(ops.imul(lead_flat, Const(n_elem, I)), "fbase")
      for off, sv in enumerate(scalars):
        pos = base if off == 0 else ops.iadd(base, Const(off, I))
        el.emit(ir.Store(arr.buf, pos, cast(sv, elt)))

  def run_nonempty():
    el.consume(x)

    def body(idx):
      lane_view = el.remap(
        x,
        (x.shape[k],),
        [(None, i, ZERO) for i in idx[:k]] + [(0, ZERO, ops.ONE)] + [(None, i, ZERO) for i in idx[k:]],
        consumed=True,
      )
      emit_stores(idx, project_raw(build_lane_state(lane_view)))

    el.index_map(lead, body, ordered=False)

  if init is None:
    el.check(cmp("gt", x.shape[k], ZERO), "ValueError", "cannot fold an empty lane without an initial state")
    run_nonempty()
  else:
    nonempty, empty = [], []
    with el.in_block(nonempty):
      run_nonempty()
    with el.in_block(empty):
      el.index_map(lead, lambda idx: emit_stores(idx, project_raw(init)), ordered=False)
    el.emit(ir.If(cmp("gt", x.shape[k], ZERO), nonempty, empty))

  for arr in out_arrays:
    el.record_store(arr)
  return _rebuild_emit_output(out_tree, out_arrays)


@libfunction("fold")
def fold(el, lift, merge, x, init=None, *, axis=None, indexed=False, emit=None):
  """Lift elements to scalar/fixed-vector tuple states and merge adjacent states in input order.

  With no axis, lift each leading slice and return one state. With an axis,
  fold the lanes along that axis and project each state with `emit` (which may
  return a scalar, a fixed-size array, or a tuple/NamedTuple of them).
  An indexed lift receives (value, index) with the index local to its lane.
  A supplied seed is applied once on the left; an unseeded empty fold raises.
  """
  indexed_b = indexed if isinstance(indexed, bool) else el.static_bool(indexed, "indexed")

  def merge_cb(a, b):
    return el.call(merge, [a, b])

  lift_cb = (lambda elem, idx: el.call(lift, [elem, idx])) if indexed_b else (lambda elem: el.call(lift, [elem]))
  emit_cb = None if emit is None else (lambda s: el.call(emit, [s]))
  return _fold_impl(el, lift_cb, merge_cb, x, init=init, axis=axis, indexed=indexed_b, emit_cb=emit_cb)


@libfunction("reduce")
def reduce(el, f, *xs, init=None, axis=None, map=None):  # noqa: A001, A002
  """Fold ``f`` over the elements of an array (all of them, or slices along ``axis``).

  With ``map=g`` and several arrays, the fold is over ``g(x1[i], ..., xn[i])``:
  ``reduce(add, x, y, map=multiply)`` is a dot product, and no array of
  products is ever built.

  ``f`` should be associative: recognised primitives (add, multiply, min,
  max, logical/bitwise and/or) are reduced in parallel."""
  x, init, fold_axis = _zipped(el, xs, init, map, axis, "reduce")
  return _reduce(el, f, x, init, fold_axis, "reduce")


@core.register(functools.reduce)
def _functools_reduce(el, f, x, initial=None):  # noqa: D401
  x = _arrayish(el, x)
  if isinstance(x, ArrayVal):
    return _reduce(el, f, x, initial, 0, "reduce")
  raise el.error(TypeInferenceError, f"reduce() expects an array or tuple, got {describe(x)}")


@libfunction("scan")
def scan(el, f, *xs, init=None, axis=0, map=None, emit=None):  # noqa: A002
  """Running fold: ``result[i] = f(result[i - 1], x[i])``.

  ``map=g`` folds over ``g(x1[i], ..., xn[i])`` for several arrays, and
  ``emit=h`` gives ``h`` of each running value instead of the value itself."""
  x, init, fold_axis = _zipped(el, xs, init, map, axis, "scan")
  out = _scan(el, f, x, init, fold_axis)
  return out if emit is None else _MAP.impl(el, emit, out, axis=None)


def _scan(el, f, x, init, axis):
  x = el.as_array(_arrayish(el, x))
  ax = _axis_arg(el, axis)
  tape = el.tape
  if tape is not None and not tape.muted and isinstance(f, prims.UFunc) and f.name == "add" and init is None:
    if tape.is_active(el, x):
      from .numpy_lib import _cumulative

      return _cumulative(el, x, None if ax is None else Const(ax, I), "add", None)
  if isinstance(init, ArrayVal) and init.rank == 0:
    init = el.as_scalar(init)
  if init is not None and not isinstance(init, Expr):
    raise el.error(UnsupportedError, "scan() with an array-valued 'init' is not supported")

  def combine(a, b):
    return el.call(f, [a, b])

  if ax is None or x.rank == 1:
    if x.rank != 1:
      x = el.flat_view(x)
    x = _prepare(el, x)
    if (
      tape is not None and not tape.muted and (tape.is_active(el, x) or (init is not None and tape.is_active(el, init)))
    ):
      from ..transforms.ad_loops import differentiate_scan

      return differentiate_scan(tape, el, combine, x, init, "scan")
    return scan_values(el, x.shape[0], lambda i: el.tmp(el.elem(x, (i,)), "x"), combine, init)
  return scan_axis(el, x, norm_axis(el, ax, x.rank), combine, init)


def scan_axis(el, x: ArrayVal, k: int, combine, init=None) -> ArrayVal:
  """Running fold along axis ``k`` of a multi-dimensional array: a map of
  :class:`~parakeet.syntax.ir.IndexScan` statements."""
  from ..ndtypes import join
  from ..syntax.ir import Fold, IndexScan, Store, Var

  x = _prepare(el, x)
  # accumulator type
  acc_type = init.type if init is not None else x.elt
  el.loop_depth += 1
  try:
    for _ in range(6):
      with el.trial():
        r = combine(Var("acc", acc_type), Var("x", x.elt))
        if not isinstance(r, Expr):
          raise el.error(TypeInferenceError, "the function passed to scan must return a scalar")
        r_type = r.type
      new = join(acc_type, r_type)
      if new is acc_type:
        break
      acc_type = new
  finally:
    el.loop_depth -= 1
  out = el.alloc_array(acc_type.strong, x.shape, label="scan")
  lead = x.shape[:k] + x.shape[k + 1 :]
  n = x.shape[k]

  def lane(idx):
    full = lambda i: tuple(idx[:k]) + (i,) + tuple(idx[k:])  # noqa: E731
    index = Var("i", I)
    el.nonneg.add(index.uid)
    el.upper[index.uid] = n
    body: list = []
    scope = el.new_scope("loop")
    with el.in_block(body, scope, loop=True):
      value = el.tmp(el.as_scalar(el.elem(x, full(index)), "element"), "x")
    a, b = Var("acc", acc_type), Var("x", value.type)
    combine_block: list = []
    with el.in_block(combine_block, loop=True):
      combined = ops.cast(el.as_scalar(combine(a, b), "scan result"), acc_type)
    result = Var("run", acc_type)
    fold = Fold(result, value, a, b, combine_block, combined, None if init is None else ops.cast(init, acc_type))
    emit = [Store(out.buf, el.index_expr(out, full(index)), ops.cast(result, acc_type.strong))]
    el.emit(IndexScan(n, index, body, fold, emit, scope, False, el.loc()))

  el.loop_nest(lead, lane)
  return out


def _accepts_unpacked_index(f, ndim: int) -> bool:
  """True when ``f`` should receive ``(i, j, ...)`` as ``ndim`` separate arguments
  rather than a single index tuple ``ij``."""
  import inspect

  from ..syntax.values import Closure, Partial, PyFunc

  if ndim < 2:
    return False
  if isinstance(f, prims.UFunc):
    return f.nin == ndim
  if isinstance(f, LibFunction) and f.name in ("min", "max"):
    return ndim == 2
  if isinstance(f, Partial):
    return _accepts_unpacked_index(f.fn, ndim + len(f.args))
  if isinstance(f, Closure):
    a = f.node.args
    if a.vararg is not None:
      return False
    n_pos = len(a.posonlyargs) + len(a.args)
    n_req = n_pos - len(a.defaults)
    return n_req <= ndim <= n_pos and not (n_req <= 1 <= n_pos)
  fn_obj = f.fn if isinstance(f, PyFunc) else (f if callable(f) else None)
  if fn_obj is not None:
    try:
      sig = inspect.signature(fn_obj)
    except (TypeError, ValueError):
      return False
    n_pos = n_req = 0
    for p in sig.parameters.values():
      if p.kind == inspect.Parameter.VAR_POSITIONAL:
        return False
      if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
        n_pos += 1
        if p.default is inspect.Parameter.empty:
          n_req += 1
    return n_req <= ndim <= n_pos and not (n_req <= 1 <= n_pos)
  return False


def _index_args(f, idx, scalar_index: bool) -> list:
  if scalar_index:
    return [idx[0]]
  if _accepts_unpacked_index(f, len(idx)):
    return list(idx)
  return [tuple(idx)]


@libfunction("imap")
def imap(el, f, shape):
  """Array of the given ``shape`` with ``result[i, j, ...] = f(i, j, ...)``.

  When ``shape`` is a tuple of length ``r >= 2``, ``f`` may take either ``r``
  index arguments (``lambda i, j: ...``) or a single index tuple (``lambda ij: ...``)."""
  scalar_index = isinstance(shape, Expr) or (isinstance(shape, ArrayVal) and shape.rank == 0)
  dims = el.check_dims(el.shape_tuple(shape, "the shape passed to imap"))
  return build_map(el, dims, lambda idx: el.call(f, _index_args(f, idx, scalar_index)), "imap")


@libfunction("ireduce")
def ireduce(el, f, shape, init=None, combine=None):
  """Reduce ``f(i, ...)`` over every index of ``shape`` with ``combine``
  (defaults to addition).

  When ``shape`` is a tuple of length ``r >= 2``, ``f`` may take either ``r``
  index arguments (``lambda i, j: ...``) or a single index tuple."""
  scalar_index = isinstance(shape, Expr) or (isinstance(shape, ArrayVal) and shape.rank == 0)
  dims = el.check_dims(el.shape_tuple(shape, "the shape passed to ireduce"))
  comb = _as_combiner_ufunc(combine) if combine is not None else prims.UFUNCS["add"]
  return reduce_values(
        el, dims, lambda idx: el.as_scalar(el.call(f, _index_args(f, idx, scalar_index)), "ireduce element"),
        lambda a, b: el.call(comb, [a, b]), init, "ireduce",
    )  # fmt: skip


@libfunction("parfor")
def parfor(el, f, bounds):
  """Call ``f(i, ...)`` for every index below ``bounds``, in parallel, for its
  side effects (typically writes to a preallocated output array)."""
  scalar_index = isinstance(bounds, Expr) or (isinstance(bounds, ArrayVal) and bounds.rank == 0)
  dims = el.check_dims(el.shape_tuple(bounds, "the bounds passed to parfor"))

  def body(idx):
    el.call(f, _index_args(f, idx, scalar_index))

  el.loop_nest(dims, body)
  return None


@libfunction("filter")
def filter(el, pred, *xs, map=None):  # noqa: A001, A002
  """Elements of ``x`` (along its first axis) for which ``pred`` is true.

  Given several arrays, ``pred`` sees one element of each and ``map=g``
  says what to keep: ``g(x1[i], ..., xn[i])`` where the predicate holds."""
  if not xs:
    raise el.error(TypeInferenceError, "filter() needs at least one array argument")
  if len(xs) > 1 and map is None:
    raise el.error(
      TypeInferenceError, f"filter() over {len(xs)} arrays needs map=: what to keep of each selected element"
    )
  xs = [el.as_array(_arrayish(el, x)) for x in xs]
  xs = [el.as_concrete(x) if x.buf is None and x.cost > CHEAP_COST else x for x in xs]
  mask = _MAP.impl(el, pred, *xs, axis=0)
  if not isinstance(mask, ArrayVal) or mask.rank != 1:
    raise el.error(TypeInferenceError, "the predicate passed to filter must return a scalar")
  mask = el.cast_array(mask, _bool())
  idx = el.nonzero1d(mask)
  if map is None:
    return el.getitem(xs[0], idx)
  return _MAP.impl(el, map, *[el.getitem(x, idx) for x in xs], axis=0)


@libfunction("filter_reduce")
def filter_reduce(el, f, pred, *xs, map=None, init=None, axis=None):  # noqa: A002
  """Reduce with ``f`` over the elements of ``xs`` where ``pred`` holds.

  Equivalent to ``reduce(f, filter(pred, *xs, map=map), init=init, axis=axis)``
  and fused by the middle end into a single guarded reduction loop with no
  intermediate index array when ``init`` is given."""
  if not xs:
    raise el.error(TypeInferenceError, "filter_reduce() needs at least one array argument")
  selected = filter.impl(el, pred, *xs, map=map)
  return reduce.impl(el, f, selected, init=init, axis=axis)


def _scatter_add_impl(
  el,
  n,
  idx,
  values,
  *,
  where=None,
  ignore_oob: bool = False,
  out_elt=None,
):
  """Shared implementation for ``scatter_add``, keyed generator scatters, and
  inverted filtered comprehensions."""
  from .numpy_lib import _arr, _emit_scatter_add

  size = cast(el.as_scalar(n, "n"), I)
  el.check(cmp("ge", size, ZERO), "ValueError", "scatter_add() length must not be negative")
  idx_arr = _arr(el, idx)
  if idx_arr.rank != 1:
    raise el.error(TypeInferenceError, "scatter_add() expects 1-d idx array")
  if idx_arr.elt.kind not in "iu":
    raise el.error(TypeInferenceError, f"scatter_add() idx must be integer, got {idx_arr.elt}")
  val_arr = _arr(el, values) if values is not None else None
  if val_arr is not None:
    if val_arr.rank != 1:
      raise el.error(TypeInferenceError, "scatter_add() expects 1-d idx and values arrays")
    if not (
      (
        isinstance(idx_arr.shape[0], Var)
        and isinstance(val_arr.shape[0], Var)
        and idx_arr.shape[0].uid == val_arr.shape[0].uid
      )
      or (
        isinstance(idx_arr.shape[0], Const)
        and isinstance(val_arr.shape[0], Const)
        and idx_arr.shape[0].value == val_arr.shape[0].value
      )
    ):
      el.check(
        cmp("eq", idx_arr.shape[0], val_arr.shape[0]), "ValueError", "scatter_add() idx and values lengths differ"
      )
  where_arr = _arr(el, where) if where is not None else None
  if where_arr is not None:
    if where_arr.rank != 1:
      raise el.error(TypeInferenceError, "scatter_add() where mask must be 1-d")
    if not (
      (
        isinstance(idx_arr.shape[0], Var)
        and isinstance(where_arr.shape[0], Var)
        and idx_arr.shape[0].uid == where_arr.shape[0].uid
      )
      or (
        isinstance(idx_arr.shape[0], Const)
        and isinstance(where_arr.shape[0], Const)
        and idx_arr.shape[0].value == where_arr.shape[0].value
      )
    ):
      el.check(
        cmp("eq", idx_arr.shape[0], where_arr.shape[0]), "ValueError", "scatter_add() idx and where lengths differ"
      )

  elt = out_elt if out_elt is not None else (val_arr.elt.strong if val_arr is not None else Int64)
  tape = el.tape
  if tape is not None and not tape.muted and val_arr is not None and tape.is_active(el, val_arr):
    from ..transforms.autodiff import Node

    with tape.computing(el):
      out = _emit_scatter_add(
        el, size, idx_arr, val_arr, elt, check_bounds=not ignore_oob, where=where_arr, ignore_oob=ignore_oob
      )
    node = Node("scatter_add", (val_arr,), (idx_arr, where_arr, ignore_oob))
    tape.record(el, node)
    tape.register(el, node, out)
    return out
  return _emit_scatter_add(
    el, size, idx_arr, val_arr, elt, check_bounds=not ignore_oob, where=where_arr, ignore_oob=ignore_oob
  )


class _ScatterAddFunction(LibFunction):
  def __call__(self, *args, **kwargs):
    # Support 2-arg Python generator/iterable form outside @par.jit:
    # par.scatter_add(n, ((idx[i], vals[i]) for i in range(N) if ...))
    if len(args) == 2 and "values" not in kwargs and not isinstance(args[1], np.ndarray):
      n = int(args[0])
      if n < 0:
        raise ValueError("scatter_add() length must not be negative")
      pairs = list(args[1])
      if not pairs:
        return np.zeros(n, dtype=np.float64)
      ks = np.asarray([p[0] for p in pairs], dtype=np.int64)
      vs = np.asarray([p[1] for p in pairs])
      return super().__call__(n, ks, vs)
    return super().__call__(*args, **kwargs)


def _scatter_add_entry(el, n, idx, values, where=None):
  """Generalized ``ReduceByIndex`` scatter-add: return a 1-d array of length
  ``n`` where ``out[idx[i]] += values[i]`` for every ``i`` (optionally guarded
  by ``where``).

  Also accepts a keyed generator expression inside ``@par.jit`` or Python::

      par.scatter_add(n, ((idx[i], vals[i]) for i in range(len(idx)) if mask[i]))

  Preserves ``values.dtype`` (supporting ``float32`` on Metal/CUDA as well as
  ``float64``/``int64``), lowers to parallel privatized buffers via
  :class:`~parakeet.syntax.ir.IndexScatter`, and differentiates via gather."""
  return _scatter_add_impl(el, n, idx, values, where=where)


scatter_add = _ScatterAddFunction("scatter_add", _scatter_add_entry)


class FixedList:
  """Python fallback for ``par.fixed_list(capacity, dtype)`` when executed outside ``@par.jit``."""

  __slots__ = ("_data", "_cap", "_dtype")

  def __init__(self, capacity: int, dtype=np.float32):
    if capacity < 1:
      raise ValueError("fixed_list() capacity must be >= 1")
    self._cap = int(capacity)
    self._dtype = np.dtype(dtype)
    self._data: list = []

  def append(self, value) -> None:
    if len(self._data) >= self._cap:
      raise IndexError(f"fixed_list capacity {self._cap} exceeded")
    self._data.append(self._dtype.type(value))

  def pop(self):
    if not self._data:
      raise IndexError("pop from empty fixed_list")
    return self._data.pop()

  def clear(self) -> None:
    self._data.clear()

  def __len__(self) -> int:
    return len(self._data)

  def __getitem__(self, i: int):
    return self._data[i]

  def __setitem__(self, i: int, value) -> None:
    self._data[i] = self._dtype.type(value)

  def __iter__(self):
    return iter(self._data)

  def __array__(self, dtype=None):
    return np.asarray(self._data, dtype=dtype or self._dtype)


class _FixedListFunction(LibFunction):
  def __call__(self, capacity: int, dtype=np.float32):
    return FixedList(capacity, dtype)


def _fixed_list_entry(el, capacity, dtype=np.float32):
  """Create a fixed-capacity thread-local list/stack backed by a stack-allocated
  buffer (zero heap ``malloc`` — native on CPU, CUDA, and Metal).

  Supports ``.append(v)``, ``.pop()``, ``.clear()``, ``len(fl)``, ``fl[i]``,
  ``fl[i] = v``, and ``for x in fl:`` inside ``@par.jit``, ``par.parfor``, and
  ``par.map`` kernels."""
  from ..ndtypes import BufType
  from ..syntax.ir import Alloc, Assign
  from ..syntax.values import FixedListVal

  cap = capacity if isinstance(capacity, int) else el.static_int(capacity, "fixed_list capacity")
  if cap < 1:
    raise el.error(StaticValueError, "fixed_list() capacity must be >= 1")
  elt = el.as_dtype(dtype, "dtype").strong
  buf = Var("fl_buf", BufType(elt))
  length = Var("fl_len", I)
  el.mutables.add(length.uid)
  el.nonneg.add(length.uid)
  el.emit(Alloc(buf, Const(cap, I), zero=False, stack=True))
  el.emit(Assign(length, ZERO))
  return FixedListVal(buf, length, cap, elt)


fixed_list = _FixedListFunction("fixed_list", _fixed_list_entry)


def _bool():
  from ..ndtypes import Bool

  return Bool


# ---------------------------------------------------------------------------
# grad: the adverb that turns a function into its derivative
# ---------------------------------------------------------------------------


@libfunction("gradient")
def _gradient(el, f, *args, argnum=0, **kwargs):
  from ..transforms import autodiff

  return autodiff.differentiate(el, f, argnum, list(args), kwargs, False, "grad")


@libfunction("value_and_gradient")
def _value_and_gradient(el, f, *args, argnum=0, **kwargs):
  from ..transforms import autodiff

  return autodiff.differentiate(el, f, argnum, list(args), kwargs, True, "value_and_grad")


class Gradient:
  """What :func:`grad` returns: call it like the function it differentiates."""

  def __init__(self, fn, argnum, impl: LibFunction):
    if not callable(fn):
      raise TypeError(f"{impl.__name__.replace('ient', '')} expects a function, got {type(fn).__name__}")
    nums = argnum if isinstance(argnum, tuple) else (argnum,)
    if not nums or not all(isinstance(k, int) and not isinstance(k, bool) and k >= 0 for k in nums):
      raise TypeError("argnum must be a non-negative integer or a tuple of them")
    self.fn = fn
    self.argnum = argnum
    self.impl = impl
    self.__name__ = getattr(fn, "__name__", "function")
    self.__doc__ = getattr(fn, "__doc__", None)

  def __repr__(self) -> str:
    return f"<parakeet {self.impl.__name__} of {self.__name__}>"

  def __call__(self, *args, **kwargs):
    from ..api import call_staged

    return call_staged(self.impl, (self.fn, *args), {**kwargs, "argnum": self.argnum})

  def __parakeet_value__(self, el):
    from ..syntax.values import Partial

    return Partial(self.impl, (el.lift(self.fn),), {"argnum": el.lift(self.argnum)})


def grad(f, argnum=0):
  """The gradient of the scalar-valued function ``f`` with respect to its
  argument number ``argnum`` (or, for a tuple, to each of those arguments).

  ``grad(f)(*args)`` takes the arguments of ``f`` and returns an array shaped
  like the differentiated argument (a tuple of arrays for a tuple of
  argument numbers; nested tuples of parameters give nested tuples of
  gradients).  Works inside compiled code and from plain Python."""
  return Gradient(f, argnum, _gradient)


def value_and_grad(f, argnum=0):
  """Like :func:`grad`, but the new function returns ``(f(*args), gradient)``."""
  return Gradient(f, argnum, _value_and_gradient)


@core.register(grad)
def _grad_staged(el, f, argnum=0):
  from ..syntax.values import Partial

  return Partial(_gradient, (f,), {"argnum": argnum})


@core.register(value_and_grad)
def _value_and_grad_staged(el, f, argnum=0):
  from ..syntax.values import Partial

  return Partial(_value_and_gradient, (f,), {"argnum": argnum})


def _make_minmax(name: str, ufunc, np_reduce, np_binary):
  """Build polymorphic ``dsl.min`` / ``dsl.max`` supporting both binary ``f(a, b)``
  (including ``combine=dsl.min`` / ``reduce(dsl.min, x)``) and array reduction
  ``f(x, axis=..., keepdims=...)``."""

  def impl(el, x, y=None, *, axis=None, keepdims=False):
    if y is not None:
      return core.call_ufunc(el, ufunc, x, y)
    return core.FUNCTIONS[np_reduce](el, x, axis=axis, keepdims=keepdims)

  class _MinMaxOp(LibFunction):
    def __call__(self, x, y=None, *, axis=None, keepdims=False):
      if y is not None:
        return np_binary(x, y)
      return np_reduce(x, axis=axis, keepdims=keepdims)

  return _MinMaxOp(
    name, impl, doc=f"Elementwise binary ``{name}(a, b)`` or array reduction ``{name}(x, axis=..., keepdims=...)``."
  )


import numpy as _np_adverbs  # noqa: E402

_DSL_MIN = _make_minmax("min", prims.minimum, _np_adverbs.min, _np_adverbs.minimum)
_DSL_MAX = _make_minmax("max", prims.maximum, _np_adverbs.max, _np_adverbs.maximum)


def _popcount_u64(el, x: Expr) -> Expr:
  """Exact branchless 64-bit SWAR popcount (works on C, multicore, CUDA, and Metal)."""
  from ..ndtypes import UInt64

  u = cast(x, UInt64)
  m1 = Const(0x5555555555555555, UInt64)
  m2 = Const(0x3333333333333333, UInt64)
  m4 = Const(0x0F0F0F0F0F0F0F0F, UInt64)
  h01 = Const(0x0101010101010101, UInt64)
  u = el.tmp(
    ops.prim(
      "sub",
      (u, ops.prim("bitand", (ops.prim("rshift", (u, Const(1, UInt64)), UInt64), m1), UInt64)),
      UInt64,
    ),
    "pc1",
  )
  u = el.tmp(
    ops.prim(
      "add",
      (
        ops.prim("bitand", (u, m2), UInt64),
        ops.prim("bitand", (ops.prim("rshift", (u, Const(2, UInt64)), UInt64), m2), UInt64),
      ),
      UInt64,
    ),
    "pc2",
  )
  u = el.tmp(
    ops.prim(
      "bitand",
      (ops.prim("add", (u, ops.prim("rshift", (u, Const(4, UInt64)), UInt64)), UInt64), m4),
      UInt64,
    ),
    "pc4",
  )
  return ops.prim("rshift", (ops.prim("mul", (u, h01), UInt64), Const(56, UInt64)), UInt64)


def _topk_sentinel(elt, largest: bool) -> Const:
  import numpy as np

  if elt.kind == "f":
    return Const(float("-inf") if largest else float("inf"), elt)
  if elt.kind == "b":
    return Const(False if largest else True, elt)
  info = np.iinfo(elt.strong.name)
  return Const(int(info.min if largest else info.max), elt)


@libfunction("topk_fold")
def topk_fold(el, k, scores, *, axis=-1, largest=True):
  """Exact ordered tie-breaking top-``k`` reduction using the associative top-``k`` monoid.

  Returns ``(values, indices)`` sorted by ``(val, index)`` with first-index tie
  breaking matching ``np.argsort(..., kind="stable")`` (ascending for
  ``largest=False``, descending for ``largest=True``; NaNs always sort last).
  When ``k`` exceeds the lane length, unfilled slots have ``index == -1``.
  On 2D pairwise score expressions, tiles automatically with zero ``O(N * M)``
  intermediate allocation on CPU, CUDA, and Metal.
  """
  from ..syntax import ir

  k_int = el.static_int(k, "k")
  if k_int < 0 or k_int > MAX_FOLD_CARRIES // 2:
    raise el.error(
      StaticValueError,
      f"topk_fold() requires 0 <= k <= {MAX_FOLD_CARRIES // 2}, got {k_int}",
    )
  largest_b = largest if isinstance(largest, bool) else el.static_bool(largest, "largest")
  arr = el.as_array(_arrayish(el, scores))
  if arr.rank == 0:
    raise el.error(TypeInferenceError, "topk_fold() needs an array with at least one dimension")
  ax = norm_axis(el, axis, arr.rank)
  elt = arr.elt.strong

  tape = el.tape
  jvp = el.jvp
  need_tape = tape is not None and not tape.muted and elt.kind == "f" and tape.is_active(el, arr)
  need_jvp = jvp is not None and not jvp.muted and elt.kind == "f" and jvp.has_tangent(el, arr)
  if need_tape or need_jvp:
    if need_tape:
      tape.muted += 1
    if need_jvp:
      jvp.muted += 1
    try:
      vals, idxs = topk_fold.impl(el, k_int, arr, axis=ax, largest=largest_b)
    finally:
      if need_tape:
        tape.muted -= 1
      if need_jvp:
        jvp.muted -= 1
    if need_jvp:
      d_arr = jvp.tangent_array(el, arr)
      if d_arr is not None:
        with jvp.computing(el):
          d_arr_c = el.as_concrete(d_arr)
          idxs_c = el.as_concrete(idxs)

          def _d_at(coord):
            raw_k = cast(el.tmp(el.elem(idxs_c, coord), "tk"), I)
            valid = cmp("ge", raw_k, ZERO)
            safe_k = ops.select(valid, raw_k, ZERO)
            src_c = tuple(coord[:ax]) + (safe_k,) + tuple(coord[ax + 1 :])
            return ops.select(valid, el.elem(d_arr_c, src_c), Const(0.0, elt))

          d_vals = el.materialize(el.delayed(elt, vals.shape, _d_at, cost=2), "d_topk")
        jvp.bind_array(el, vals, d_vals)
    if need_tape:
      from ..transforms.autodiff import Node

      node = Node("topk", (arr,), (el.as_concrete(idxs), ax))
      tape.record(el, node)
      tape.register(el, node, vals)
    return (vals, idxs)
  if k_int == 0:
    out_shape = arr.shape[:ax] + (ZERO,) + arr.shape[ax + 1 :]
    return el.alloc_array(elt, out_shape, label="topk_v"), el.alloc_array(I, out_shape, label="topk_i")

  sent_v = _topk_sentinel(elt, largest_b)
  neg1 = Const(-1, I)

  def better(va: Expr, ia: Expr, vb: Expr, ib: Expr) -> Expr:
    a_val = cmp("ge", ia, ZERO)
    b_val = cmp("ge", ib, ZERO)
    if elt.kind == "f":
      a_nan = cmp("ne", va, va)
      b_nan = cmp("ne", vb, vb)
      v_win = cmp("gt" if largest_b else "lt", va, vb)
      tie = ops.land(cmp("eq", va, vb), cmp("lt", ia, ib))
      nan_tie = ops.land(a_nan, ops.land(b_nan, cmp("lt", ia, ib)))
      both = ops.lor(ops.land(ops.lnot(a_nan), ops.lor(b_nan, ops.lor(v_win, tie))), nan_tie)
    else:
      v_win = cmp("gt" if largest_b else "lt", va, vb)
      tie = ops.land(cmp("eq", va, vb), cmp("lt", ia, ib))
      both = ops.lor(v_win, tie)
    return ops.land(a_val, ops.lor(ops.lnot(b_val), both))

  def lift(val, idx):
    v0 = cast(el.as_scalar(val, "topk element"), elt)
    vs = [v0] + [sent_v] * (k_int - 1)
    ix = [cast(idx, I)] + [neg1] * (k_int - 1)
    return (_packed_const_array(el, elt, (k_int,), vs), _packed_const_array(el, I, (k_int,), ix))

  def merge(left, right):
    lv_arr, li_arr = left
    rv_arr, ri_arr = right
    lv = [el.elem(lv_arr, (Const(t, I),)) for t in range(k_int)]
    li = [el.elem(li_arr, (Const(t, I),)) for t in range(k_int)]
    rv = [el.elem(rv_arr, (Const(t, I),)) for t in range(k_int)]
    ri = [el.elem(ri_arr, (Const(t, I),)) for t in range(k_int)]
    out_v: list[Expr] = []
    out_i: list[Expr] = []
    pa: Expr = ZERO
    for t in range(k_int):
      if t == 0:
        av, ai, bv, bi = lv[0], li[0], rv[0], ri[0]
      else:
        pb = el.tmp(ops.isub(Const(t, I), pa), "pb")
        av = el.tmp(_select_tree(lv[: t + 1], pa, 0, t + 1), "av")
        ai = el.tmp(_select_tree(li[: t + 1], pa, 0, t + 1), "ai")
        bv = el.tmp(_select_tree(rv[: t + 1], pb, 0, t + 1), "bv")
        bi = el.tmp(_select_tree(ri[: t + 1], pb, 0, t + 1), "bi")
      take_a = el.tmp(better(av, ai, bv, bi), "ta")
      out_v.append(el.tmp(ops.select(take_a, av, bv), "mv"))
      out_i.append(el.tmp(ops.select(take_a, ai, bi), "mi"))
      if t + 1 < k_int:
        pa = el.tmp(ops.select(take_a, ops.iadd(pa, ops.ONE), pa), "pa")
    return (_packed_const_array(el, elt, (k_int,), out_v), _packed_const_array(el, I, (k_int,), out_i))

  init_state = (
    _packed_const_array(el, elt, (k_int,), [sent_v] * k_int),
    _packed_const_array(el, I, (k_int,), [neg1] * k_int),
  )

  # Build top-k directly via _fold_impl (host IR builders, works for 1D and ND, triggers pairwise_fold on 2D!)
  if arr.rank == 1:
    res = _fold_impl(el, lift, merge, arr, init=init_state, axis=0, indexed=True)
    # Materialize delayed 1D outputs cleanly
    out_v = el.alloc_array(elt, (Const(k_int, I),), label="topk_v")
    out_i = el.alloc_array(I, (Const(k_int, I),), label="topk_i")
    for t in range(k_int):
      el.emit(ir.Store(out_v.buf, Const(t, I), el.elem(res[0], (Const(t, I),))))
      el.emit(ir.Store(out_i.buf, Const(t, I), el.elem(res[1], (Const(t, I),))))
    el.record_store(out_v)
    el.record_store(out_i)
    return (out_v, out_i)

  vals, idxs = _fold_impl(
    el,
    lift,
    merge,
    arr,
    init=init_state,
    axis=ax,
    indexed=True,
    emit_cb=lambda st: st,
  )
  # Move the trailing k-axis back to `ax` if ax != arr.rank - 1
  if ax != arr.rank - 1:
    perm = tuple(range(ax)) + (arr.rank - 1,) + tuple(range(ax, arr.rank - 1))
    vals = el.transpose(vals, perm)
    idxs = el.transpose(idxs, perm)
  return (vals, idxs)


@libfunction("topk")
def topk(el, scores, k, *, axis=-1, largest=True):
  """Alias for :func:`topk_fold` with ``(scores, k)`` argument order."""
  return topk_fold.impl(el, k, scores, axis=axis, largest=largest)


@libfunction("tanimoto_topk")
def tanimoto_topk(el, queries, refs, k):
  """Exact top-``k`` Tanimoto similarity search over packed ``uint64`` bit-fingerprints.

  ``queries`` has shape ``(N, W)`` and ``refs`` has shape ``(M, W)`` (e.g., ``W = 32``
  for 2048-bit Morgan fingerprints). Precomputes per-fingerprint popcounts in
  ``O((N + M) * W)`` and streams pairwise ``popcount(q & r) / (pop_q + pop_r - inter)``
  directly into the associative top-``k`` monoid with zero ``O(N * M)`` allocation.
  Returns ``(similarities: float32[N, k], indices: int64[N, k])``.
  """
  from ..ndtypes import Float32, ScalarType, UInt64
  from ..syntax import ir

  q = _prepare(el, el.as_array(_arrayish(el, queries)))
  r = _prepare(el, el.as_array(_arrayish(el, refs)))
  if q.rank != 2 or r.rank != 2:
    raise el.error(TypeInferenceError, "tanimoto_topk() expects 2D fingerprint arrays (N, W) and (M, W)")
  if q.elt.kind not in "iu" or r.elt.kind not in "iu":
    raise el.error(TypeInferenceError, "tanimoto_topk() expects integer/uint64 bit-packed fingerprints")
  nq, wq = q.shape
  nr, wr = r.shape
  if not el.dim_equal(wq, wr):
    el.check(cmp("eq", wq, wr), "ValueError", "fingerprint word widths do not match ({0} vs {1})", wq, wr)

  def _as_u64_bits(e):
    """Zero-extend sub-64-bit signed words through their same-width unsigned type before widening to UInt64."""
    if e.type.kind == "i" and e.type.bits < 64:
      return cast(cast(e, ScalarType(f"uint{e.type.bits}")), UInt64)
    return cast(e, UInt64)

  def _row_popcounts(mat, nrows):
    out = el.alloc_array(UInt64, (nrows,), label="pop")

    def row_body(idx):
      i = idx[0]
      w = ir.Var("w", I)
      body: list = []
      sc = el.new_scope("loop")
      with el.in_block(body, sc, loop=True):
        cnt = _popcount_u64(el, _as_u64_bits(el.elem(mat, (i, w))))
      res = ir.Var("pop_sum", UInt64)
      a, b = ir.Var("pa", UInt64), ir.Var("pb", UInt64)
      f = ir.Fold(res, cnt, a, b, [], ops.prim("add", (a, b), UInt64), init=Const(0, UInt64), op="add")
      el.emit(ir.IndexReduce((wq,), (w,), body, [f], sc, el.scope, False, el.loc()))
      el.emit(ir.Store(out.buf, i, res))

    el.index_map((nrows,), row_body, ordered=False)
    el.record_store(out)
    return out

  pop_q = _row_popcounts(q, nq)
  pop_r = _row_popcounts(r, nr)

  # Build delayed pairwise Tanimoto matrix so _fold_impl materializes it as a sibling
  # inside `nonempty` -> fusion.py + pairwise_fold.py eliminate the O(N*M) allocation!
  def pair_elem(idx):
    i, j = idx
    w = ir.Var("w", I)
    body: list = []
    sc = el.new_scope("loop")
    with el.in_block(body, sc, loop=True):
      qv = _as_u64_bits(el.elem(q, (i, w)))
      rv = _as_u64_bits(el.elem(r, (j, w)))
      inter_w = _popcount_u64(el, ops.prim("bitand", (qv, rv), UInt64))
    inter = ir.Var("inter", UInt64)
    a, b = ir.Var("ia", UInt64), ir.Var("ib", UInt64)
    f = ir.Fold(inter, inter_w, a, b, [], ops.prim("add", (a, b), UInt64), init=Const(0, UInt64), op="add")
    el.emit(ir.IndexReduce((wq,), (w,), body, [f], sc, el.scope, False, el.loc()))
    pq = el.elem(pop_q, (i,))
    pr = el.elem(pop_r, (j,))
    union = el.tmp(ops.prim("sub", (ops.prim("add", (pq, pr), UInt64), inter), UInt64), "union")
    return ops.select(
      cmp("gt", union, Const(0, UInt64)),
      ops.prim("div", (cast(inter, Float32), cast(union, Float32)), Float32),
      Const(0.0, Float32),
    )

  scores = el.delayed(Float32, (nq, nr), pair_elem, cost=CHEAP_COST + 1)
  return topk_fold.impl(el, k, scores, axis=1, largest=True)


@libfunction("ipa_attention")
def ipa_attention(el, q, k, v, coords_q, coords_k, gamma, *, scale=None, bias=None):
  """3D distance-biased Invariant Point Attention (AlphaFold2/3 IPA / SE(3) attention).

  Computes ``softmax((Q @ K^T) * scale - gamma * ||coords_q[i] - coords_k[j]||^2 [+ bias]) @ V``
  in a single Law 1 SRAM-tiled monoidal pass with zero ``O(N^2)`` allocation, and
  differentiates via Law 2 ``D_i``-recomputed backward w.r.t. ``(q, k, v, coords_q, coords_k, bias)``.
  """
  from ..transforms.algebraic_vjp import emit_ipa_attention

  q_a = el.as_array(_arrayish(el, q))
  k_a = el.as_array(_arrayish(el, k))
  v_a = el.as_array(_arrayish(el, v))
  xq_a = el.as_array(_arrayish(el, coords_q))
  xk_a = el.as_array(_arrayish(el, coords_k))
  b_a = None if bias is None else el.as_array(_arrayish(el, bias))
  g_e = el.as_scalar(gamma, "gamma")
  sc_e = None if scale is None else el.as_scalar(scale, "scale")
  return emit_ipa_attention(el, q_a, k_a, v_a, xq_a, xk_a, g_e, scale=sc_e, bias=b_a)


@libfunction("stop_gradient")
def stop_gradient(el, x):
  """Zero-copy gradient barrier: returns ``x`` with zero derivative under ``grad``."""
  from ..transforms.ad_loops import stop_gradient_val

  return stop_gradient_val(el, x)


@libfunction("custom_vjp_call")
def _custom_vjp_staged(el, fn, fwd, bwd, *args, **kwargs):
  from ..transforms.ad_loops import custom_vjp_call

  return custom_vjp_call(el, fn, fwd, bwd, list(args), kwargs)


class CustomVJP:
  """Function with a user-defined reverse-mode VJP rule (``@par.custom_vjp``).

  Register forward and backward rules via ``.defvjp(fwd, bwd)``, where:
  * ``fwd(*args) -> (output, residuals)``
  * ``bwd(residuals, g_output) -> cotangent_or_tuple``
  """

  def __init__(self, fn):
    if not callable(fn):
      raise TypeError(f"custom_vjp expects a callable, got {type(fn).__name__}")
    self.fn = fn
    self.fwd = None
    self.bwd = None
    self.__name__ = getattr(fn, "__name__", "custom_vjp_fn")
    self.__doc__ = getattr(fn, "__doc__", None)

  def defvjp(self, fwd, bwd):
    if not callable(fwd) or not callable(bwd):
      raise TypeError("defvjp(fwd, bwd) expects two callables")
    self.fwd = fwd
    self.bwd = bwd
    return self

  def __repr__(self) -> str:
    return f"<parakeet custom_vjp {self.__name__}>"

  def __call__(self, *args, **kwargs):
    return self.fn(*args, **kwargs)

  def __parakeet_value__(self, el):
    from ..syntax.values import Partial

    return Partial(_custom_vjp_staged, (el.lift(self.fn), el.lift(self.fwd), el.lift(self.bwd)), {})


def custom_vjp(fn):
  """Decorate ``fn`` so it can register a custom reverse-mode VJP with ``.defvjp(fwd, bwd)``."""
  return CustomVJP(fn)


@libfunction("checkpoint_call")
def _checkpoint_staged(el, fn, *args, **kwargs):
  from ..transforms.ad_loops import checkpoint_call

  return checkpoint_call(el, fn, list(args), kwargs)


class Checkpointed:
  """Rematerialized function wrapper returned by :func:`checkpoint`."""

  def __init__(self, fn):
    if not callable(fn):
      raise TypeError(f"checkpoint expects a callable, got {type(fn).__name__}")
    self.fn = fn
    self.__name__ = getattr(fn, "__name__", "checkpointed")
    self.__doc__ = getattr(fn, "__doc__", None)

  def __repr__(self) -> str:
    return f"<parakeet checkpoint {self.__name__}>"

  def __call__(self, *args, **kwargs):
    return self.fn(*args, **kwargs)

  def __parakeet_value__(self, el):
    from ..syntax.values import Partial

    return Partial(_checkpoint_staged, (el.lift(self.fn),), {})


def checkpoint(fn):
  """Drop forward intermediates of ``fn`` and recompute them during ``grad``."""
  return Checkpointed(fn)


@core.register(checkpoint)
def _checkpoint_inline(el, fn):
  from ..syntax.values import Partial

  return Partial(_checkpoint_staged, (fn,), {})


# ---------------------------------------------------------------------------
# Forward-mode JVP, @custom_jvp, linearize, general VJP, HVP, Jacobians
# ---------------------------------------------------------------------------


@libfunction("jvp_apply")
def _jvp_staged(el, fn, primals, tangents):
  from ..transforms.jvp import jvp_call

  return jvp_call(el, fn, primals, tangents, "jvp")


def jvp(fn, primals, tangents):
  """Forward-mode Jacobian-vector product: returns ``(fn(*primals), d_fn)``.

  ``primals`` and ``tangents`` are tuples of arguments matching ``fn`` (or single
  values for unary functions). Works from plain Python and inside ``@par.jit``
  or other AD transforms with zero tape and ``O(1)`` extra memory.
  """
  from ..api import call_staged

  p = primals if isinstance(primals, tuple) else tuple(primals) if isinstance(primals, list) else (primals,)
  t = tangents if isinstance(tangents, tuple) else tuple(tangents) if isinstance(tangents, list) else (tangents,)
  return call_staged(_jvp_staged, (fn, p, t), {})


@core.register(jvp)
def _jvp_inline(el, fn, primals, tangents):
  from ..transforms.jvp import jvp_call

  return jvp_call(el, fn, primals, tangents, "jvp")


@libfunction("custom_jvp_call")
def _custom_jvp_staged(el, fn, jvp_rule, *args, **kwargs):
  from ..transforms.jvp import custom_jvp_call

  return custom_jvp_call(el, fn, jvp_rule, list(args), kwargs)


class CustomJVP:
  """Function with a user-defined forward-mode JVP rule (``@par.custom_jvp``).

  Register a tangent rule via ``.defjvp(jvp_rule)``, where:
  * ``jvp_rule(primals, tangents) -> (primal_out, tangent_out)``
  """

  def __init__(self, fn):
    if not callable(fn):
      raise TypeError(f"custom_jvp expects a callable, got {type(fn).__name__}")
    self.fn = fn
    self.jvp_rule = None
    self.__name__ = getattr(fn, "__name__", "custom_jvp_fn")
    self.__doc__ = getattr(fn, "__doc__", None)

  def defjvp(self, jvp_rule):
    if not callable(jvp_rule):
      raise TypeError("defjvp(jvp_rule) expects a callable")
    self.jvp_rule = jvp_rule
    return self

  def __repr__(self) -> str:
    return f"<parakeet custom_jvp {self.__name__}>"

  def __call__(self, *args, **kwargs):
    return self.fn(*args, **kwargs)

  def __parakeet_value__(self, el):
    from ..syntax.values import Partial

    return Partial(_custom_jvp_staged, (el.lift(self.fn), el.lift(self.jvp_rule)), {})


def custom_jvp(fn):
  """Decorate ``fn`` so it can register a custom forward-mode JVP with ``.defjvp(rule)``."""
  return CustomJVP(fn)


@libfunction("vjp_apply")
def _vjp_apply_staged(el, fn, primals, cotangent):
  from ..transforms.jvp import vjp_apply

  return vjp_apply(el, fn, primals, cotangent, "vjp")


def vjp(fn, *primals):
  """General non-scalar reverse-mode AD: returns ``(primal_out, vjp_fn)``.

  ``vjp_fn(cotangent)`` returns a tuple of input cotangents matching ``primals``
  for arbitrary scalar, array, tuple, record, or ``RaggedArray`` outputs.
  """
  from ..api import call_staged
  from ..api import jit as _jit

  primal_out = _jit(fn)(*primals)

  def pull(cotangent):
    _, grads = call_staged(_vjp_apply_staged, (fn, primals, cotangent), {})
    return grads

  return primal_out, pull


def linearize(fn, *primals):
  """Linearize ``fn`` at ``primals``: returns ``(primal_out, lin_fn)`` where
  ``lin_fn(*tangents)`` computes the directional derivative ``JVP(fn, primals, tangents)``."""
  from ..api import call_staged
  from ..api import jit as _jit

  primal_out = _jit(fn)(*primals)

  def lin_fn(*tangents):
    _, dout = call_staged(_jvp_staged, (fn, primals, tangents), {})
    return dout

  return primal_out, lin_fn


@libfunction("linear_transpose_apply")
def _linear_transpose_staged(el, linear_fn, example_inputs, cotangent):
  from ..transforms.transpose import linear_transpose_apply

  return linear_transpose_apply(el, linear_fn, example_inputs, cotangent, "linear_transpose")


def linear_transpose(linear_fn, *example_inputs):
  """Return ``transpose_fn(cotangent)`` computing ``L^T(cotangent)`` for a linear function ``L``."""
  from ..api import call_staged

  def transpose_fn(cotangent):
    return call_staged(_linear_transpose_staged, (linear_fn, example_inputs, cotangent), {})

  return transpose_fn


@libfunction("hvp_apply")
def _hvp_staged(el, fn, primals, tangents, argnum=None):
  from ..transforms.jvp import hvp_call

  return hvp_call(el, fn, primals, tangents, argnum=argnum, what="hvp")


def hvp(fn, primals, tangents, argnum=None):
  """Exact Hessian-vector product ``H(fn)(primals) @ tangents`` via forward-over-reverse AD.

  Runs in ~2x reverse-mode cost with zero ``O(N^2)`` Hessian allocation.
  """
  from ..api import call_staged

  p = tuple(primals) if isinstance(primals, list) else primals
  t = tuple(tangents) if isinstance(tangents, list) else tangents
  return call_staged(_hvp_staged, (fn, p, t), {"argnum": argnum})


@core.register(hvp)
def _hvp_inline(el, fn, primals, tangents, argnum=None):
  from ..transforms.jvp import hvp_call

  return hvp_call(el, fn, primals, tangents, argnum=argnum, what="hvp")


@libfunction("jacfwd_apply")
def _jacfwd_staged(el, fn, *args, argnum=0, **kwargs):
  from ..transforms.jvp import jacfwd_call

  return jacfwd_call(el, fn, argnum, list(args), kwargs, "jacfwd")


@libfunction("jacrev_apply")
def _jacrev_staged(el, fn, *args, argnum=0, **kwargs):
  from ..transforms.jvp import jacrev_call

  return jacrev_call(el, fn, argnum, list(args), kwargs, "jacrev")


@libfunction("hessian_apply")
def _hessian_staged(el, fn, *args, argnum=0, **kwargs):
  from ..transforms.jvp import hessian_call

  return hessian_call(el, fn, argnum, list(args), kwargs, "hessian")


def jacfwd(fn, argnum=0):
  """Forward-mode Jacobian of ``fn`` with respect to argument ``argnum``."""
  return Gradient(fn, argnum, _jacfwd_staged)


def jacrev(fn, argnum=0):
  """Reverse-mode Jacobian of ``fn`` with respect to argument ``argnum``."""
  return Gradient(fn, argnum, _jacrev_staged)


def hessian(fn, argnum=0):
  """Exact Hessian of scalar-valued ``fn`` via forward-over-reverse AD."""
  return Gradient(fn, argnum, _hessian_staged)


@core.register(jacfwd)
def _jacfwd_inline(el, fn, argnum=0):
  from ..syntax.values import Partial

  return Partial(_jacfwd_staged, (fn,), {"argnum": argnum})


@core.register(jacrev)
def _jacrev_inline(el, fn, argnum=0):
  from ..syntax.values import Partial

  return Partial(_jacrev_staged, (fn,), {"argnum": argnum})


@core.register(hessian)
def _hessian_inline(el, fn, argnum=0):
  from ..syntax.values import Partial

  return Partial(_hessian_staged, (fn,), {"argnum": argnum})


@core.register(builtins.map)
def _builtin_map(el, f, *args):
  return map.impl(el, f, *args, axis=0)


@core.register(builtins.filter)
def _builtin_filter(el, pred, x):
  return filter.impl(el, pred, x)


__all__ = [
    "LibFunction", "map", "each", "allpairs", "outer_map", "reduce", "scan", "imap", "ireduce",
    "parfor", "filter", "filter_reduce", "scatter_add", "scan_axis", "grad", "value_and_grad", "Gradient",
    "topk_fold", "topk", "tanimoto_topk", "ipa_attention", "stop_gradient", "custom_vjp", "CustomVJP",
    "checkpoint", "Checkpointed", "jvp", "custom_jvp", "CustomJVP", "vjp", "linearize", "linear_transpose",
    "hvp", "jacfwd", "jacrev", "hessian",
]  # fmt: skip
_ = (Const, I)
