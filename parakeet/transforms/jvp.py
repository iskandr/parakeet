"""Staged forward-mode differentiation (JVP), linearization, general VJP,
Hessian-vector products (HVP), Jacobians, and ``@custom_jvp``.

Forward-mode AD propagates ``(primal, tangent)`` pairs during elaboration with
zero tape and ``O(1)`` extra memory — even through arbitrary runtime ``for`` /
``while`` loops, ``par.fold``, ``par.scan``, ``par.map``, ``RaggedArray``,
records, and ``grad`` (giving forward-over-reverse ``hvp`` with zero ``O(N^2)``
Hessian materialization).
"""

from __future__ import annotations

import contextlib
from typing import Any

from ..errors import TypeInferenceError, UnsupportedError
from ..ndtypes import ScalarType
from ..ragged import RaggedArray
from ..syntax import ops
from ..syntax.ir import Assign, Const, Expr, Load, Store, Var
from ..syntax.ops import ZERO, I, cast, cmp, iadd
from ..syntax.records import record_type
from ..syntax.values import ArrayVal, describe
from .ad_loops import flatten_tree, unflatten_tree, zero_like_leaf, zero_like_tree
from .derivatives import is_float, pushforward

# ---------------------------------------------------------------------------
# JVPContext: staged tangent environment (supports arbitrary nesting)
# ---------------------------------------------------------------------------


class JVPContext:
  """Active tangent mapping for one level of ``par.jvp`` elaboration.

  Supports arbitrary nesting (``jvp(jvp(f))``) via ``outer`` and composes
  directly with reverse-mode ``Tape`` (``hvp = jvp(grad(f))``).
  """

  def __init__(self, outer: JVPContext | None = None):
    self.outer: JVPContext | None = outer
    self.muted: int = 0
    # Scalar Var.uid -> tangent Expr (same float ScalarType)
    self.scalar_t: dict[int, Expr] = {}
    # Concrete buffer Var.uid -> (tangent BufType Var, tangent roots)
    self.buf_t: dict[int, Var] = {}
    self.buf_roots: dict[int, Any] = {}
    # Optional 1-hot symbolic tangent: buf.uid -> (flat_pos: Expr, elt: ScalarType)
    self.buf_one_hot: dict[int, tuple[Expr, ScalarType]] = {}
    # Explicit delayed ArrayVal id -> tangent ArrayVal (rare; most delayed arrays pushforward via elem)
    self.arr_t: dict[int, ArrayVal] = {}
    # Keep delayed ArrayVal keys alive while context is live
    self.keep: list[Any] = []
    # Mutable loop/branch target (Var.uid or id(ArrayVal)) -> paired mutable tangent target
    self.targets: dict[int, Any] = {}
    # Cache of delayed array activity probes (id(arr) -> bool)
    self._known_active: dict[int, bool] = {}

  @contextlib.contextmanager
  def computing(self, el):
    """Step aside while emitting tangent computations (popping to ``outer`` if nested)."""
    if self.outer is None:
      self.muted += 1
      try:
        yield
      finally:
        self.muted -= 1
    else:
      saved = el.jvp
      el.jvp = self.outer
      try:
        yield
      finally:
        el.jvp = saved

  # ---- scalar & buffer tangent lookup --------------------------------------

  def tangent_var(self, el, v: Var) -> Expr | None:
    if not is_float(v):
      return None
    if v.uid in self.scalar_t:
      return self.scalar_t[v.uid]
    # Mutable scalar target paired with a mutable tangent Var
    tgt = self.targets.get(v.uid)
    if isinstance(tgt, Var):
      return tgt
    # Pure SSA definition in el.defs: differentiate lazily and cache
    defn = el.defs.get(v.uid)
    if defn is not None and v.uid not in el.mutables:
      with self.computing(el):
        dt = pushforward(
          el,
          defn,
          lambda u: self.tangent_var(el, u),
          lambda ld: self.tangent_load(el, ld),
          bound=v,
        )
      if dt is not None:
        dt = cast(dt, v.type)
        self.scalar_t[v.uid] = dt
      return dt
    return None

  def tangent_load(self, el, ld: Load) -> Expr | None:
    if not is_float(ld):
      return None
    if ld.buf.uid in self.buf_one_hot:
      pos, elt = self.buf_one_hot[ld.buf.uid]
      return ops.select(cmp("eq", ld.index, pos), Const(1.0, elt), Const(0.0, elt))
    dbuf = self.buf_t.get(ld.buf.uid)
    if dbuf is None:
      return None
    return Load(dbuf, ld.index)

  def tangent_expr(self, el, e: Expr, bound: Var | None = None) -> Expr | None:
    if isinstance(e, Const) or not is_float(e):
      return None
    if isinstance(e, Var):
      return self.tangent_var(el, e)
    with self.computing(el):
      return pushforward(
        el,
        e,
        lambda u: self.tangent_var(el, u),
        lambda ld: self.tangent_load(el, ld),
        bound=bound,
      )

  def has_tangent_expr(self, el, e: Expr) -> bool:
    if isinstance(e, Const) or not is_float(e):
      return False
    if isinstance(e, Var):
      if e.uid in self.scalar_t or e.uid in self.targets:
        return True
      defn = el.defs.get(e.uid)
      if defn is not None and e.uid not in el.mutables:
        return self.has_tangent_expr(el, defn)
      return False
    if isinstance(e, Load):
      return e.buf.uid in self.buf_t or e.buf.uid in self.buf_one_hot
    from ..syntax.ir import Cast, Prim

    if isinstance(e, Cast):
      return is_float(e.arg) and self.has_tangent_expr(el, e.arg)
    if isinstance(e, Prim):
      if e.op == "select":
        return self.has_tangent_expr(el, e.args[1]) or self.has_tangent_expr(el, e.args[2])
      return any(is_float(a) and self.has_tangent_expr(el, a) for a in e.args)
    return False

  def tangent_array(self, el, arr: ArrayVal) -> ArrayVal | None:
    """Return the tangent ``ArrayVal`` of ``arr`` (or ``None`` if structurally zero)."""
    if arr.elt.kind != "f":
      return None
    if arr.buf is not None:
      dbuf = self.buf_t.get(arr.buf.uid)
      if dbuf is None:
        return None
      out = ArrayVal(
        arr.elt,
        arr.shape,
        buf=dbuf,
        offset=arr.offset,
        strides=arr.strides,
        roots=self.buf_roots.get(arr.buf.uid),
      )
      out.contig = arr.contig
      out.inj = arr.inj
      return out
    if id(arr) in self.arr_t:
      return self.arr_t[id(arr)]
    if not self.has_tangent_array(el, arr):
      return None
    # Delayed array with active inputs: build a matching delayed tangent view
    ctx = self

    def d_at(idx):
      v = arr.fn(idx)
      dt = ctx.tangent_expr(el, v)
      return Const(0.0, arr.elt) if dt is None else cast(dt, arr.elt)

    d_arr = el.delayed(arr.elt, arr.shape, d_at, cost=arr.cost + 4)
    self.arr_t[id(arr)] = d_arr
    self.keep.append(arr)
    return d_arr

  def has_tangent_array(self, el, arr: ArrayVal) -> bool:
    if arr.elt.kind != "f":
      return False
    if arr.buf is not None:
      return arr.buf.uid in self.buf_t
    if id(arr) in self.arr_t:
      return True
    hit = self._known_active.get(id(arr))
    if hit is not None:
      return hit
    probe = tuple(Var("i", I) for _ in arr.shape)
    for p, d in zip(probe, arr.shape):
      el.nonneg.add(p.uid)
      el.upper[p.uid] = d
    saved_s = dict(self.scalar_t)
    saved_b = dict(self.buf_t)
    saved_br = dict(self.buf_roots)
    saved_a = dict(self.arr_t)
    saved_t = dict(self.targets)
    el.loop_depth += 1
    try:
      with el.trial():
        r = arr.fn(probe)
        active = self.has_tangent_expr(el, r)
    finally:
      el.loop_depth -= 1
      self.scalar_t = saved_s
      self.buf_t = saved_b
      self.buf_roots = saved_br
      self.arr_t = saved_a
      self.targets = saved_t
    self._known_active[id(arr)] = active
    self.keep.append(arr)
    return active

  def has_tangent(self, el, v) -> bool:
    if isinstance(v, Expr):
      return self.has_tangent_expr(el, v)
    if isinstance(v, ArrayVal):
      return self.has_tangent_array(el, v)
    if isinstance(v, RaggedArray):
      return self.has_tangent_array(el, v.values)
    if record := record_type(v):
      return self.has_tangent(el, record.values(v))
    if isinstance(v, tuple):
      return any(self.has_tangent(el, x) for x in v)
    return False

  def bind_array(self, el, primal: ArrayVal, tangent: ArrayVal) -> None:
    """Associate concrete/delayed ``primal`` with its ``tangent`` array."""
    if primal.elt.kind != "f":
      return
    if primal.buf is not None:
      if tangent.buf is None or not tangent.contig or tangent.elt is not primal.elt:
        with self.computing(el):
          tangent = el.materialize(el.cast_array(tangent, primal.elt), "dt")
      self.buf_t[primal.buf.uid] = tangent.buf
      self.buf_roots[primal.buf.uid] = tangent.roots
    else:
      self.arr_t[id(primal)] = tangent
      self.keep.append(primal)

  def ensure_zero_buf_tangent(self, el, arr: ArrayVal, label: str = "d_adj") -> ArrayVal | None:
    """Ensure concrete float buffer ``arr`` has a zero-initialized tangent buffer
    allocated in the current scope (used for mutable adjoint / checkpoint buffers)."""
    if arr.elt.kind != "f" or arr.buf is None:
      return None
    if arr.buf.uid in self.buf_t:
      return self.tangent_array(el, arr)
    with self.computing(el):
      d_arr = el.alloc_array(arr.elt, arr.shape, zero=True, label=label)
    self.buf_t[arr.buf.uid] = d_arr.buf
    self.buf_roots[arr.buf.uid] = d_arr.roots
    return d_arr

  # ---- mutable loop & branch target management -----------------------------

  def make_tangent_target(self, el, target, name: str, home, init_block: list | None = None):
    """Allocate a paired mutable tangent target matching ``target`` (for loops/branches)."""
    if isinstance(target, Var):
      if not is_float(target):
        return None
      dt = Var(f"d_{name}", target.type)
      el.mutables.add(dt.uid)
      if init_block is not None:
        init_block.append(Assign(dt, Const(0.0, target.type)))
      self.targets[target.uid] = dt
      return dt
    if isinstance(target, ArrayVal):
      if target.elt.kind != "f":
        return None
      d_arr = el.make_target(
        ("a", target.elt, target.rank), target, f"d_{name}", home, init_block=init_block, unknown_roots=True
      )
      self.targets[id(target)] = d_arr
      self.buf_t[target.buf.uid] = d_arr.buf
      self.buf_roots[target.buf.uid] = d_arr.roots
      self.keep.append(target)
      return d_arr
    if record := record_type(target):
      self.make_tangent_target(el, record.values(target), name, home, init_block=init_block)
      return None
    if isinstance(target, tuple):
      for k, t in enumerate(target):
        self.make_tangent_target(el, t, f"{name}_{k}", home, init_block=init_block)
      return None
    return None

  def assign_tangent_target(self, el, target, v) -> bool:
    """Assign the tangent of ``v`` into the paired tangent target of ``target``.

    Returns ``False`` if ``v`` carries a tangent but ``target`` does not yet have
    a paired tangent target (signaling the caller to restart with a JVP loop hint).
    """
    if isinstance(target, Var):
      if not is_float(target):
        return True
      dt_var = self.targets.get(target.uid)
      dv = self.tangent_expr(el, el.as_scalar(v, "value"))
      if dt_var is None:
        return dv is None
      with self.computing(el):
        el.emit(Assign(dt_var, cast(dv, dt_var.type) if dv is not None else Const(0.0, dt_var.type)))
      return True
    if isinstance(target, ArrayVal):
      if target.elt.kind != "f":
        return True
      d_target = self.targets.get(id(target))
      dv = self.tangent_array(el, v) if isinstance(v, ArrayVal) else None
      if d_target is None:
        return dv is None
      with self.computing(el):
        if dv is None:
          dv = el.alloc_array(target.elt, v.shape, zero=True, label="dz")
        el.assign_target(d_target, dv)
      return True
    if record := record_type(target):
      return self.assign_tangent_target(el, record.values(target), record.values(v))
    if isinstance(target, tuple):
      ok = True
      for t, x in zip(target, v):
        if not self.assign_tangent_target(el, t, x):
          ok = False
      return ok
    return True

  # ---- pytree binding & extraction -----------------------------------------

  def bind_input(self, el, primal, tangent, what: str = "jvp"):
    """Bind ``(primal, tangent)`` tree leaves into this JVPContext and return the
    bound primal tree to pass into the differentiated function."""
    if isinstance(primal, RaggedArray):
      if not isinstance(tangent, RaggedArray):
        raise el.error(
          TypeInferenceError,
          f"{what}: tangent for RaggedArray must also be a RaggedArray, got {describe(tangent)}",
        )
      pv = self.bind_input(el, primal.values, tangent.values, what)
      return RaggedArray(pv, primal.offsets)
    if record := record_type(primal):
      t_rec = record_type(tangent)
      if t_rec != record:
        raise el.error(
          TypeInferenceError,
          f"{what}: primal and tangent record types differ ({describe(primal)} vs {describe(tangent)})",
        )
      return record.build(self.bind_input(el, record.values(primal), t_rec.values(tangent), what))
    if isinstance(primal, tuple):
      if not isinstance(tangent, tuple) or len(primal) != len(tangent):
        raise el.error(
          TypeInferenceError,
          f"{what}: primal and tangent tuple structures differ ({describe(primal)} vs {describe(tangent)})",
        )
      return tuple(self.bind_input(el, p, t, what) for p, t in zip(primal, tangent))
    if isinstance(primal, Expr):
      if not is_float(primal):
        return primal
      leaf = Var("p", primal.type)
      el.block.append(Assign(leaf, primal))
      if self.outer is not None:
        # Outer JVP / tape differentiates through the primal copy
        el.defs[leaf.uid] = primal
      t_expr = cast(el.as_scalar(tangent, f"{what} tangent"), primal.type)
      self.scalar_t[leaf.uid] = t_expr
      return leaf
    if isinstance(primal, ArrayVal):
      if primal.elt.kind != "f":
        return primal
      if not isinstance(tangent, ArrayVal):
        raise el.error(
          TypeInferenceError,
          f"{what}: tangent for array must be an array, got {describe(tangent)}",
        )
      if tangent.rank != primal.rank:
        raise el.error(
          TypeInferenceError,
          f"{what}: primal and tangent ranks differ ({primal.rank} vs {tangent.rank})",
        )
      for d_p, d_t in zip(primal.shape, tangent.shape):
        el.check(cmp("eq", d_p, d_t), "ValueError", f"{what}: primal and tangent shapes differ")
      # Fresh contiguous copy for primal if delayed, non-contiguous, or already aliased in buf_t
      if primal.buf is None or not primal.contig or primal.buf.uid in self.buf_t:
        with self.computing(el):
          primal = el.materialize(primal, "p")
      with self.computing(el):
        t_arr = el.cast_array(tangent, primal.elt)
        if t_arr.buf is None or not t_arr.contig:
          t_arr = el.materialize(t_arr, "t")
      self.buf_t[primal.buf.uid] = t_arr.buf
      self.buf_roots[primal.buf.uid] = t_arr.roots
      return primal
    # Static / discrete leaf (int, bool, str, None, FrozenHashTable metadata, etc.)
    return primal

  def bind_output_tree(self, el, primal, tangent) -> None:
    """Bind an explicit ``(primal, tangent)`` tree (e.g. from ``@custom_jvp``)."""
    if isinstance(primal, RaggedArray):
      if isinstance(tangent, RaggedArray):
        self.bind_output_tree(el, primal.values, tangent.values)
      return
    if record := record_type(primal):
      t_rec = record_type(tangent)
      if t_rec == record:
        self.bind_output_tree(el, record.values(primal), t_rec.values(tangent))
      return
    if isinstance(primal, tuple):
      if isinstance(tangent, tuple) and len(primal) == len(tangent):
        for p, t in zip(primal, tangent):
          self.bind_output_tree(el, p, t)
      return
    if isinstance(primal, Expr) and is_float(primal):
      t_e = cast(el.as_scalar(tangent, "custom_jvp tangent"), primal.type)
      if isinstance(primal, Var):
        self.scalar_t[primal.uid] = t_e
      return
    if isinstance(primal, ArrayVal) and primal.elt.kind == "f":
      self.bind_array(el, primal, el.as_array(tangent))

  def extract_primal_and_tangent(self, el, v):
    """Materialize any active delayed outputs while JVP is active, returning ``(primal, tangent)``."""
    if isinstance(v, RaggedArray):
      pv, tv = self.extract_primal_and_tangent(el, v.values)
      return RaggedArray(pv, v.offsets), RaggedArray(tv, v.offsets)
    if record := record_type(v):
      pv, tv = self.extract_primal_and_tangent(el, record.values(v))
      return record.build(pv), record.build(tv)
    if isinstance(v, tuple):
      pairs = [self.extract_primal_and_tangent(el, x) for x in v]
      return tuple(p for p, _ in pairs), tuple(t for _, t in pairs)
    if isinstance(v, Expr):
      if not is_float(v):
        return v, v
      dt = self.tangent_expr(el, v)
      return v, (Const(0.0, v.type) if dt is None else cast(dt, v.type))
    if isinstance(v, ArrayVal):
      if v.elt.kind != "f":
        return v, v
      if v.buf is None and self.has_tangent_array(el, v):
        v = el.materialize(v, "p")
      dt = self.tangent_array(el, v)
      if dt is None:
        with self.computing(el):
          return v, el.alloc_array(v.elt, v.shape, zero=True, label="dz")
      with self.computing(el):
        return v, el.as_concrete(el.cast_array(dt, v.elt))
    return v, v


# ---------------------------------------------------------------------------
# Public staged AD transforms: jvp, custom_jvp, linearize, vjp, hvp, jacfwd/rev
# ---------------------------------------------------------------------------


def _as_arg_tuple(v, name: str, el) -> tuple:
  if isinstance(v, tuple):
    return v
  return (v,)


def jvp_call(el, fn, primals, tangents, what: str = "jvp"):
  """Elaborate ``fn(*primals)`` in forward-mode AD, returning ``(primal_out, tangent_out)``."""
  p_tuple = _as_arg_tuple(primals, "primals", el)
  t_tuple = _as_arg_tuple(tangents, "tangents", el)
  if len(p_tuple) != len(t_tuple):
    raise el.error(
      TypeInferenceError,
      f"{what}: primals ({len(p_tuple)}) and tangents ({len(t_tuple)}) must have the same length",
    )
  p_bound_in = [el.bindable(a, "primal") for a in p_tuple]
  t_bound_in = [el.bindable(a, "tangent") for a in t_tuple]

  outer = el.jvp
  ctx = JVPContext(outer=outer)
  el.jvp_live += 1
  try:
    bound_args = [ctx.bind_input(el, p, t, what) for p, t in zip(p_bound_in, t_bound_in)]
    el.jvp = ctx
    try:
      out = el.call(fn, bound_args, {})
      out, dout = ctx.extract_primal_and_tangent(el, out)
    finally:
      el.jvp = outer
  finally:
    el.jvp_live -= 1
    el.jvp = outer
    if not el.ad_live and not el.jvp_live:
      el.defs.clear()
  return (out, dout)


def custom_jvp_call(el, fn, jvp_rule, args: list, kwargs: dict):
  """Staged dispatch for ``@par.custom_jvp`` functions."""
  jvp = el.jvp
  if jvp is None or jvp.muted or jvp_rule is None:
    return el.call(fn, args, kwargs)
  if kwargs:
    raise el.error(UnsupportedError, "custom_jvp() does not currently support keyword arguments inside jvp()")
  if not any(jvp.has_tangent(el, a) for a in args):
    return el.call(fn, args, kwargs)

  primals = tuple(args)
  tangents = tuple(jvp.extract_primal_and_tangent(el, a)[1] for a in args)
  with jvp.computing(el):
    res = el.call(jvp_rule, [primals, tangents], {})
  if not isinstance(res, tuple) or len(res) != 2:
    raise el.error(
      TypeInferenceError,
      f"custom_jvp rule must return a 2-tuple (primal_out, tangent_out), got {describe(res)}",
    )
  primal_out, tangent_out = res

  # Ensure float scalar leaves are named Vars so scalar_t can map them cleanly
  def _bind_out(p, t):
    if isinstance(p, RaggedArray):
      return RaggedArray(_bind_out(p.values, t.values), p.offsets)
    if record := record_type(p):
      return record.build(_bind_out(record.values(p), record.values(t)))
    if isinstance(p, tuple):
      return tuple(_bind_out(x, y) for x, y in zip(p, t))
    if isinstance(p, Expr) and is_float(p):
      pv = Var("cjvp_p", p.type)
      el.block.append(Assign(pv, p))
      if jvp.outer is not None:
        el.defs[pv.uid] = p
      jvp.scalar_t[pv.uid] = cast(el.as_scalar(t, "custom_jvp tangent"), p.type)
      return pv
    if isinstance(p, ArrayVal) and p.elt.kind == "f":
      with jvp.computing(el):
        pc = el.as_concrete(p)
        tc = el.as_concrete(el.cast_array(t, pc.elt))
      jvp.buf_t[pc.buf.uid] = tc.buf
      jvp.buf_roots[pc.buf.uid] = tc.roots
      return pc
    return p

  return _bind_out(primal_out, tangent_out)


def vjp_apply(el, fn, primals, cotangent, what: str = "vjp"):
  """General non-scalar reverse-mode AD: returns ``(primal_out, input_cotangents_tuple)``.

  Supports arbitrary output pytrees (scalars, N-D arrays, tuples, records,
  ``RaggedArray``) and arbitrary differentiable input pytrees.
  """
  from . import autodiff as _ad

  p_tuple = _as_arg_tuple(primals, "primals", el)
  outer = el.tape
  if outer is not None and (outer.muted or outer.parent is not None):
    raise el.error(UnsupportedError, f"{what} inside a mapped differentiated function is not supported")

  args = [el.bindable(a, "arg") for a in p_tuple]
  ct = el.bindable(cotangent, "cotangent")
  tape = _ad.Tape(el, outer=outer)
  el.ad_live += 1
  try:
    holders = []
    bound_args = []
    for a in args:
      ba, h = _ad._wrt_leaf(el, tape, a, what, in_record=True)
      bound_args.append(ba)
      holders.append(h)
    el.tape = tape
    try:
      out = el.call(fn, bound_args, {})
    finally:
      el.tape = outer
    # Validate & coerce cotangent structure against output structure
    out_leaves, out_spec = flatten_tree(out)
    ct_leaves, ct_spec = flatten_tree(ct)
    if len(out_leaves) != len(ct_leaves):
      raise el.error(
        TypeInferenceError,
        f"{what}: cotangent structure does not match output ({describe(ct)} vs {describe(out)})",
      )
    seed_leaves = []
    for ol, cl in zip(out_leaves, ct_leaves):
      if isinstance(ol, Expr) and is_float(ol):
        seed_leaves.append(cast(el.as_scalar(cl, f"{what} cotangent"), ol.type))
      elif isinstance(ol, ArrayVal) and ol.elt.kind == "f":
        ca = el.as_array(cl)
        if ca.rank != ol.rank:
          raise el.error(TypeInferenceError, f"{what}: cotangent rank {ca.rank} does not match output rank {ol.rank}")
        for d1, d2 in zip(ol.shape, ca.shape):
          el.check(cmp("eq", d1, d2), "ValueError", f"{what}: cotangent shape does not match output shape")
        seed_leaves.append(el.cast_array(ca, ol.elt))
      else:
        seed_leaves.append(zero_like_leaf(el, ol))
    seed_tree = unflatten_tree(seed_leaves, out_spec)
    tape.backward(el, out, seed_tree)
    grads = tuple(_ad._adjoint(el, h, a) for h, a in zip(holders, bound_args))
  finally:
    el.ad_live -= 1
    el.tape = outer
    if not el.ad_live:
      el.ad_vars.clear()
      el.ad_bufs.clear()
      el.tainted_vars.clear()
      el.tainted_bufs.clear()
      if not el.jvp_live:
        el.defs.clear()
  return (out, grads)


def hvp_call(el, fn, primals, tangents, argnum=None, what: str = "hvp"):
  """Forward-over-reverse Hessian-vector product ``H(fn)(primals) @ tangents``.

  Uses ``jvp(grad(fn), primals, tangents)`` so cost is ~2x reverse-mode with
  zero ``O(N^2)`` Hessian allocation.
  """
  from . import autodiff as _ad

  single_input = not isinstance(primals, tuple)
  p_tuple = (primals,) if single_input else primals
  t_tuple = (tangents,) if single_input else tangents
  if len(p_tuple) != len(t_tuple):
    raise el.error(
      TypeInferenceError,
      f"{what}: primals ({len(p_tuple)}) and tangents ({len(t_tuple)}) must have the same length",
    )
  if argnum is None:
    if len(p_tuple) == 1:
      wrt = 0
    else:
      wrt = tuple(range(len(p_tuple)))
  else:
    wrt = argnum

  p_bound_in = [el.bindable(a, "primal") for a in p_tuple]
  t_bound_in = [el.bindable(a, "tangent") for a in t_tuple]

  outer = el.jvp
  ctx = JVPContext(outer=outer)
  el.jvp_live += 1
  try:
    bound_args = [ctx.bind_input(el, p, t, what) for p, t in zip(p_bound_in, t_bound_in)]
    el.jvp = ctx
    try:
      grad_out = _ad.differentiate(el, fn, wrt, bound_args, {}, False, "grad")
      _, hvp_out = ctx.extract_primal_and_tangent(el, grad_out)
    finally:
      el.jvp = outer
  finally:
    el.jvp_live -= 1
    el.jvp = outer
    if not el.ad_live and not el.jvp_live:
      el.defs.clear()
  return hvp_out


def _basis_vector(el, shape: tuple[Expr, ...], elt: ScalarType, flat_idx: Expr) -> ArrayVal:
  """Delayed 1-hot basis array of ``shape`` with ``1.0`` at flat index ``flat_idx``."""
  if not shape:
    return el.delayed(elt, (), lambda _: Const(1.0, elt), cost=1)
  strides: list[Expr] = [Const(1, I)] * len(shape)
  for k in range(len(shape) - 2, -1, -1):
    strides[k] = el.tmp(ops.imul(strides[k + 1], shape[k + 1]), "bs")

  def at(idx):
    flat: Expr = ZERO
    for iv, st in zip(idx, strides):
      flat = iadd(flat, ops.imul(iv, st))
    return ops.select(cmp("eq", flat, flat_idx), Const(1.0, elt), Const(0.0, elt))

  return el.delayed(elt, shape, at, cost=2)


def _jacobian_leaf_info(el, x, what: str):
  if isinstance(x, Expr) and is_float(x):
    return (), x.type.strong, Const(1, I), True
  if isinstance(x, ArrayVal) and x.elt.kind == "f":
    return x.shape, x.elt, el.tmp(ops.iprod(x.shape), "n_in"), False
  raise el.error(TypeInferenceError, f"{what} expects a floating-point scalar or array argument, got {describe(x)}")


def jacfwd_call(el, fn, argnum, args: list, kwargs: dict, what: str = "jacfwd"):
  """Forward-mode Jacobian of ``fn`` with respect to positional argument ``argnum``."""
  if kwargs:
    raise el.error(UnsupportedError, f"{what}() does not currently support keyword arguments")
  k = el.static_int(argnum, f"argnum for {what}")
  if not 0 <= k < len(args):
    raise el.error(TypeInferenceError, f"{what}: invalid argnum {k} for {len(args)} arguments")
  args = [el.bindable(a, "arg") for a in args]
  x = args[k]
  in_shape, in_elt, n_in, x_is_scalar = _jacobian_leaf_info(el, x, what)

  # Probe output shape & dtype with a dry JVP
  zero_tangs = tuple(zero_like_tree(el, a) for a in args)
  if x_is_scalar:
    tangs = list(zero_tangs)
    tangs[k] = Const(1.0, in_elt)
    _, dout = jvp_call(el, fn, tuple(args), tuple(tangs), what)
    return dout

  with el.trial():
    tangs0 = list(zero_tangs)
    tangs0[k] = el.materialize(_basis_vector(el, in_shape, in_elt, ZERO), "e0")
    _, sample = jvp_call(el, fn, tuple(args), tuple(tangs0), what)

  if isinstance(sample, Expr) and is_float(sample):
    out_shape, out_elt, out_is_scalar = (), sample.type.strong, True
  elif isinstance(sample, ArrayVal) and sample.elt.kind == "f":
    out_shape, out_elt, out_is_scalar = sample.shape, sample.elt, False
  else:
    raise el.error(TypeInferenceError, f"{what} expects a float scalar or array output, got {describe(sample)}")

  jac_flat = el.alloc_array(out_elt, (*out_shape, n_in), zero=False, label="jacfwd")

  def col_body(j: Expr) -> None:
    tangs = list(zero_tangs)
    tangs[k] = el.materialize(_basis_vector(el, in_shape, in_elt, j), "ej")
    _, col = jvp_call(el, fn, tuple(args), tuple(tangs), what)
    if out_is_scalar:
      el.emit(Store(jac_flat.buf, j, cast(col, out_elt)))
    else:
      # Store col[...] into jac_flat[..., j]
      dest_slice = el.getitem(jac_flat, (*(slice(None),) * len(out_shape), j))
      el.store_array(dest_slice, el.cast_array(col, out_elt))

  el.for_range(ZERO, n_in, col_body, name="j")
  el.record_store(jac_flat)
  return el.reshape(jac_flat, (*out_shape, *in_shape), checked=False)


def jacrev_call(el, fn, argnum, args: list, kwargs: dict, what: str = "jacrev"):
  """Reverse-mode Jacobian of ``fn`` with respect to positional argument ``argnum``."""
  if kwargs:
    raise el.error(UnsupportedError, f"{what}() does not currently support keyword arguments")
  k = el.static_int(argnum, f"argnum for {what}")
  if not 0 <= k < len(args):
    raise el.error(TypeInferenceError, f"{what}: invalid argnum {k} for {len(args)} arguments")
  args = [el.bindable(a, "arg") for a in args]
  x = args[k]
  in_shape, in_elt, _, x_is_scalar = _jacobian_leaf_info(el, x, what)

  # Probe output shape & dtype
  with el.trial():
    sample = el.call(fn, list(args), {})
  if isinstance(sample, Expr) and is_float(sample):
    out_shape, out_elt, n_out, out_is_scalar = (), sample.type.strong, Const(1, I), True
  elif isinstance(sample, ArrayVal) and sample.elt.kind == "f":
    out_shape, out_elt, n_out, out_is_scalar = sample.shape, sample.elt, el.tmp(ops.iprod(sample.shape), "n_out"), False
  else:
    raise el.error(TypeInferenceError, f"{what} expects a float scalar or array output, got {describe(sample)}")

  if out_is_scalar:
    _, grads = vjp_apply(el, fn, tuple(args), Const(1.0, out_elt), what)
    return grads[k]

  jac_flat = el.alloc_array(in_elt, (n_out, *in_shape), zero=False, label="jacrev")

  def row_body(i: Expr) -> None:
    ct = el.materialize(_basis_vector(el, out_shape, out_elt, i), "ei")
    _, grads = vjp_apply(el, fn, tuple(args), ct, what)
    gk = grads[k]
    if x_is_scalar:
      el.emit(Store(jac_flat.buf, i, cast(gk, in_elt)))
    else:
      dest_slice = el.getitem(jac_flat, (i, *(slice(None),) * len(in_shape)))
      el.store_array(dest_slice, el.cast_array(gk, in_elt))

  el.for_range(ZERO, n_out, row_body, name="i")
  el.record_store(jac_flat)
  return el.reshape(jac_flat, (*out_shape, *in_shape), checked=False)


def hessian_call(el, fn, argnum, args: list, kwargs: dict, what: str = "hessian"):
  """Exact Hessian matrix/tensor via forward-over-reverse ``jacfwd(grad(fn))``."""
  if kwargs:
    raise el.error(UnsupportedError, f"{what}() does not currently support keyword arguments")
  k = el.static_int(argnum, f"argnum for {what}")
  if not 0 <= k < len(args):
    raise el.error(TypeInferenceError, f"{what}: invalid argnum {k} for {len(args)} arguments")
  args = [el.bindable(a, "arg") for a in args]
  x = args[k]
  in_shape, in_elt, n_in, x_is_scalar = _jacobian_leaf_info(el, x, what)
  zero_tangs = tuple(zero_like_tree(el, a) for a in args)

  if x_is_scalar:
    tangs = list(zero_tangs)
    tangs[k] = Const(1.0, in_elt)
    return hvp_call(el, fn, tuple(args), tuple(tangs), argnum=k, what=what)

  hess_flat = el.alloc_array(in_elt, (*in_shape, n_in), zero=False, label="hess")

  def col_body(j: Expr) -> None:
    tangs = list(zero_tangs)
    tangs[k] = el.materialize(_basis_vector(el, in_shape, in_elt, j), "ej")
    hv = hvp_call(el, fn, tuple(args), tuple(tangs), argnum=k, what=what)
    dest_slice = el.getitem(hess_flat, (*(slice(None),) * len(in_shape), j))
    el.store_array(dest_slice, el.cast_array(hv, in_elt))

  el.for_range(ZERO, n_in, col_body, name="j")
  el.record_store(hess_flat)
  return el.reshape(hess_flat, (*in_shape, *in_shape), checked=False)


__all__ = [
  "JVPContext",
  "custom_jvp_call",
  "hessian_call",
  "hvp_call",
  "jacfwd_call",
  "jacrev_call",
  "jvp_call",
  "vjp_apply",
]
