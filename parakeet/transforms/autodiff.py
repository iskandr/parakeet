"""Reverse-mode differentiation of the array operators.

``grad(f)`` is an adverb like ``map`` or ``reduce``: it is applied while the
calling function is elaborated, and what it produces is ordinary loop IR that
is fused, tiled and parallelised with the rest.

While ``f`` is elaborated a :class:`Tape` records each array operator applied
to a value that depends on the differentiated arguments.  The record is made
where the operator still has its algebraic meaning, so every rule below is
one line of algebra:

===============  ==========================================================
element-wise     a map of the scalar function's partial derivative
                 (:mod:`parakeet.transforms.derivatives`), summed over the
                 dimensions the argument was broadcast along
reduce           ``add``: the adjoint broadcast back; ``min``/``max``: sent
                 to the winning elements; ``multiply``: times the product of
                 the others
matmul           two more matmuls
map, allpairs    a loop over the same index space that differentiates the
                 element function again, one element at a time
views            none: a slice of an array is the same slice of its adjoint
===============  ==========================================================

Scalars are not recorded at all.  An IR expression is its own tape, and a
variable bound to one remembers its definition, so the adjoint of a scalar
is pushed straight down its expression tree.

The tape is static but the program is not: a loop body runs many times, a
branch maybe never.  An operator applied to a differentiated value anywhere
but in straight-line code therefore asks for the enclosing loop to be
unrolled (possible when its trip count is a compile-time constant) and is an
error otherwise.  Anything else that could carry a derivative past the tape
unseen - a variable merged after a branch, an element stored into an array -
is *tainted*, and meeting a tainted value during differentiation is an error
too.  The gradient is right or there is none.
"""

from __future__ import annotations

import ast
import contextlib
import copy
from typing import Any

from .. import prims
from ..errors import InternalError, TypeInferenceError, UnsupportedError
from ..ndtypes import Bool, BufType, ScalarType
from ..syntax import ops
from ..syntax.ir import (
  Assign,
  Cast,
  Const,
  Expr,
  IndexReduce,
  IndexScan,
  Load,
  Prim,
  Store,
  Var,
  same,
  walk_expr,
  walk_stmts,
)
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, isub
from ..syntax.records import record_type
from ..syntax.values import CHEAP_COST, ArrayVal, describe
from .derivatives import backprop, is_float

MAX_UNROLL = 4096
UF = prims.UFUNCS


# ---------------------------------------------------------------------------
# what depends on a differentiated value
# ---------------------------------------------------------------------------


def depends(el, e: Expr) -> bool:
  """Does the float expression ``e`` mention a value under differentiation?"""
  stack = [e]
  while stack:
    e = stack.pop()
    if isinstance(e, Const) or not is_float(e):
      continue
    if isinstance(e, Var):
      if e.uid in el.ad_vars:
        return True
    elif isinstance(e, Load):
      if e.buf.uid in el.ad_bufs:
        return True
    elif isinstance(e, Prim):
      stack.extend(e.args)
    elif isinstance(e, Cast):
      stack.append(e.arg)
  return False


def note_def(el, v: Var, e: Expr) -> None:
  """``v`` was just bound to the pure expression ``e``."""
  if is_float(e):
    el.defs[v.uid] = e
    if depends(el, e):
      el.ad_vars.add(v.uid)


def observe(el, stmt) -> None:
  """Taint whatever a statement the tape does not know about writes to."""
  for s in walk_stmts([stmt]):
    if isinstance(s, Assign):
      t = s.var.type
      if isinstance(t, BufType):
        if isinstance(s.value, Var) and s.value.uid in el.ad_bufs:
          el.ad_bufs.add(s.var.uid)
          el.tainted_bufs.setdefault(s.var.uid, _info(el, "an array chosen at run time (after a branch or in a loop)"))
      elif t.kind == "f" and s.var.uid not in el.defs and depends(el, s.value):
        el.ad_vars.add(s.var.uid)
        el.tainted_vars.setdefault(s.var.uid, _info(el, "a variable assigned in a branch or a loop"))
    elif isinstance(s, IndexReduce):
      for f in s.folds:
        if is_float(f.result) and (depends(el, f.value) or (f.init is not None and depends(el, f.init))):
          el.ad_vars.add(f.result.uid)
          el.tainted_vars.setdefault(f.result.uid, _info(el, "a reduction"))
    elif isinstance(s, IndexScan):
      f = s.fold
      if is_float(f.result) and (depends(el, f.value) or (f.init is not None and depends(el, f.init))):
        for st in s.emit:
          if isinstance(st, Store):
            el.ad_bufs.add(st.buf.uid)
            el.tainted_bufs.setdefault(st.buf.uid, _info(el, "a scan"))
    elif isinstance(s, Store):
      if is_float(s.value) and depends(el, s.value):
        el.ad_bufs.add(s.buf.uid)
        el.tainted_bufs.setdefault(s.buf.uid, _info(el, "an array whose elements were assigned one by one"))


def _info(el, what: str) -> tuple:
  return (el.loc(), what, el.blame(), el.if_stack[-1] if el.if_stack else None)


def _tainted(el, info) -> UnsupportedError:
  loc, what, blamed, branch = info
  if branch is not None:
    el.request_both(*branch)  # start over running both branches, if this one allows it
  where = f" (line {loc.lineno})" if loc is not None and getattr(loc, "lineno", None) else ""
  if blamed is not None:
    return el.error(
      UnsupportedError, f"cannot differentiate through {blamed}(){where}: its derivative is not implemented"
    )
  return el.error(
    UnsupportedError,
    f"cannot differentiate through {what}{where}",
    hint="build the value with array operations instead: np.where or a conditional expression for a choice, "
    "a comprehension, parakeet.map or a NumPy function for an array, parakeet.reduce or np.sum for a total",
  )


# ---------------------------------------------------------------------------
# where adjoints accumulate
# ---------------------------------------------------------------------------


class ScalarAdj:
  """The adjoint of a scalar variable."""

  def __init__(self, tape: Tape, var: Var, source: Node | None = None):
    self.tape = tape
    self.var = var
    self.source = source
    self.parts: list[Expr] = []
    self.cell: Var | None = None  # a one-element buffer, for adjoints added inside loops
    self.needs_cell = False
    self.redirect: tuple | None = None  # (buffer, index): this chunk's own cell in a parallel loop

  def add(self, el, c: Expr, via: Tape) -> None:
    if via.dry and self.tape is not via:
      self.needs_cell = True
      via.touched.add(self)
      return
    c = cast(c, self.var.type)
    if self.redirect is not None:
      buf, at = self.redirect
      el.emit(Store(buf, at, ops.prim("add", (Load(buf, at), cast(c, buf.type.elt)), buf.type.elt)))
      return
    if el.block is self.tape.home:
      self.parts.append(el.tmp(c, "adj"))
      return
    if self.cell is None:
      raise InternalError("adjoint of a scalar is needed inside a loop that was not anticipated")
    el.emit(
      Store(self.cell, ZERO, ops.prim("add", (Load(self.cell, ZERO), cast(c, self.cell.type.elt)), self.cell.type.elt))
    )

  def allocate(self, el) -> None:
    if self.needs_cell and self.cell is None:
      arr = el.alloc_array(self.var.type.strong, (ONE,), zero=True, label="adj")
      self.cell = arr.buf

  def total(self, el) -> Expr | None:
    t = self.var.type
    total: Expr | None = None
    for p in self.parts:
      total = p if total is None else ops.prim("add", (total, p), t)
    if self.cell is not None:
      c = cast(Load(self.cell, ZERO), t)
      total = c if total is None else ops.prim("add", (total, c), t)
    return None if total is None else el.tmp(total, "adj")


class RootAdj:
  """The adjoint of a dense array.  Views of the array are the same views of
  the adjoint, so slicing, transposing and reshaping need no rules."""

  def __init__(self, tape: Tape, arr: ArrayVal, source: Node | None = None):
    self.tape = tape
    self.arr = arr
    self.source = source
    self.whole: ArrayVal | None = None  # sum of the contributions covering the whole array
    self.buf: ArrayVal | None = None  # zero-initialised accumulator, when views or loops need one
    self.needs_buffer = False
    self.stored: Any = None
    self.mutable = False  # stored into by recorded operators: adjoints live in the buffer only
    self.redirect: ArrayVal | None = None  # this chunk's own copy of the accumulator in a parallel loop

  def is_whole(self, el, v: ArrayVal) -> bool:
    a = self.arr
    return v is a or (
      v.rank == a.rank
      and same(v.offset, a.offset)
      and all(el.dim_equal(x, y) for x, y in zip(v.shape, a.shape))
      and all(same(x, y) for x, y in zip(v.strides, a.strides))
    )

  def allocate(self, el) -> None:
    if self.buf is None:
      self.buf = el.alloc_array(self.arr.elt, self.arr.shape, zero=True, label="adj")

  def _ensure(self, el) -> ArrayVal:
    if self.buf is None:
      if el.block is not self.tape.home:
        raise InternalError("adjoint of an array is needed inside a loop that was not anticipated")
      self.allocate(el)
    return self.buf  # type: ignore[return-value]

  def target(self, el) -> ArrayVal:
    """Where adjoints are being accumulated right now."""
    return self.redirect if self.redirect is not None else self._ensure(el)

  def view(self, el, v: ArrayVal) -> ArrayVal:
    buf = self.target(el)
    out = ArrayVal(
      self.arr.elt, v.shape, buf=buf.buf, offset=el.tmp(iadd(buf.offset, isub(v.offset, self.arr.offset)), "off"),
      strides=v.strides, roots=buf.roots, contig=False,
    )  # fmt: skip
    out.inj = v.inj
    out.base = buf if buf.contig else buf.base
    return out

  def add(self, el, v: ArrayVal, g, via: Tape) -> None:
    if self.stored is not None:
      raise _tainted(el, self.stored)
    if via.dry and self.tape is not via:
      self.needs_buffer = True
      via.touched.add(self)
      return
    at_home = el.block is self.tape.home
    if self.buf is None and at_home and not self.mutable and self.is_whole(el, v) and isinstance(g, ArrayVal):
      g = settled(el, el.cast_array(g, self.arr.elt))
      self.whole = g if self.whole is None else el.apply_ufunc(UF["add"], [self.whole, g])
      return
    accumulate(el, self.view(el, v), g)

  def scatter(self, el, pos: Expr, c: Expr, via: Tape) -> None:
    if self.stored is not None:
      raise _tainted(el, self.stored)
    if via.dry and self.tape is not via:
      self.needs_buffer = True
      via.touched.add(self)
      return
    buf = self.target(el)
    elt = self.arr.elt
    p = el.tmp(iadd(buf.offset, isub(pos, self.arr.offset)), "pos")
    store_element(el, buf, p, ops.prim("add", (Load(buf.buf, p), cast(c, elt)), elt))

  def total(self, el) -> ArrayVal | None:
    if self.stored is not None:
      raise _tainted(el, self.stored)
    if self.buf is None:
      return self.whole
    if self.whole is not None:
      accumulate(el, self.buf, self.whole)
      self.whole = None
    return self.buf


def settled(el, g: ArrayVal) -> ArrayVal:
  """An adjoint that will be held on to.  Cheap expressions stay delayed and
  fuse into whatever uses them; anything expensive is evaluated now, before
  a store into some adjoint buffer could make it stale."""
  if g.buf is None and g.cost > CHEAP_COST:
    return el.as_concrete(g)
  return g


def store_element(el, arr: ArrayVal, pos: Expr, v: Expr) -> None:
  """``arr.buf[pos] = v``, recorded if a tape is listening."""
  tape = el.tape
  if tape is not None and not tape.muted and tape.store_element(el, arr, pos, v):
    return
  el.emit(Store(arr.buf, pos, v))
  el.record_store(arr)


def accumulate(el, dest: ArrayVal, g) -> None:
  """``dest += g`` for a view of an adjoint buffer."""
  elt = dest.elt
  if el.tape is not None:
    # the gradient being built is itself differentiated: an ordinary, recorded update
    el.store_array(dest, el.apply_ufunc(UF["add"], [dest, g]))
    return
  if isinstance(g, ArrayVal):
    if g.buf is None and g.cost > CHEAP_COST:
      g = el.as_concrete(g)
    el.consume(g)
    g = el.broadcast_to(g, dest.shape, consumed=True)

  def body(idx):
    pos = el.tmp(el.index_expr(dest, idx), "pos")
    x = el.elem(g, idx) if isinstance(g, ArrayVal) else g
    el.emit(Store(dest.buf, pos, ops.prim("add", (Load(dest.buf, pos), cast(x, elt)), elt)))

  if not isinstance(g, ArrayVal):
    g = el.tmp(g, "g")
  # two indices of a view that broadcasts (or repeats) hit the same element
  el.loop_nest(dest.shape, body, parallel=dest.inj)
  el.record_store(dest)


# ---------------------------------------------------------------------------
# the tape
# ---------------------------------------------------------------------------


class Node:
  """One recorded operator application."""

  __slots__ = ("kind", "inputs", "out", "info", "g", "holder", "tick", "origin")

  def __init__(self, kind: str, inputs: tuple, info: Any = None):
    self.kind = kind
    self.inputs = inputs
    self.info = info
    self.out: Any = None
    self.tick = 0
    self.origin: Any = None  # stands in for an AST node as the origin of arrays its rule creates
    self.g: ArrayVal | None = None  # adjoint of a delayed result
    self.holder: RootAdj | ScalarAdj | None = None  # adjoint of a concrete or scalar result


class Tape:
  def __init__(self, el, parent: Tape | None = None, dry: bool = False, key: tuple = (), outer: Tape | None = None):
    self.parent = parent
    self.dry = dry
    # the tape of a gradient this whole computation is itself being differentiated by (grad of grad)
    self.outer = outer
    self.key = key  # identifies this tape across restarts of the elaboration
    self.sites: dict[tuple, int] = {}
    self.block = el.block  # operators are only recorded in this block: straight-line code
    self.if_depth = len(el.if_stack)
    self.home: list | None = None  # where the backward pass is emitted
    self.outer_loops = {id(ctx) for fr in el.frames for ctx in fr.loops}
    self.nodes: list[Node] = []
    self.roots: dict[int, RootAdj] = {}
    self.scalars: dict[int, ScalarAdj] = {}
    self.delayed: dict[int, Node] = {}
    self.known: dict[int, bool] = {}
    self.keep: list = []
    self.muted = 0
    self.touched: set = set()  # (a dry run) the adjoints of enclosing tapes it would add to
    self.tick = 0  # orders recorded operators, allocations and in-place stores
    self.born: dict = {}  # root -> tick of its allocation (roots not listed are older than the tape)
    self.stored_at: dict = {}  # root -> tick of the last store into it
    self.unknown_store = -1  # tick of the last store into an array whose origin is unknown

  @contextlib.contextmanager
  def computing(self, el):
    """While a recorded operator computes its result this tape steps aside.
    If the gradient being built is itself being differentiated, the enclosing
    tape records that computation instead."""
    if self.outer is None:
      self.muted += 1
      try:
        yield
      finally:
        self.muted -= 1
    else:
      el.tape = self.outer
      try:
        yield
      finally:
        el.tape = self

  def first_order(self, el, what: str) -> None:
    if self.outer is not None:
      raise el.error(UnsupportedError, f"second derivatives through {what} are not supported")

  # ---- lookup --------------------------------------------------------------------

  def find_root(self, uid: int) -> RootAdj | None:
    t: Tape | None = self
    while t is not None:
      r = t.roots.get(uid)
      if r is not None:
        return r
      t = t.parent
    return None

  def find_scalar(self, uid: int) -> ScalarAdj | None:
    t: Tape | None = self
    while t is not None:
      s = t.scalars.get(uid)
      if s is not None:
        return s
      t = t.parent
    return None

  def find_delayed(self, arr: ArrayVal) -> tuple[Tape, Node] | None:
    t: Tape | None = self
    while t is not None:
      n = t.delayed.get(id(arr))
      if n is not None:
        return t, n
      t = t.parent
    return None

  # ---- activity --------------------------------------------------------------------

  def is_active(self, el, v) -> bool:
    """Does ``v`` carry a derivative?  Raises if it does but the tape lost track of it."""
    if isinstance(v, Expr):
      return is_float(v) and depends(el, v)
    if record := record_type(v):
      return self.is_active(el, record.values(v))
    if isinstance(v, tuple):
      return any(self.is_active(el, x) for x in v)
    if not isinstance(v, ArrayVal) or v.elt.kind != "f":
      return False
    if v.buf is not None:
      uid = v.buf.uid
      if uid not in el.ad_bufs:
        return False
      if uid in el.tainted_bufs:
        raise _tainted(el, el.tainted_bufs[uid])
      root = self.find_root(uid)
      if root is None:
        if self.outer is not None:
          return False  # it matters to the enclosing gradient only: a constant here
        raise InternalError("an array depends on a differentiated value but has no adjoint")
      if root.stored is not None:
        raise _tainted(el, root.stored)
      return True
    if self.find_delayed(v) is not None:
      return True
    return self.classify(el, v)

  def classify(self, el, arr: ArrayVal) -> bool:
    """Activity of a delayed array no rule produced: evaluate one element."""
    hit = self.known.get(id(arr))
    if hit is not None:
      return hit
    probe = tuple(Var("i", I) for _ in arr.shape)
    for p, d in zip(probe, arr.shape):
      el.nonneg.add(p.uid)
      el.upper[p.uid] = d
    self.muted += 1
    el.loop_depth += 1
    try:
      with el.trial():
        r = arr.fn(probe)
        active = depends(el, r)
        if active:
          # the element must be a pure expression of differentiated values
          self.push(el, r, Const(1.0, r.type), dry=True)
    finally:
      el.loop_depth -= 1
      self.muted -= 1
    self.known[id(arr)] = active
    self.keep.append(arr)
    if active:
      self.first_order(el, "this indexing or library function")
      node = Node("opaque", (arr,))
      self.record(el, node)
      self.delayed[id(arr)] = node
      node.out = arr
    return active

  # ---- recording ---------------------------------------------------------------------

  def record(self, el, node: Node) -> None:
    if el.block is not self.block:
      self.not_static(el)
    self.tick += 1
    node.tick = self.tick
    self.nodes.append(node)
    # Arrays the rule of this operator creates need an origin that is the same
    # object every time the function is elaborated: the n-th operator of its
    # kind recorded while evaluating a given piece of source
    site = el.node_stack[-1] if el.node_stack else None
    where = (id(site), node.kind)
    n = self.sites[where] = self.sites.get(where, 0) + 1
    key = (*self.key, id(site), node.kind, n)
    origin = el.state.tokens.get(key)
    if origin is None:
      origin = el.state.tokens[key] = copy.copy(site) if isinstance(site, ast.AST) else ast.Pass()
      el.state.keep.append(site)
    node.origin = origin

  def not_static(self, el) -> None:
    if len(el.if_stack) > self.if_depth:
      el.request_both(*el.if_stack[-1])
    for fr in reversed(el.frames):
      if fr.loops and id(fr.loops[-1]) not in self.outer_loops:
        el.request_unroll(fr, fr.loops[-1].node)
    message = "cannot differentiate an array operation that only runs conditionally"
    hint = (
      "an if statement can be differentiated when its branches only assign variables (no return, raise, "
      "loop exit or stores into arrays); otherwise use np.where or a conditional expression"
    )
    if el.if_stack and not el.if_stack[-1][0].globals.get("__parakeet_lib__"):
      raise el.error(UnsupportedError, message, el.if_stack[-1][1], hint)  # the user's own if statement
    raise el.no_derivative(message, hint=hint)

  def register(self, el, node: Node, out) -> None:
    """Make ``out`` the result of ``node``."""
    node.out = out
    if isinstance(out, Expr):
      if isinstance(out, Var):
        node.holder = self.scalars[out.uid] = ScalarAdj(self, out, node)
        el.ad_vars.add(out.uid)
        el.tainted_vars.pop(out.uid, None)
    elif out.buf is not None:
      node.holder = self.root(el, out, node)
    else:
      self.delayed[id(out)] = node
      self.keep.append(out)

  def root(self, el, arr: ArrayVal, source: Node | None = None) -> RootAdj:
    r = self.roots[arr.buf.uid] = RootAdj(self, arr, source)
    el.ad_bufs.add(arr.buf.uid)
    el.tainted_bufs.pop(arr.buf.uid, None)
    return r

  def leaf_scalar(self, el, v: Var) -> ScalarAdj:
    s = self.scalars[v.uid] = ScalarAdj(self, v)
    el.ad_vars.add(v.uid)
    return s

  # ---- hooks called by the frontend ------------------------------------------------------

  def elementwise(self, el, args: list, active: list[bool], fn, elt: ScalarType, cost: int):
    # the backward pass evaluates the arguments again: they must be cheap to re-read
    args = [el.as_concrete(a) if isinstance(a, ArrayVal) and a.buf is None and a.cost > CHEAP_COST else a for a in args]
    args = [a if isinstance(a, ArrayVal) else el.tmp(a) for a in args]
    with self.computing(el):
      out = el.elementwise(args, fn, elt, cost)
    node = Node("elementwise", tuple(args), (tuple(active), fn, elt, cost))
    self.record(el, node)
    self.register(el, node, out)
    return out

  def materialized(self, el, arr: ArrayVal, out: ArrayVal, info=None) -> None:
    """``out`` is a fresh copy of ``arr``."""
    info = info or el.tainted_bufs.get(out.buf.uid)
    if arr.elt.kind != "f" or info is None:
      return  # nothing that was copied depends on a differentiated value
    if self.outer is not None:
      self.outer.materialized(el, arr, out, info)
    if not self.is_active(el, arr):
      if self.outer is not None:
        return
      raise _tainted(el, info)
    node = Node("alias", (arr,))
    self.record(el, node)
    self.register(el, node, out)

  def reduce(self, el, plain, op: str, a: ArrayVal, axis, init, what: str, keepdims):
    if op != "add" and a.buf is None and a.cost > CHEAP_COST:
      a = el.as_concrete(a)
    if init is not None:
      init = el.tmp(init)
    from ..lib.numpy_lib import _reduce_prim

    with self.computing(el):
      out = _reduce_prim(el, op, a, axis, init, what, keepdims)
      if isinstance(out, ArrayVal):
        out = el.as_concrete(out)
    if not isinstance(keepdims, bool):
      keepdims = el.static_bool(keepdims, "keepdims")
    active_a = self.is_active(el, a)
    active_init = init is not None and self.is_active(el, init)
    inputs = (a, init) if init is not None else (a,)
    node = Node("reduce", inputs, (op, axis, keepdims, active_a, active_init))
    self.record(el, node)
    self.register(el, node, out)
    return out

  def matmul(self, el, plain, a: ArrayVal, b: ArrayVal, what: str):
    from ..lib.numpy_lib import matmul

    with self.computing(el):
      out = matmul(el, a, b, what)
      if isinstance(out, ArrayVal):
        out = el.as_concrete(out)
    node = Node("matmul", (a, b), (self.is_active(el, a), self.is_active(el, b)))
    self.record(el, node)
    self.register(el, node, out)
    return out

  def gather(self, el, arr: ArrayVal, items: list):
    """Record fancy-indexing gather ``arr[idx...]`` so backward emits an
    ``IndexScatter`` (parallel GPU ``atomicAdd`` / privatized CPU scatter)
    instead of falling back to an opaque sequential loop."""
    if el.block is not self.block:
      self.not_static(el)
    arr = el.as_concrete(arr)
    index_arrays = tuple(it for it in items if isinstance(it, ArrayVal))
    with self.computing(el):
      shape, src_index, base = el._fancy_plan(arr, items)
      if base.buf is None:
        return None
      index_roots = [el.get_roots(it) for it in index_arrays]
      out = el.delayed(
        base.elt,
        shape,
        lambda idx: el.elem(base, src_index(idx)),
        roots=el.union_roots(el.get_roots(base), *index_roots),
        cost=max(base.cost + 2, CHEAP_COST + 1),
      )
    node = Node("gather", (base,), (shape, src_index, index_arrays))
    self.record(el, node)
    self.register(el, node, out)
    return out

  def op(self, el, kind: str, inputs: tuple, info, compute):
    """Record an operator whose result ``compute()`` builds without help."""
    self.first_order(el, kind)
    self.muted += 1
    try:
      out = compute()
      if isinstance(out, ArrayVal):
        out = el.as_concrete(out)
    finally:
      self.muted -= 1
    node = Node(kind, tuple(inputs), info)
    self.record(el, node)
    self.register(el, node, out)
    return out

  def fold(self, el, combine, x: ArrayVal, init, what: str):
    """A fold with an arbitrary function.  Its derivative needs every running
    value, so the fold is computed as a scan and replayed backwards."""
    from ..lib.core import scan_values

    self.first_order(el, "a fold with a function of your own")
    if x.rank != 1:
      x = el.flat_view(x)
    if x.buf is None and x.cost > CHEAP_COST:
      x = el.as_concrete(x)
    n = el.tmp(x.shape[0], "n")
    context = el.capture()

    def step(acc, xi):
      with el.restored(context):
        return el.as_scalar(combine(acc, xi), "reduction result")

    self.muted += 1
    try:
      if init is None:
        el.check(cmp("gt", n, 0), "ValueError", f"zero-size array to {what}, which has no identity")
      accs = el.as_concrete(scan_values(el, n, lambda i: el.tmp(el.elem(x, (i,)), "x"), combine, init, what))
      out = Var("red", accs.elt)
      el.mutables.add(out.uid)
      if init is None:
        el.emit(Assign(out, Load(accs.buf, isub(n, 1))))
      else:
        from ..syntax.ir import If

        el.emit(Assign(out, cast(init, accs.elt)))
        el.emit(If(cmp("gt", n, 0), [Assign(out, Load(accs.buf, isub(n, 1)))], []))
    finally:
      self.muted -= 1
    info = (accs, init, step, n)
    # what the backward loop will add to, found out now
    probe = Var("i", I)
    el.nonneg.add(probe.uid)
    el.upper[probe.uid] = n
    el.loop_depth += 1
    try:
      with el.trial():
        cell = el.alloc_array(accs.elt, (ONE,), zero=True, label="adj")
        _fold_step(self, el, x, info, probe, cell, dry=True, key="probe")
    finally:
      el.loop_depth -= 1
    node = Node("fold", (x,), info)
    self.record(el, node)
    self.register(el, node, out)
    return out

  def detach(self, el, arr: ArrayVal) -> ArrayVal:
    """``arr`` is about to be turned into a concrete array in place.  If it is
    the delayed result of a recorded operator, hand that role to a copy of
    the delayed value, which is what gets evaluated."""
    node = self.delayed.pop(id(arr), None)
    if node is None:
      return arr
    self.first_order(el, "an in-place update of an expression")
    ghost = ArrayVal(arr.elt, arr.shape, fn=arr.fn, roots=arr.roots, cost=arr.cost, reads=arr.reads)
    ghost.depth, ghost.epoch, ghost.block, ghost.origin, ghost.uses = (
      arr.depth,
      arr.epoch,
      arr.block,
      arr.origin,
      arr.uses,
    )
    self.delayed[id(ghost)] = node
    self.keep.append(ghost)
    node.out = ghost
    return ghost

  def _mutable_root(self, el, dest: ArrayVal, active: bool) -> RootAdj | None:
    """The adjoint of the array ``dest`` is a view of, if a store into it can be recorded."""
    uid = dest.buf.uid
    if uid in el.tainted_bufs:
      raise _tainted(el, el.tainted_bufs[uid])
    root = self.find_root(uid)
    if root is None:
      if not active:
        return None  # nothing differentiated is involved
      base = dest if dest.contig else dest.base
      if base is None or base.buf is not dest.buf:
        return None  # not a view of a whole array we know: the store taints it instead
      root = self.root(el, base)
    if root.tape is not self:
      raise el.no_derivative(
        "cannot differentiate a store into an array from outside the function being mapped",
        hint="return the values and let the map build the array",
      )
    root.mutable = root.needs_buffer = True
    return root

  def store(self, el, dest: ArrayVal, value) -> bool:
    """``dest[...] = value``.  Its adjoint: what arrives for the stored region
    belongs to ``value``, and nothing of it to what was there before."""
    active = self.is_active(el, value)
    root = self._mutable_root(el, dest, active)
    if root is None:
      return False
    with self.computing(el):
      el.store_array(dest, value)
    el.tainted_bufs.pop(dest.buf.uid, None)
    self.record(el, Node("store", (value,), (dest, root, active)))
    return True

  def store_element(self, el, arr: ArrayVal, pos: Expr, v: Expr) -> bool:
    active = self.is_active(el, v)
    root = self._mutable_root(el, arr, active)
    if root is None:
      return False
    with self.computing(el):
      store_element(el, arr, pos, v)
    el.tainted_bufs.pop(arr.buf.uid, None)
    self.record(el, Node("store_element", (v,), (pos, root, active)))
    return True

  def pack(self, el, parts: list[tuple[ArrayVal, Any]], out: ArrayVal) -> None:
    """``out`` was assembled from ``parts``: (view of out, value stored there)."""
    live = [(view, v) for view, v in parts if self.is_active(el, v)]
    if not live:
      return
    self.first_order(el, "building an array from parts")
    node = Node("pack", tuple(v for _, v in live), [view for view, _ in live])
    self.record(el, node)
    self.register(el, node, out)

  def map(self, el, plain, shape, fn, what: str):
    """A map whose element function is user code: None if it carries no derivative."""
    from ..lib.core import sub_view

    shape = tuple(el.tmp(d, "n") for d in shape)
    context = el.capture()

    def at(idx):
      with el.restored(context):
        return fn(idx)

    probe = tuple(Var("i", I) for _ in shape)
    for p, d in zip(probe, shape):
      el.nonneg.add(p.uid)
      el.upper[p.uid] = d
    el.loop_depth += 1
    try:
      with el.trial():
        child = Tape(el, self, dry=True, key=(*self.key, id(el.node_stack[-1]) if el.node_stack else 0, "probe"))
        el.tape = child
        try:
          r = at(probe)
        finally:
          el.tape = self
        if isinstance(r, tuple):
          return None  # the plain map reports the error
        active = child.is_active(el, r)
        if active:
          if isinstance(r, ArrayVal):
            seed: Any = el.alloc_array(r.elt, r.shape, label="seed")
          else:
            seed = Var("seed", r.type)
          el.tape = None
          try:
            child.backward(el, r, seed)
          finally:
            el.tape = self
    finally:
      el.loop_depth -= 1
    if not active:
      return None
    if self.dry:
      for t in child.touched:
        if t.tape is not self:
          self.touched.add(t)
    self.first_order(el, f"{what} with a function of your own")
    el.tape = None
    try:
      out = el.as_concrete(el.as_array(plain(el, shape, at, what)))
    finally:
      el.tape = self
    node = Node("map", (), (shape, at, sub_view, list(child.touched)))
    self.record(el, node)
    self.register(el, node, out)
    return out

  def note_alloc(self, root) -> None:
    self.tick += 1
    self.born[root] = self.tick

  def note_store(self, el, arr: ArrayVal) -> None:
    """Part of ``arr`` was overwritten by code the tape did not record."""
    self.tick += 1
    roots = el.get_roots(arr)
    if roots is None and arr.buf is not None:
      roots = el.reach.get(arr.buf.uid)  # a merged variable: what it may refer to
    if roots is None:
      self.unknown_store = self.tick
    else:
      for r in roots:
        self.stored_at[r] = self.tick
    if arr.buf is None:
      return
    root = self.find_root(arr.buf.uid)
    if root is not None and root.stored is None and not self.muted:
      root.stored = _info(el, "an array that is modified in place after it was computed")

  # ---- the backward pass --------------------------------------------------------------------

  def add(self, el, v, g) -> None:
    """Add ``g`` to the adjoint of ``v``."""
    if isinstance(v, Expr):
      if is_float(v) and depends(el, v):
        self.push(el, v, el.as_scalar(g, "adjoint") if isinstance(g, ArrayVal) else g)
      return
    if not isinstance(v, ArrayVal) or v.elt.kind != "f":
      return
    if v.buf is not None:
      if v.buf.uid in el.tainted_bufs:
        raise _tainted(el, el.tainted_bufs[v.buf.uid])
      root = self.find_root(v.buf.uid)
      if root is not None:
        root.add(el, v, g, self)
      return
    found = self.find_delayed(v)
    if found is None:
      return
    owner, node = found
    if owner is not self or el.block is not self.home:
      if self.dry:
        return
      raise InternalError("adjoint of a delayed array is needed outside the tape that owns it")
    g = settled(el, el.cast_array(el.as_array(g), v.elt))
    node.g = g if node.g is None else el.apply_ufunc(UF["add"], [node.g, g])

  def push(self, el, e: Expr, seed: Expr, dry: bool = False) -> None:
    """Send the adjoint ``seed`` of the scalar ``e`` down to what it was computed from."""
    via = _DRY if dry else self

    def on_var(v: Var, adj: Expr):
      s = self.find_scalar(v.uid)
      if s is not None:
        s.add(el, adj, via)
        return None
      if v.uid in el.tainted_vars:
        raise _tainted(el, el.tainted_vars[v.uid])
      return el.defs.get(v.uid)

    def on_load(load: Load, adj: Expr) -> None:
      uid = load.buf.uid
      if uid in el.tainted_bufs:
        raise _tainted(el, el.tainted_bufs[uid])
      root = self.find_root(uid)
      if root is not None:
        root.scatter(el, load.index, adj, via)

    backprop(el, e, seed, on_var, on_load, live=lambda x: depends(el, x))

  def backward(self, el, out, seed) -> None:
    """Emit the adjoint computation of everything recorded, given ``d out = seed``."""
    from .algebraic_vjp import optimize_tape

    self.home = el.block
    optimize_tape(self, el, out)
    for r in list(self.roots.values()):
      if r.needs_buffer:
        r.allocate(el)
    for s in list(self.scalars.values()):
      s.allocate(el)
    self.add(el, out, seed)
    for node in reversed(self.nodes):
      holder = node.holder
      if holder is not None:
        g = holder.total(el)
      else:
        g = node.g
      if g is None and node.kind not in ("store", "store_element"):
        continue
      el.node_stack.append(node.origin)
      try:
        _BACKWARD[node.kind](self, el, node, g)
      finally:
        el.node_stack.pop()

  def stale(self, el, node: Node, v=None, roots_known: bool = False) -> None:
    """The backward pass is about to read again what ``node`` read (the array
    ``v``, or anything at all): it must still hold what the forward pass saw."""
    blind = self.unknown_store > node.tick  # a store through a variable of unknown origin
    if v is None:
      # anything that existed when the operator ran
      changed = blind or any(t > node.tick and self.born.get(r, 0) < node.tick for r, t in self.stored_at.items())
    elif isinstance(v, ArrayVal) and (v.buf is not None or roots_known):
      roots = el.get_roots(v)
      if roots is None:
        changed = blind or any(t > node.tick for t in self.stored_at.values())
      else:
        changed = blind or any(t > node.tick and el.may_alias(roots, {r}) for r, t in self.stored_at.items())
    else:
      changed = False  # delayed arrays are checked (and re-evaluated eagerly) by the frontend itself
    if changed:
      raise el.error(
        UnsupportedError,
        "cannot differentiate: an array is modified in place after a differentiated computation used it",
        hint="assign the modified array to a new variable instead of updating it in place",
      )


class _Dry:
  """Stands in for a tape when a push only has to find out what it would touch."""

  dry = True
  touched: set = set()


_DRY: Any = _Dry()


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------


def unbroadcast(el, g: ArrayVal, target):
  """Sum ``g`` over the dimensions along which ``target`` was broadcast to it."""
  from ..lib.numpy_lib import _reduce_prim as reduce_plain

  zero = Const(0, g.elt)
  if not isinstance(target, ArrayVal) or target.rank == 0:
    return g if g.rank == 0 else reduce_plain(el, "add", g, None, zero, "sum", False)
  lead = g.rank - target.rank
  axes = list(range(lead))
  for k, d in enumerate(target.shape):
    if el.dim_equal(d, g.shape[k + lead]):
      continue
    if isinstance(d, Const) and d.value == 1:
      axes.append(k + lead)
      continue
    # whether this dimension was stretched is only known at run time
    out = el.alloc_array(g.elt, target.shape, zero=True, label="adj")
    accumulate(el, el.broadcast_to(out, g.shape), g)
    return out
  if not axes:
    return g
  if lead:
    g = reduce_plain(el, "add", g, tuple(range(lead)), zero, "sum", False)
    axes = [a - lead for a in axes if a >= lead]
  if axes:
    g = reduce_plain(el, "add", g, tuple(axes), zero, "sum", True)
  return g


def _mentioned(el, exprs: list[Expr]) -> set[int]:
  """Identities of everything the expressions are computed from, through the
  definitions of the variables they use."""
  out: set[int] = set()
  stack = list(exprs)
  while stack:
    for x in walk_expr(stack.pop()):
      if id(x) in out:
        continue
      out.add(id(x))
      if isinstance(x, Var) and x.uid in el.defs:
        stack.append(el.defs[x.uid])
  return out


def _elementwise(tape: Tape, el, node: Node, g: ArrayVal) -> None:
  active, fn, elt, cost = node.info
  args = node.inputs
  n = len(args)
  if sum(active) > 1 and g.buf is None and g.cost > CHEAP_COST:
    g = el.as_concrete(g)
  for k in range(n):
    if not active[k]:
      continue
    a = args[k]
    t = a.elt if isinstance(a, ArrayVal) else a.type.strong
    # the arguments this derivative reads must still hold what the operator saw
    # (the derivative of a sum reads none: x += y can be undone without the old x)
    with el.trial():
      probes = [Var("x", x.elt if isinstance(x, ArrayVal) else x.type) for x in args]
      seen: list[Expr] = []
      backprop(
        el,
        fn(*probes),
        Var("g", elt),
        lambda v, adj, k=k: seen.append(adj) if v is probes[k] else el.defs.get(v.uid),
        lambda *_: None,
      )
      used = _mentioned(el, seen)
    for x, probe in zip(args, probes):
      if id(probe) in used:
        tape.stale(el, node, x)

    def partial(*xs, k=k, t=t):
      *xs, gv = xs
      mark = Var._counter
      leaves = []
      for x in xs:
        leaf = Var("x", x.type)
        el.block.append(Assign(leaf, x))
        note_def(el, leaf, x)  # an enclosing gradient differentiates through this copy
        leaves.append(leaf)
      r = fn(*leaves)
      found: list[Expr] = []

      def on_var(v: Var, adj: Expr):
        if v is leaves[k]:
          found.append(adj)
          return None
        if any(v is leaf for leaf in leaves):
          return None  # another argument: a constant for this partial derivative
        return el.defs.get(v.uid) if v.uid > mark else None

      backprop(el, r, cast(gv, r.type), on_var, lambda load, adj: None)
      return cast(found[0], t) if found else Const(0.0, t)

    contrib = el.elementwise([*args, g], partial, t, cost + 4)
    tape.add(el, a, unbroadcast(el, contrib, a))


def _alias(tape: Tape, el, node: Node, g: ArrayVal) -> None:
  tape.add(el, node.inputs[0], g)


def _opaque(tape: Tape, el, node: Node, g: ArrayVal) -> None:
  arr = node.inputs[0]
  if g.buf is None and g.cost > CHEAP_COST:
    g = el.as_concrete(g)
  el.consume(g)
  # a delayed array knows which arrays it reads; without that, anything older counts
  tape.stale(el, node, arr if arr.roots is not None else None, roots_known=True)

  def body(idx):
    tape.push(el, arr.fn(tuple(idx)), el.tmp(el.elem(g, idx), "g"))

  el.seq_nest(arr.shape, body)


def _reduce(tape: Tape, el, node: Node, g) -> None:
  from ..lib.numpy_lib import _axes
  from ..lib.numpy_lib import _reduce_prim as reduce_plain

  a = node.inputs[0]
  init = node.inputs[1] if len(node.inputs) > 1 else None
  op, axis, keepdims, active_a, active_init = node.info
  y = node.out
  axes = _axes(el, axis, a.rank)
  elt = a.elt

  def spread(v):
    """A reduced value, shaped so that it broadcasts against ``a``."""
    if not isinstance(v, ArrayVal):
      return cast(v, elt)
    if not keepdims and axes is not None and len(axes) < a.rank:
      v = el.expand_dims(v, axes)
    return v

  def to_a(contrib):
    if not isinstance(contrib, ArrayVal) or contrib.rank != a.rank:
      scalar = el.tmp(contrib if isinstance(contrib, Expr) else el.as_scalar(contrib))
      return el.delayed(elt, a.shape, lambda idx: cast(scalar, elt), cost=0)
    return el.broadcast_to(contrib, a.shape)

  def sum_all(v):
    if not isinstance(v, ArrayVal) or v.rank == 0:
      return v if isinstance(v, Expr) else el.as_scalar(v)
    return reduce_plain(el, "add", v, None, Const(0, elt), "sum", False)

  gb = spread(g)
  if op == "add":
    if active_a:
      tape.add(el, a, to_a(gb))
    if active_init:
      tape.add(el, init, sum_all(g))
    return
  tape.stale(el, node, a)
  yb = spread(y)
  zero = Const(0, elt)
  one = Const(1, elt)
  if op == "mul":
    # Zero-safe derivative of prod: handle 0, 1, or >=2 zeros along the reduced axis
    if active_a:
      is_zero = el.apply_ufunc(UF["eq"], [a, zero])
      if is_zero.buf is None:
        is_zero = el.as_concrete(is_zero)
      n_zeros = reduce_plain(el, "add", el.cast_array(is_zero, elt), axis, zero, "sum", True)
      if isinstance(n_zeros, ArrayVal) and n_zeros.buf is None:
        n_zeros = el.as_concrete(n_zeros)
      safe_a = el.elementwise([is_zero, a], lambda z, x: ops.select(cast(z, Bool), one, cast(x, elt)), elt, 1)
      nonzero_prod = reduce_plain(el, "mul", safe_a, axis, one if init is None else cast(init, elt), "prod", True)
      if isinstance(nonzero_prod, ArrayVal) and nonzero_prod.buf is None:
        nonzero_prod = el.as_concrete(nonzero_prod)

      def prod_partial(is_zero_el, zero_count, prod_other, safe_val, grad_val):
        """Zero-safe derivative of a product: 0 zeros -> prod/x, 1 zero -> prod of others at that zero, >=2 -> 0."""
        at_zero = ops.select(cmp("eq", zero_count, one), prod_other, zero)
        at_nonzero = ops.select(cmp("eq", zero_count, zero), ops.prim("div", (prod_other, safe_val), elt), zero)
        return ops.prim("mul", (cast(grad_val, elt), ops.select(cast(is_zero_el, Bool), at_zero, at_nonzero)), elt)

      contrib = el.elementwise([is_zero, n_zeros, nonzero_prod, safe_a, gb], prod_partial, elt, 4)
      tape.add(el, a, to_a(contrib))
    if active_init:
      prod_a = reduce_plain(el, "mul", a, axis, one, "prod", keepdims)
      tape.add(el, init, sum_all(el.apply_ufunc(UF["mul"], [g, prod_a])))
    return
  # min / max: ties (including initial, if supplied) share the adjoint equally
  hit = el.apply_ufunc(UF["eq"], [a, yb])
  if hit.buf is None:
    hit = el.as_concrete(hit)
  count = reduce_plain(el, "add", el.cast_array(hit, elt), axis, zero, "sum", True)
  init_hit = None
  if init is not None:
    init_c = cast(init, elt)
    init_hit = el.apply_ufunc(UF["eq"], [init_c, yb])
    count = el.apply_ufunc(
      UF["add"], [count, el.cast_array(init_hit, elt) if isinstance(init_hit, ArrayVal) else cast(init_hit, elt)]
    )
  if isinstance(count, ArrayVal) and count.buf is None:
    count = el.as_concrete(count)
  share = el.apply_ufunc(UF["div"], [gb, el.apply_ufunc(UF["max"], [count, one])])
  if isinstance(share, Expr):
    share = el.tmp(share)
  if active_a:
    contrib = el.elementwise([hit, share], lambda h, s: ops.select(cast(h, Bool), cast(s, elt), zero), elt, 1)
    tape.add(el, a, to_a(contrib))
  if active_init and init_hit is not None:
    if isinstance(init_hit, ArrayVal) or isinstance(share, ArrayVal):
      init_share = el.elementwise([init_hit, share], lambda h, s: ops.select(cast(h, Bool), cast(s, elt), zero), elt, 1)
    else:
      init_share = ops.select(cast(init_hit, Bool), cast(share, elt), zero)
    tape.add(el, init, sum_all(init_share))


def _matmul(tape: Tape, el, node: Node, g) -> None:
  from ..lib.numpy_lib import matmul as matmul_plain

  a, b = node.inputs
  active_a, active_b = node.info
  tape.stale(el, node, a)
  tape.stale(el, node, b)
  if isinstance(g, ArrayVal) and g.buf is None:
    g = el.as_concrete(g)

  def outer(u, v):
    return el.apply_ufunc(UF["mul"], [el.expand_dims(u, [1]), el.expand_dims(v, [0])])

  if a.rank == 1 and b.rank == 1:
    ga, gb = el.apply_ufunc(UF["mul"], [g, b]), el.apply_ufunc(UF["mul"], [g, a])
  elif a.rank == 2 and b.rank == 2:
    ga = matmul_plain(el, g, el.transpose(b), "matmul") if active_a else None
    gb = matmul_plain(el, el.transpose(a), g, "matmul") if active_b else None
  elif a.rank == 2:
    ga = outer(g, b) if active_a else None
    gb = matmul_plain(el, el.transpose(a), g, "matmul") if active_b else None
  else:
    ga = matmul_plain(el, b, g, "matmul") if active_a else None
    gb = outer(a, g) if active_b else None
  if active_a:
    tape.add(el, a, ga)
  if active_b:
    tape.add(el, b, gb)


def _pack(tape: Tape, el, node: Node, g: ArrayVal) -> None:
  holder = node.holder
  assert isinstance(holder, RootAdj)
  for view, v in zip(node.info, node.inputs):
    part = holder.view(el, view) if holder.buf is not None else _same_view(el, g, holder.arr, view)
    tape.add(el, v, part if isinstance(v, ArrayVal) else el.as_scalar(part, "adjoint"))


def _same_view(el, g: ArrayVal, arr: ArrayVal, view: ArrayVal) -> ArrayVal:
  """The part of ``g`` (an adjoint of ``arr``) that corresponds to ``view`` of ``arr``."""
  if g.buf is None or not g.contig:
    g = el.materialize(g)
  out = ArrayVal(
    g.elt, view.shape, buf=g.buf, offset=el.tmp(iadd(g.offset, isub(view.offset, arr.offset)), "off"),
    strides=view.strides, roots=g.roots,
  )  # fmt: skip
  out.inj = view.inj
  return out


PRIVATE_BYTES = 1 << 28  # memory a parallel backward loop may spend on per-chunk accumulators


def _map(tape: Tape, el, node: Node, g: ArrayVal) -> None:
  """Differentiate the element function again for every element.

  Every element may add to the adjoint of anything its function used, so in
  parallel each chunk of the loop gets accumulators of its own, summed at
  the end: a reduction whose accumulator is the whole set of adjoints."""
  from ..lib.numpy_lib import reduce_plain
  from ..syntax.ir import RtCall

  shape, at, sub_view, touched = node.info
  tape.stale(el, node)
  g = el.as_concrete(g)
  lead = len(shape)

  def body(idx):
    child = Tape(el, tape, dry=tape.dry, key=(*tape.key, id(node.origin)))
    el.tape = child
    try:
      r = at(tuple(idx))
    finally:
      el.tape = None
    if isinstance(r, ArrayVal):
      seed: Any = sub_view(el, g, idx) if g.rank > lead else el.tmp(el.elem(g, idx), "g")
    else:
      seed = el.tmp(el.elem(g, idx), "g")
    child.backward(el, r, seed)
    if tape.dry:
      for t in child.touched:
        if t.tape is not tape:
          tape.touched.add(t)

  roots = [t for t in touched if isinstance(t, RootAdj)]
  scalars = [t for t in touched if isinstance(t, ScalarAdj)]
  if tape.dry or not el.can_parallel(shape) or any(r.stored is not None for r in roots):
    el.seq_nest(shape, body)
    return
  total = el.tmp(ops.iprod(shape), "total")
  sizes: Expr = Const(len(scalars) + 1, I)
  for r in roots:
    r.target(el)  # the shared accumulator has to exist before the loop
    sizes = iadd(sizes, ops.iprod(r.arr.shape))
  budget = Const(PRIVATE_BYTES // 8, I)
  nchunks = ops.imin(ops.imin(total, RtCall("num_threads", (), I)), ops.imax(ops.idiv(budget, sizes), 1))
  nchunks = el.tmp(ops.imax(nchunks, 1), "nchunks")
  if isinstance(nchunks, Var):
    el.nonneg.add(nchunks.uid)
  private = [el.alloc_array(r.arr.elt, (nchunks, *r.arr.shape), zero=True, label="adj") for r in roots]
  cells = [el.alloc_array(s.var.type.strong, (nchunks,), zero=True, label="adj") for s in scalars]

  def chunk_body(chunk, start, stop):
    saved = [r.redirect for r in roots], [s.redirect for s in scalars]
    try:
      for r, p in zip(roots, private):
        r.redirect = sub_view(el, p, (chunk,))
        r.redirect.inj = True
      for s, c in zip(scalars, cells):
        s.redirect = (c.buf, chunk)
      el.chunk_iter(shape, start, stop, body)
    finally:
      for r, old in zip(roots, saved[0]):
        r.redirect = old
      for s, old in zip(scalars, saved[1]):
        s.redirect = old

  # a fold, as far as backends are concerned: the chunks are tied to the accumulators
  el.par_loop(total, chunk_body, kind="reduce", nchunks=nchunks, base_depth=1 if lead == 1 else 2)
  for r, p in zip(roots, private):
    summed = reduce_plain(el, "add", p, Const(0, I), Const(0, p.elt), "sum", False)
    accumulate(el, r.view(el, r.arr), summed)
  for s, c in zip(scalars, cells):
    s.add(el, reduce_plain(el, "add", c, None, Const(0, c.elt), "sum", False), tape)


def _store(tape: Tape, el, node: Node, g) -> None:
  dest, root, active = node.info
  (value,) = node.inputs
  region = root.view(el, dest)
  taken = el.materialize(region, "adj") if active else None
  el.store_array(region, Const(0.0, region.elt))
  if taken is not None:
    tape.add(el, value, unbroadcast(el, taken, value))


def _store_element(tape: Tape, el, node: Node, g) -> None:
  pos, root, active = node.info
  (v,) = node.inputs
  buf = root.target(el)
  p = el.tmp(iadd(buf.offset, isub(pos, root.arr.offset)), "pos")
  taken = el.tmp(Load(buf.buf, p), "g") if el.tape is not None else el.copy_scalar(Load(buf.buf, p), "g")
  store_element(el, buf, p, Const(0.0, buf.elt))
  if active:
    tape.add(el, v, taken)


def _cumsum(tape: Tape, el, node: Node, g: ArrayVal) -> None:
  """d cumsum: every element feeds all the later ones, a cumsum from the other end."""
  from ..lib.numpy_lib import cumulative_plain, flip

  (a,) = node.inputs
  axis = node.info
  g = el.as_concrete(g)
  along = None if g.rank == 1 else axis
  back = flip(el, el.as_concrete(cumulative_plain(el, flip(el, g, along), along, "add")), along)
  if back.rank != a.rank:
    back = el.reshape(el.as_concrete(back), a.shape, checked=True)
  tape.add(el, a, back)


def _fold_step(tape: Tape, el, x: ArrayVal, info, i: Expr, cell: ArrayVal, dry: bool, key) -> None:
  """One step of a fold, backwards: the adjoint of the running value after
  element ``i`` (in ``cell``) becomes that of the one before, and of ``x[i]``."""
  accs, init, step, _ = info
  t = accs.elt
  before = Load(accs.buf, ops.imax(isub(i, 1), 0))
  if init is not None:
    before = ops.select(cmp("gt", i, 0), before, cast(init, t))
  acc = Var("acc", t)
  el.block.append(Assign(acc, before))
  child = Tape(el, tape, dry=dry, key=(*tape.key, key))
  leaf = child.leaf_scalar(el, acc)
  el.tape = child
  try:
    r = step(acc, el.tmp(el.elem(x, (i,)), "x"))
  finally:
    el.tape = None
  child.backward(el, r, el.tmp(Load(cell.buf, ZERO), "g"))
  total = leaf.total(el)
  el.emit(Store(cell.buf, ZERO, cast(total, t) if total is not None else Const(0.0, t)))


def _fold(tape: Tape, el, node: Node, g: Expr) -> None:
  (x,) = node.inputs
  accs, init, step, n = node.info
  tape.stale(el, node)
  cell = el.alloc_array(accs.elt, (ONE,), zero=True, label="adj")
  el.emit(Store(cell.buf, ZERO, cast(g, accs.elt)))
  lower = ZERO if init is not None else ONE
  el.for_range(
    isub(n, 1), isub(lower, 1), lambda i: _fold_step(tape, el, x, node.info, i, cell, False, id(node.origin)),
    step=Const(-1, I),
  )  # fmt: skip
  rest = el.tmp(Load(cell.buf, ZERO), "g")
  if init is None:
    tape.add(el, el.tmp(el.elem(x, (ZERO,)), "x"), rest)  # the fold started from x[0]
  else:
    tape.add(el, init, rest)


def _scatter_add(tape: Tape, el, node: Node, g: ArrayVal) -> None:
  (values,) = node.inputs
  (idx,) = node.info
  # Adjoint of scatter_add(n, idx, values) w.r.t. values is gather: d_values = g[idx]
  g = el.as_concrete(g)
  gathered = el.getitem(g, idx)
  tape.add(el, values, gathered)


def _gather(tape: Tape, el, node: Node, g) -> None:
  from ..syntax.ir import IndexScatter, ScatterUpdate

  (base,) = node.inputs
  shape, src_index, index_arrays = node.info
  tape.stale(el, node, base)
  for ia in index_arrays:
    tape.stale(el, node, ia)
  root = tape.find_root(base.buf.uid)
  if root is None:
    return
  if isinstance(g, ArrayVal):
    if g.buf is None and g.cost > CHEAP_COST:
      g = el.as_concrete(g)
    el.consume(g)
    g = el.broadcast_to(g, shape, consumed=True)
  else:
    g = el.tmp(g, "g")
  adj_view = root.view(el, base)
  dest_size = el.tmp(ops.iprod(root.arr.shape), "dsize")
  shape_t = tuple(el.tmp(d, "n") for d in shape)
  idx_vars = tuple(Var("i", I) for _ in shape_t)
  for v, d in zip(idx_vars, shape_t):
    el.nonneg.add(v.uid)
    el.upper[v.uid] = d
  threads = el.can_parallel(shape_t)
  blk: list = []
  scope = el.new_scope("loop")
  with el.in_block(blk, scope, loop=True, par=threads):
    target = el.tmp(el.index_expr(adj_view, src_index(idx_vars)), "pos")
    val_expr = el.elem(g, idx_vars) if isinstance(g, ArrayVal) else g
    val = el.tmp(cast(val_expr, adj_view.elt), "g")
  el.emit(
    IndexScatter(
      shape_t, idx_vars, blk, dest_size, target, [ScatterUpdate(adj_view.buf, val, "add")], scope, threads, el.loc()
    )
  )
  el.record_store(adj_view)


def _algebraic_vjp(tape: Tape, el, node: Node, g) -> None:
  from .algebraic_vjp import run_algebraic_vjp

  run_algebraic_vjp(tape, el, node, g)


def _segmented_reduce_ad(tape: Tape, el, node: Node, g) -> None:
  from ..lib.ragged_lib import _segmented_reduce_vjp

  _segmented_reduce_vjp(tape, el, node, g)


def _segmented_softmax_ad(tape: Tape, el, node: Node, g) -> None:
  from ..lib.ragged_lib import _segmented_softmax_vjp

  _segmented_softmax_vjp(tape, el, node, g)


def _varlen_flash_attention_ad(tape: Tape, el, node: Node, g) -> None:
  from ..lib.ragged_lib import _varlen_flash_attention_vjp

  _varlen_flash_attention_vjp(tape, el, node, g)


def _block_sparse_attention_ad(tape: Tape, el, node: Node, g) -> None:
  from ..lib.spatial_lib import _block_sparse_attention_vjp

  _block_sparse_attention_vjp(tape, el, node, g)


_BACKWARD = {
  "elementwise": _elementwise,
  "alias": _alias,
  "opaque": _opaque,
  "reduce": _reduce,
  "matmul": _matmul,
  "pack": _pack,
  "map": _map,
  "cumsum": _cumsum,
  "store": _store,
  "store_element": _store_element,
  "fold": _fold,
  "scatter_add": _scatter_add,
  "gather": _gather,
  "algebraic_vjp": _algebraic_vjp,
  "segmented_reduce": _segmented_reduce_ad,
  "segmented_softmax": _segmented_softmax_ad,
  "varlen_flash_attention": _varlen_flash_attention_ad,
  "block_sparse_attention": _block_sparse_attention_ad,
}


class _StaticLeaf:
  """Non-differentiated structural leaf (e.g. integer CSR offsets in a RaggedArray)."""

  __slots__ = ("value",)

  def __init__(self, value):
    self.value = value


# ---------------------------------------------------------------------------
# grad
# ---------------------------------------------------------------------------


def _wrt_leaf(el, tape: Tape, v, what: str):
  """A differentiated argument as the function will see it, plus how to read
  its adjoint afterwards."""
  from ..ragged import RaggedArray

  if isinstance(v, RaggedArray):
    vals_in, vals_holder = _wrt_leaf(el, tape, v.values, what)
    return RaggedArray(vals_in, v.offsets), RaggedArray(vals_holder, _StaticLeaf(v.offsets))
  if record := record_type(v):
    values, holders = _wrt_leaf(el, tape, record.values(v), what)
    return record.build(values), record.build(holders)
  if isinstance(v, tuple):
    parts = [_wrt_leaf(el, tape, x, what) for x in v]
    return tuple(p[0] for p in parts), tuple(p[1] for p in parts)
  if isinstance(v, Expr) and is_float(v):
    leaf = Var("x", v.type)
    el.block.append(Assign(leaf, v))
    if tape.outer is not None:
      note_def(el, leaf, v)  # the enclosing gradient differentiates through the copy
    return leaf, tape.leaf_scalar(el, leaf)
  if isinstance(v, ArrayVal) and v.elt.kind == "f":
    # the adjoint mirrors the memory of the array, so the array must be dense and its own
    if v.buf is None or not v.contig or v.buf.uid in el.ad_bufs:
      v = el.materialize(v, "x")
    return v, tape.root(el, v)
  raise el.error(
    TypeInferenceError,
    f"{what} differentiates with respect to floating point scalars and arrays (and tuples of them), not {describe(v)}",
  )


def _adjoint(el, holder, v):
  if isinstance(holder, _StaticLeaf):
    return holder.value
  if record := record_type(holder):
    return record.build(_adjoint(el, record.values(holder), record.values(v)))
  if isinstance(holder, tuple):
    return tuple(_adjoint(el, h, x) for h, x in zip(holder, v))
  if isinstance(holder, ScalarAdj):
    total = holder.total(el)
    return Const(0.0, holder.var.type) if total is None else total
  total = holder.total(el)
  if total is None:
    return el.alloc_array(v.elt, v.shape, zero=True, label="grad")
  return el.as_concrete(el.cast_array(total, v.elt))


def differentiate(el, f, argnum, args: list, kwargs: dict, with_value: bool, what: str):
  """Elaborate ``f(*args)`` and the gradient of its (scalar) result."""
  outer = el.tape  # set when this gradient is itself being differentiated
  if outer is not None and (outer.muted or outer.parent is not None):
    raise el.error(UnsupportedError, f"{what} inside a function that is mapped and differentiated is not supported")
  if isinstance(argnum, tuple):
    wrt = [el.static_int(a, f"the argument numbers passed to {what}") for a in argnum]
    single = False
  else:
    wrt = [el.static_int(argnum, f"the argument number passed to {what}")]
    single = True
  for k in wrt:
    if not 0 <= k < len(args):
      raise el.error(
        TypeInferenceError,
        f"{what}: there is no positional argument {k} (the function was called with {len(args)})",
      )
  if len(set(wrt)) != len(wrt):
    raise el.error(TypeInferenceError, f"{what}: argument numbers must be distinct")
  args = [el.bindable(a, "arg") for a in args]
  tape = Tape(el, outer=outer)
  el.ad_live += 1
  try:
    holders = {}
    for k in wrt:
      args[k], holders[k] = _wrt_leaf(el, tape, args[k], what)
    el.tape = tape
    try:
      out = el.call(f, args, kwargs)
    finally:
      el.tape = outer
    if isinstance(out, ArrayVal) and out.rank == 0:
      out = el.as_scalar(out)
    if not isinstance(out, Expr) or not is_float(out):
      raise el.error(
        TypeInferenceError,
        f"{what} needs a function that returns a floating point scalar; this one returns {describe(out)}",
        hint="reduce the result to one number first, for example with np.sum or np.mean",
      )
    out = el.tmp(out, "value")
    tape.backward(el, out, Const(1.0, out.type))
    grads = [_adjoint(el, holders[k], args[k]) for k in wrt]
  finally:
    el.ad_live -= 1
    el.tape = outer
    if not el.ad_live:
      # nothing depends on a differentiated value any more
      el.ad_vars.clear()
      el.ad_bufs.clear()
      el.tainted_vars.clear()
      el.tainted_bufs.clear()
  result = grads[0] if single else tuple(grads)
  return (out, result) if with_value else result


__all__ = ["MAX_UNROLL", "Tape", "accumulate", "depends", "differentiate", "note_def", "observe", "unbroadcast"]
_ = cmp
