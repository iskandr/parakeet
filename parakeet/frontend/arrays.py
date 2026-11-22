"""Array machinery of the frontend: views, broadcasting, fusion and loops.

Everything here is mixed into :class:`parakeet.frontend.elab.Elaborator`.  The central
idea is that array expressions are *delayed*: ``np.tanh(x * a) + b`` builds a
function from index to element, and a loop is only emitted when something
consumes the array (a store, a reduction, the function's return).  A delayed
array that turns out to be used in a way that would duplicate work or observe
a later mutation triggers a restart in which it is evaluated eagerly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..errors import InternalError, StaticValueError, TypeInferenceError, UnsupportedError
from ..ndtypes import Bool, BufType, Int64, ScalarType
from ..syntax import ops
from ..syntax.ir import (
  Alloc,
  Assign,
  Break,
  Check,
  Const,
  ErrorInfo,
  Expr,
  ForRange,
  If,
  Load,
  ParLoop,
  ReleaseScope,
  RtCall,
  Scope,
  Store,
  Var,
  While,
  same,
)
from ..syntax.ops import ONE, ZERO, I, cmp, iadd, idiv, imax, imin, imod, imul, iprod, isub, land, lor
from ..syntax.values import CHEAP_COST, INF_COST, ArrayVal, Root, describe


class Restart(Exception):
  """Abort the current elaboration and start over with updated assumptions."""


class LiveRoots:
  """Root set of a merge target; grows while the function is elaborated."""

  __slots__ = ("known", "consulted", "hint_key")

  def __init__(self, known, hint_key):
    self.known: set[Root] = set(known)
    self.consulted = False
    self.hint_key = hint_key


class ArrayMixin:
  # attributes provided by Elaborator
  block: list
  scope: Scope
  loop_depth: int
  par_depth: int
  state: Any
  options: Any
  tape: Any  # the autodiff tape recording array operators, if any

  # ------------------------------------------------------------------ roots

  def get_roots(self, arr: ArrayVal):
    r = arr.roots
    if isinstance(r, LiveRoots):
      r.consulted = True
      return frozenset(r.known)
    return r

  @staticmethod
  def union_roots(*sets):
    out: set = set()
    for s in sets:
      if s is None:
        return None
      out |= s
    return frozenset(out)

  @staticmethod
  def may_alias(a, b) -> bool:
    if a is None or b is None:
      return True
    if not a or not b:
      return False
    if a & b:
      return True
    # two different inputs may overlap in the caller's memory
    return any(r.scope is None for r in a) and any(r.scope is None for r in b)

  def record_store(self, arr: ArrayVal) -> None:
    roots = self.get_roots(arr)
    self.stores.append(roots)
    if roots is None:
      self.writes_unknown = True
    else:
      for r in roots:
        if r.param is not None:
          r.param.writes = True
    if self.tape is not None:
      self.tape.note_store(self, arr)

  # --------------------------------------------------------------- creation

  def c_strides(self, shape: Sequence[Expr]) -> tuple[Expr, ...]:
    strides: list[Expr] = []
    acc: Expr = ONE
    for dim in reversed(shape):
      strides.append(acc)
      acc = self.tmp(imul(acc, dim), "stride")
    return tuple(reversed(strides))

  def check_dims(self, shape: Sequence[Expr]) -> tuple[Expr, ...]:
    out = []
    for d in shape:
      d = self.tmp(d, "dim")
      if isinstance(d, Const):
        if d.value < 0:
          raise self.error(StaticValueError, "negative dimensions are not allowed")
      elif not self.is_nonneg(d):
        self.check(cmp("ge", d, 0), "ValueError", "negative dimensions are not allowed")
        self.nonneg.add(d.uid)
      out.append(d)
    return tuple(out)

  def alloc_array(self, elt: ScalarType, shape: Sequence[Expr], zero: bool = False, label: str = "arr") -> ArrayVal:
    shape = self.check_dims(shape)
    buf = Var(label, BufType(elt))
    self.emit(Alloc(buf, self.tmp(iprod(shape), "count"), zero))
    self.scope.owned.append(buf)
    root = Root(self.scope, label)
    if self.tape is not None:
      self.tape.note_alloc(root)
    arr = ArrayVal(
            elt, shape, buf=buf, offset=ZERO, strides=self.c_strides(shape),
            roots=frozenset({root}), contig=True,
        )  # fmt: skip
    arr.inj = True
    return arr

  def delayed(
    self,
    elt: ScalarType,
    shape: Sequence[Expr],
    fn: Callable[[tuple[Expr, ...]], Expr],
    roots=frozenset(),
    cost: int = 1,
    reads: tuple[ArrayVal, ...] | None = None,
    origin: Any = None,
  ) -> ArrayVal:
    arr = ArrayVal(elt, tuple(shape), fn=fn, roots=roots, cost=min(cost, INF_COST), reads=reads)
    arr.depth = self.loop_depth
    arr.epoch = len(self.stores)
    arr.block = self.block
    arr.origin = origin if origin is not None else (self.node_stack[-1] if self.node_stack else None)
    if not self.options.fusion or (arr.origin is not None and id(arr.origin) in self.state.strict):
      return self.as_concrete(arr)
    return arr

  def scalar_to_array(self, x: Expr) -> ArrayVal:
    x = self.tmp(x)
    return self.delayed(x.type, (), lambda idx: x, cost=0)

  # ---------------------------------------------------------------- elements

  def index_expr(self, arr: ArrayVal, idx: Sequence[Expr]) -> Expr:
    pos = arr.offset
    for i, s in zip(idx, arr.strides):
      pos = iadd(pos, imul(i, s))
    return pos

  def elem(self, arr: ArrayVal, idx: Sequence[Expr]) -> Expr:
    """Element of ``arr`` at a full index (no bounds checks, no bookkeeping)."""
    if arr.buf is not None:
      return Load(arr.buf, self.index_expr(arr, idx))
    return arr.fn(tuple(idx))

  # ------------------------------------------------------------- consumption

  def conflict(self, arr: ArrayVal, why: str = "") -> None:
    origin = arr.origin
    if origin is None:
      raise InternalError(f"delayed array without origin cannot be made eager ({why})")
    key = id(origin)
    self.state.keep.append(origin)
    if key not in self.state.eager:
      self.state.eager.add(key)
    elif key not in self.state.strict:
      self.state.strict.add(key)
    else:
      raise InternalError(f"could not resolve evaluation order of an array expression ({why})")
    raise Restart(why)

  def check_fresh(self, arr: ArrayVal) -> None:
    if arr.epoch < len(self.stores):
      roots = arr.roots
      for written in self.stores[arr.epoch :]:
        if self.may_alias(roots, written):
          self.conflict(arr, "an array it reads was modified before it was evaluated")
      arr.epoch = len(self.stores)

  def consume(self, arr: ArrayVal) -> None:
    """Note that the whole of ``arr`` is about to be evaluated once."""
    if arr.buf is not None:
      return
    self.check_fresh(arr)
    if self.loop_depth > arr.depth:
      self.conflict(arr, "evaluated inside a loop")
    if self.journal is not None:
      self.journal.append((arr, arr.uses))
    arr.uses += 1
    if arr.uses > 1 and arr.cost > CHEAP_COST:
      self.conflict(arr, "evaluated more than once")

  def as_concrete(self, arr: ArrayVal) -> ArrayVal:
    if arr.buf is not None:
      return arr
    return self.materialize(arr)

  def materialize(self, arr: ArrayVal, label: str = "arr") -> ArrayVal:
    """Evaluate ``arr`` into a freshly allocated C-contiguous array."""
    self.consume(arr)
    out = self.alloc_array(arr.elt, arr.shape, label=label)
    self.fill(out, arr)
    tape = self.tape
    if tape is not None and not tape.muted:
      tape.materialized(self, arr, out)
    return out

  def fill(self, dest: ArrayVal, src, parallel: bool = True) -> None:
    """``dest[...] = src`` where shapes already agree (or ``src`` is a scalar)."""
    elt = dest.elt
    if isinstance(src, ArrayVal):

      def body(idx):
        self.emit(Store(dest.buf, self.index_expr(dest, idx), ops.cast(self.elem(src, idx), elt)))

    else:
      value = self.tmp(ops.cast(src, elt))

      def body(idx):
        self.emit(Store(dest.buf, self.index_expr(dest, idx), value))

    self.loop_nest(dest.shape, body, parallel=parallel)

  def force_inplace(self, arr: ArrayVal) -> None:
    """Turn a delayed array into a concrete one *in place* (for mutation)."""
    if arr.buf is not None:
      return
    if self.block is not arr.block:
      self.conflict(arr, "mutated after a branch")
    source = arr
    tape = self.tape
    if tape is not None and not tape.muted:
      source = tape.detach(arr)  # the tape knows this array as a delayed value: keep that apart
    out = self.materialize(source)
    arr.buf, arr.offset, arr.strides = out.buf, out.offset, out.strides
    arr.shape, arr.roots, arr.contig, arr.fn, arr.reads = out.shape, out.roots, True, None, None
    arr.inj = True

  # ------------------------------------------------------------------- loops

  def new_scope(self, kind: str) -> Scope:
    return Scope(kind, self.scope)

  def for_range(
    self,
    start,
    stop,
    body: Callable[[Var], None],
    step=ONE,
    simd: bool = False,
    name: str = "i",
    reduction: tuple | None = None,
  ) -> None:
    var = Var(name, I)
    if self.is_nonneg(start) and isinstance(step, Const) and step.value > 0:
      self.nonneg.add(var.uid)
    if isinstance(step, Const) and step.value > 0:
      self.upper[var.uid] = stop
    blk: list = []
    scope = self.new_scope("loop")
    with self.in_block(blk, scope, loop=True):
      body(var)
      self.emit(ReleaseScope(scope))
    self.emit(ForRange(var, ops._i(start), ops._i(stop), ops._i(step), blk, scope, simd, reduction))

  def can_parallel(self, shape: Sequence[Expr]) -> bool:
    if not self.options.parallel or self.par_depth > 0:
      return False
    if all(isinstance(d, Const) for d in shape):
      total = 1
      for d in shape:
        total *= d.value
      return total >= 64
    return True

  def loop_nest(self, shape: Sequence[Expr], body: Callable[[tuple[Expr, ...]], None], parallel: bool = True) -> None:
    """Run ``body(idx)`` for every index of ``shape`` (in parallel if allowed)."""
    if not shape:
      body(())
      return
    if parallel and self.can_parallel(shape):
      total = self.tmp(iprod(shape), "total")
      self.par_loop(
        total,
        lambda chunk, start, stop: self.chunk_iter(shape, start, stop, body),
        base_depth=1 if len(shape) == 1 else 2,
      )
    else:
      self.seq_nest(shape, body)

  def seq_nest(self, shape: Sequence[Expr], body, _idx: tuple = (), reduction: tuple | None = None) -> None:
    if len(_idx) == len(shape):
      body(_idx)
      return
    dim = shape[len(_idx)]
    last = len(_idx) == len(shape) - 1
    self.for_range(
      ZERO,
      dim,
      lambda i: self.seq_nest(shape, body, _idx + (i,), reduction),
      simd=last,
      reduction=reduction if last else None,
    )

  def par_loop(
    self,
    total: Expr,
    body: Callable[[Var, Var, Var], None],
    kind: str = "map",
    nchunks: Expr | None = None,
    base_depth: int = 1,
    work: Expr | None = None,
  ) -> Expr:
    """Emit a chunked parallel loop over ``[0, total)``.

    ``base_depth`` is the loop nesting the chunk iteration itself needs;
    anything deeper means each element is expensive ("heavy"), which lets
    the runtime use threads even for few elements."""
    from ..backends.cfamily import loop_depth

    if nchunks is None:
      nchunks = self.tmp(RtCall("num_chunks", (total, Const(kind == "reduce", Bool)), I), "nchunks")
      self.nonneg.add(nchunks.uid)
    chunk, start, stop = Var("chunk", I), Var("start", I), Var("stop", I)
    for v in (chunk, start, stop):
      self.nonneg.add(v.uid)
    blk: list = []
    scope = self.new_scope("par")
    with self.in_block(blk, scope, loop=True, par=True):
      body(chunk, start, stop)
      self.emit(ReleaseScope(scope))
    heavy = loop_depth(blk) > base_depth
    if kind != "map":
      work = None  # the chunks of a fold are tied to its partial results: size alone decides
    elif work is None and heavy:
      work = self.tmp(inner_work(self, blk, base_depth), "work")
    self.emit(ParLoop(total, nchunks, chunk, start, stop, blk, scope, kind, self.loc(), heavy, work))
    return nchunks

  def unravel(self, flat: Expr, shape: Sequence[Expr]) -> tuple[Expr, ...]:
    """Index tuple of flat position ``flat`` in a C-ordered ``shape``."""
    idx: list[Expr] = []
    rest = flat
    for k in range(len(shape) - 1, 0, -1):
      dim = shape[k]
      q = self.tmp(idiv(rest, dim), "q")
      idx.append(self.tmp(isub(rest, imul(q, dim)), "i"))
      rest = q
    idx.append(rest)
    out = tuple(reversed(idx))
    for v in out:
      if isinstance(v, Var):
        self.nonneg.add(v.uid)
    return out

  def chunk_iter(self, shape: Sequence[Expr], start: Expr, stop: Expr, body, reduction: tuple | None = None) -> None:
    """Visit flat positions ``[start, stop)`` of ``shape`` in order."""
    if len(shape) == 1:

      def flat_body(i: Var) -> None:
        self.upper[i.uid] = shape[0]
        body((i,))

      self.for_range(start, stop, flat_body, simd=True, reduction=reduction)
      return
    inner = shape[-1]
    flat = Var("flat", I)
    self.mutables.add(flat.uid)
    self.nonneg.add(flat.uid)
    self.emit(Assign(flat, start))
    blk: list = []
    scope = self.new_scope("loop")
    with self.in_block(blk, scope, loop=True):
      self.emit(If(cmp("ge", flat, stop), [Break()], []))
      row = self.tmp(idiv(flat, inner), "row")
      j0 = self.tmp(isub(flat, imul(row, inner)), "j0")
      j1 = self.tmp(imin(inner, iadd(j0, isub(stop, flat))), "j1")
      self.nonneg.update((row.uid, j0.uid))
      outer = self.unravel(row, shape[:-1])
      for v, dim in zip(outer, shape[:-1]):
        if isinstance(v, Var):
          self.upper[v.uid] = dim

      def inner_body(j: Var) -> None:
        self.upper[j.uid] = inner
        body(outer + (j,))

      self.for_range(j0, j1, inner_body, simd=True, name="j", reduction=reduction)
      self.emit(Assign(flat, iadd(flat, isub(j1, j0))))
      self.emit(ReleaseScope(scope))
    self.emit(While(blk, scope))

  # ------------------------------------------------------------------- facts

  def _find(self, uid: int) -> int:
    parent = self.facts
    root = uid
    while parent.get(root, root) != root:
      root = parent[root]
    return root

  def dim_equal(self, a: Expr, b: Expr) -> bool:
    if same(a, b):
      return True
    if isinstance(a, Var) and isinstance(b, Var):
      return self._find(a.uid) == self._find(b.uid)
    return False

  def assume_equal(self, a: Expr, b: Expr) -> None:
    if isinstance(a, Var) and isinstance(b, Var):
      ra, rb = self._find(a.uid), self._find(b.uid)
      if ra != rb:
        self.facts[ra] = rb
      if a.uid in self.not_one or b.uid in self.not_one:
        self.not_one.update((a.uid, b.uid))

  def is_not_one(self, e: Expr) -> bool:
    if isinstance(e, Const):
      return e.value != 1
    return isinstance(e, Var) and e.uid in self.not_one

  def is_nonneg(self, e, _depth: int = 0) -> bool:
    if isinstance(e, int):
      return e >= 0
    if isinstance(e, Const):
      return e.value >= 0
    if isinstance(e, Var):
      return e.uid in self.nonneg
    if _depth < 6 and isinstance(e, ops.Prim):
      if e.op in ("add", "mul", "min", "floordiv", "mod") and e.type.kind in "iu":
        if e.op == "min":
          return all(self.is_nonneg(a, _depth + 1) for a in e.args)
        return all(self.is_nonneg(a, _depth + 1) for a in e.args)
      if e.op == "max":
        return any(self.is_nonneg(a, _depth + 1) for a in e.args)
      if e.op == "select":
        return self.is_nonneg(e.args[1], _depth + 1) and self.is_nonneg(e.args[2], _depth + 1)
    if isinstance(e, ops.Cast) and _depth < 6:
      return e.type.kind in "iu" and e.arg.type.kind in "iub" and self.is_nonneg(e.arg, _depth + 1)
    return False

  def proven_less(self, i: Expr, dim: Expr) -> bool:
    """Is ``i < dim`` known from the loop structure?"""
    if isinstance(i, Const) and isinstance(dim, Const):
      return i.value < dim.value
    if isinstance(i, Var):
      bound = self.upper.get(i.uid)
      if bound is not None:
        if self.dim_equal(bound, dim):
          return True
        if isinstance(bound, ops.Prim) and bound.op == "min":
          return any(self.dim_equal(a, dim) for a in bound.args)
    return False

  # ------------------------------------------------------------------- views

  def remap(self, arr: ArrayVal, shape: Sequence[Expr], mapping, consumed: bool = False) -> ArrayVal:
    """General strided view.

    ``mapping[k] = (j, start, step)`` says that source dimension ``k`` is
    indexed by ``start + idx[j] * step`` (or just ``start`` if ``j`` is None).
    """
    shape = tuple(shape)
    arr = self.settle(arr)
    if arr.buf is not None:
      offset = arr.offset
      strides: list[Expr] = [ZERO] * len(shape)
      for (j, start, step), s in zip(mapping, arr.strides):
        offset = iadd(offset, imul(start, s))
        if j is not None:
          strides[j] = iadd(strides[j], imul(step, s))
      out = ArrayVal(
                arr.elt, shape, buf=arr.buf, offset=self.tmp(offset, "off"),
                strides=tuple(self.tmp(s, "stride") for s in strides), roots=arr.roots,
            )  # fmt: skip
      base = arr if arr.contig else arr.base
      if base is not None:
        if (
          len(shape) == base.rank
          and same(out.offset, base.offset)
          and all(a is b for a, b in zip(out.strides, base.strides))
          and all(a is b for a, b in zip(out.shape, base.shape))
        ):
          return base  # e.g. the transpose of a transpose
        out.base = base
      used = [j for j, _, _ in mapping if j is not None]
      out.inj = (
        arr.inj
        and sorted(used) == list(range(len(shape)))
        and all(isinstance(step, Const) and step.value != 0 for j, _, step in mapping if j is not None)
      )
      return out
    if not consumed:
      self.check_fresh(arr)
      if self.loop_depth > arr.depth:
        self.conflict(arr, "sliced inside a loop")
      if self.journal is not None:
        self.journal.append((arr, arr.uses))
      arr.uses += 1
      if arr.uses > 1 and arr.cost > CHEAP_COST:
        self.conflict(arr, "used more than once")
    src = arr.fn
    mapping = [(j, self.tmp(start), self.tmp(step)) for j, start, step in mapping]

    def fn(idx):
      return src(tuple(start if j is None else iadd(start, imul(idx[j], step)) for j, start, step in mapping))

    out = self.delayed(arr.elt, shape, fn, roots=arr.roots, cost=arr.cost, origin=arr.origin)
    if out.buf is None:
      out.epoch = arr.epoch
    return out

  def settle(self, arr: ArrayVal) -> ArrayVal:
    """Under differentiation, evaluate a delayed array that carries a
    derivative before taking a view of it: views of a concrete array need no
    rule of their own (they are the same views of its adjoint)."""
    tape = self.tape
    if tape is not None and not tape.muted and arr.buf is None and self.block is tape.block:
      if tape.find_delayed(arr) is not None:
        return self.as_concrete(arr)
    return arr

  def axis_view(self, arr: ArrayVal, axis: int, i: Expr, consumed: bool = False):
    """``arr`` with ``axis`` fixed at ``i``: a scalar for 1-d arrays, else a view."""
    if arr.rank == 1:
      return self.tmp(self.elem(arr, (i,)), "x")
    mapping = []
    for k in range(arr.rank):
      if k < axis:
        mapping.append((k, ZERO, ONE))
      elif k == axis:
        mapping.append((None, i, ZERO))
      else:
        mapping.append((k - 1, ZERO, ONE))
    return self.remap(arr, arr.shape[:axis] + arr.shape[axis + 1 :], mapping, consumed)

  def transpose(self, arr: ArrayVal, perm: Sequence[int] | None = None) -> ArrayVal:
    if perm is None:
      perm = tuple(reversed(range(arr.rank)))
    if sorted(perm) != list(range(arr.rank)):
      raise self.error(StaticValueError, f"axes {tuple(perm)} don't match array of rank {arr.rank}")
    if tuple(perm) == tuple(range(arr.rank)):
      return arr
    mapping: list = [None] * arr.rank
    for j, k in enumerate(perm):
      mapping[k] = (j, ZERO, ONE)
    return self.remap(arr, tuple(arr.shape[k] for k in perm), mapping)

  def expand_dims(self, arr: ArrayVal, positions: Sequence[int]) -> ArrayVal:
    rank = arr.rank + len(positions)
    positions = sorted(p % rank for p in positions)
    shape: list[Expr] = []
    mapping = []
    for j in range(rank):
      if j in positions:
        shape.append(ONE)
      else:
        mapping.append((j, ZERO, ONE))
        shape.append(arr.shape[len(mapping) - 1])
    return self.remap(arr, shape, mapping)

  def flat_view(self, arr: ArrayVal) -> ArrayVal:
    """1-d C-order view of ``arr`` (copying only if it is strided)."""
    return self.reshape(arr, (self.tmp(iprod(arr.shape), "size"),), checked=True)

  def reshape(self, arr: ArrayVal, shape: Sequence[Expr], checked: bool = False) -> ArrayVal:
    shape = list(shape)
    arr = self.settle(arr)
    unknown = [k for k, d in enumerate(shape) if isinstance(d, Const) and d.value == -1]
    size = self.tmp(iprod(arr.shape), "size")
    if len(unknown) > 1:
      raise self.error(StaticValueError, "can only specify one unknown dimension")
    if unknown:
      known = self.tmp(iprod([d for k, d in enumerate(shape) if k != unknown[0]]), "known")
      self.check(
                land(cmp("gt", known, 0), cmp("eq", imod(size, imax(known, 1)), 0)),
                "ValueError", "cannot reshape array of size {0} into the requested shape", size,
            )  # fmt: skip
      shape[unknown[0]] = self.tmp(idiv(size, imax(known, 1)), "dim")
      self.nonneg.add(shape[unknown[0]].uid) if isinstance(shape[unknown[0]], Var) else None
    elif not checked:
      shape = list(self.check_dims(shape))
      new_size = iprod(shape)
      if not same(new_size, size):
        self.check(
                    cmp("eq", new_size, size), "ValueError",
                    "cannot reshape array of size {0} into a shape of size {1}", size, new_size,
                )  # fmt: skip
    shape_t = tuple(self.tmp(d, "dim") for d in shape)
    if len(shape_t) == arr.rank and all(self.dim_equal(a, b) for a, b in zip(shape_t, arr.shape)):
      return arr
    if arr.buf is not None and not arr.contig and self.tape is not None and not self.tape.muted:
      if self.tape.is_active(self, arr):
        arr = self.materialize(arr)  # then the reshaped array is a view again
    if arr.buf is not None and arr.contig:
      out = ArrayVal(
                arr.elt, shape_t, buf=arr.buf, offset=arr.offset,
                strides=self.c_strides(shape_t), roots=arr.roots, contig=True,
            )  # fmt: skip
      out.inj = arr.inj
      return out
    if arr.rank <= 1 and len(shape_t) >= 1 and arr.buf is not None:
      # a strided 1-d (or 0-d) source: scale the C strides of the new shape
      base = arr.strides[0] if arr.rank == 1 else ZERO
      strides = tuple(self.tmp(imul(s, base), "stride") for s in self.c_strides(shape_t))
      return ArrayVal(arr.elt, shape_t, buf=arr.buf, offset=arr.offset, strides=strides, roots=arr.roots)
    # general case: index arithmetic over the source (copies when stored into)
    self.consume(arr)
    src_shape = arr.shape
    new_strides = self.c_strides(shape_t)

    def fn(idx):
      flat: Expr = ZERO
      for i, s in zip(idx, new_strides):
        flat = iadd(flat, imul(i, s))
      return self.elem(arr, self.unravel(self.tmp(flat, "flat"), src_shape)) if src_shape else self.elem(arr, ())

    return self.delayed(
      arr.elt, shape_t, fn, roots=self.get_roots(arr), cost=arr.cost + 2 * arr.rank, origin=arr.origin
    )

  # ------------------------------------------------------------ broadcasting

  def broadcast_dim(self, a: Expr, b: Expr, what: str = "operands") -> Expr:
    if self.dim_equal(a, b):
      return a
    if isinstance(a, Const) and a.value == 1:
      return b
    if isinstance(b, Const) and b.value == 1:
      return a
    if isinstance(a, Const) and isinstance(b, Const):
      raise self.error(
        TypeInferenceError,
        f"{what} could not be broadcast together: dimensions {a.value} and {b.value} differ",
      )
    msg = what + " could not be broadcast together: dimensions {0} and {1} differ"
    if self.is_not_one(a) and self.is_not_one(b):
      self.check(cmp("eq", a, b), "ValueError", msg, a, b)
      self.assume_equal(a, b)
      return a
    ok = lor(cmp("eq", a, b), lor(cmp("eq", a, 1), cmp("eq", b, 1)))
    self.check(ok, "ValueError", msg, a, b)
    res = self.tmp(ops.select(cmp("eq", a, 1), b, a), "dim")
    if isinstance(res, Var) and self.is_nonneg(a) and self.is_nonneg(b):
      self.nonneg.add(res.uid)
    return res

  def broadcast_shapes(self, shapes: Sequence[Sequence[Expr]], what: str = "operands") -> tuple[Expr, ...]:
    rank = max(len(s) for s in shapes)
    out: list[Expr] = []
    for d in range(rank):
      dim: Expr | None = None
      for s in shapes:
        k = d - (rank - len(s))
        if k < 0:
          continue
        dim = s[k] if dim is None else self.broadcast_dim(dim, s[k], what)
      out.append(dim)  # type: ignore[arg-type]
    return tuple(out)

  def broadcast_to(
    self, arr: ArrayVal, shape: Sequence[Expr], consumed: bool = False, check: bool = False, what: str = ""
  ) -> ArrayVal:
    """View ``arr`` as ``shape`` (dims already known compatible unless ``check``)."""
    shape = tuple(shape)
    lead = len(shape) - arr.rank
    if lead < 0:
      raise self.error(
        TypeInferenceError,
        f"cannot broadcast a {arr.rank}-dimensional array to {len(shape)} dimensions {what}".rstrip(),
      )
    mapping = []
    identity = lead == 0
    for k, d in enumerate(arr.shape):
      r = shape[k + lead]
      if self.dim_equal(d, r):
        mapping.append((k + lead, ZERO, ONE))
        continue
      identity = False
      if isinstance(d, Const) and d.value == 1:
        mapping.append((None, ZERO, ZERO))
        continue
      if check:
        if isinstance(d, Const) and isinstance(r, Const):
          raise self.error(
            TypeInferenceError,
            f"could not broadcast array with dimension {d.value} to dimension {r.value} {what}".rstrip(),
          )
        self.check(
                    lor(cmp("eq", d, r), cmp("eq", d, 1)), "ValueError",
                    "could not broadcast input array with dimension {0} to dimension {1} " + what, d, r,
                )  # fmt: skip
      if self.is_not_one(d):
        self.assume_equal(d, r)
        mapping.append((k + lead, ZERO, ONE))
      else:
        mult = self.tmp(ops.select(cmp("eq", d, 1), ZERO, ONE), "bmul")
        mapping.append((k + lead, ZERO, mult))
    if identity:
      return arr
    return self.remap(arr, shape, mapping, consumed=consumed)

  def elementwise(self, args: Sequence[Any], fn: Callable[..., Expr], elt: ScalarType, cost: int = 1) -> ArrayVal:
    """Delayed array computing ``fn(*elements)`` over the broadcast of ``args``.

    ``args`` mixes arrays and scalar expressions; scalars are passed through.
    """
    tape = self.tape
    if tape is not None and not tape.muted and elt.kind == "f":
      active = [tape.is_active(self, a) for a in args]
      if any(active):
        return tape.elementwise(self, list(args), active, fn, elt, cost)
    return self.elementwise_plain(args, fn, elt, cost)

  def elementwise_plain(self, args: Sequence[Any], fn: Callable[..., Expr], elt: ScalarType, cost: int = 1) -> ArrayVal:
    arrays = [a for a in args if isinstance(a, ArrayVal)]
    shape = self.broadcast_shapes([a.shape for a in arrays])
    adapted: list[Any] = []
    total_cost = cost
    roots: Any = frozenset()
    reads: list[ArrayVal] | None = []
    for a in args:
      if isinstance(a, ArrayVal):
        self.consume(a)
        b = self.broadcast_to(a, shape, consumed=True)
        adapted.append(b)
        total_cost += a.cost
        roots = self.union_roots(roots, self.get_roots(a))
        if reads is not None:
          if b.buf is not None:
            reads.append(b)
          elif b.reads is not None and b is a:
            reads.extend(b.reads)
          else:
            reads = None
      else:
        adapted.append(self.tmp(a))

    def at(idx):
      return fn(*[self.elem(a, idx) if isinstance(a, ArrayVal) else a for a in adapted])

    return self.delayed(elt, shape, at, roots=roots, cost=total_cost, reads=None if reads is None else tuple(reads))

  def cast_array(self, arr: ArrayVal, elt: ScalarType) -> ArrayVal:
    elt = elt.strong
    if arr.elt is elt:
      return arr
    return self.elementwise([arr], lambda x: ops.cast(x, elt), elt, cost=0)

  # ---------------------------------------------------------------- indexing

  def index_error(self, axis: int):
    return ("IndexError", f"index {{0}} is out of bounds for axis {axis} with size {{1}}")

  def normalize_index(self, i: Expr, dim: Expr, axis: int = 0) -> Expr:
    """Wrap a possibly negative index and bounds-check it."""
    t = i.type
    if not isinstance(t, ScalarType) or t.kind not in "iu":
      raise self.error(
        TypeInferenceError,
        f"only integers, slices, `...`, None and integer or boolean arrays are valid indices (got {describe(i)})",
      )
    i = ops.cast(i, I)
    exc, msg = self.index_error(axis)
    if isinstance(i, Const):
      if isinstance(dim, Const):
        if not -dim.value <= i.value < dim.value:
          raise self.error(
            StaticValueError,
            f"index {i.value} is out of bounds for axis {axis} with size {dim.value}",
          )
        return Const(i.value % dim.value, I)
      if i.value < 0:
        if self.options.bounds_check:
          self.check(cmp("ge", dim, -i.value), exc, msg, i, dim)
        return self.tmp(iadd(dim, i), "i")
      if self.options.bounds_check:
        self.check(cmp("lt", i, dim), exc, msg, i, dim)
      return i
    i = self.tmp(i, "i")
    if self.is_nonneg(i):
      if self.options.bounds_check and not self.proven_less(i, dim):
        self.check(cmp("lt", i, dim), exc, msg, i, dim)
      return i
    wrapped = self.tmp(ops.select(cmp("lt", i, 0), iadd(i, dim), i), "i")
    if self.options.bounds_check:
      self.check(land(cmp("ge", wrapped, 0), cmp("lt", wrapped, dim)), exc, msg, i, dim)
    if isinstance(wrapped, Var):
      self.nonneg.add(wrapped.uid)
    return wrapped

  def _slice_bound(self, v, dim: Expr, lo, hi_is_dim: bool) -> Expr:
    """Clamp a slice bound the way ``slice.indices`` does (positive step:
    into [0, dim]; negative step: into [-1, dim - 1])."""
    v = ops.cast(self.as_index(v, "slice bounds"), I)
    if isinstance(v, Const):
      if v.value >= 0:
        return imin(v, dim) if hi_is_dim else imin(v, isub(dim, 1))
      w = iadd(dim, v)
      return imax(w, lo)
    v = self.tmp(v)
    if self.is_nonneg(v):
      return imin(v, dim) if hi_is_dim else imin(v, isub(dim, 1))
    w = ops.select(cmp("lt", v, 0), iadd(v, dim), v)
    hi = dim if hi_is_dim else isub(dim, 1)
    return imin(imax(w, lo), hi)

  def slice_params(self, sl: slice, dim: Expr) -> tuple[Expr, Expr, Expr]:
    """``(start, step, length)`` of ``sl`` applied to a dimension of size ``dim``."""
    step = ONE if sl.step is None else ops.cast(self.as_index(sl.step, "slice step"), I)
    if isinstance(step, Const):
      if step.value == 0:
        raise self.error(StaticValueError, "slice step cannot be zero")
      if step.value > 0:
        start = ZERO if sl.start is None else self._slice_bound(sl.start, dim, ZERO, True)
        stop = dim if sl.stop is None else self._slice_bound(sl.stop, dim, ZERO, True)
        span = isub(stop, start)
        if step.value != 1:
          span = idiv(iadd(span, step.value - 1), step)
        if sl.start is None and sl.stop is None and step.value == 1:
          return ZERO, ONE, dim
        length = span if (sl.start is None and step.value == 1) else imax(span, 0)
        length = self.tmp(length, "len")
      else:
        neg = Const(-1, I)
        start = isub(dim, 1) if sl.start is None else self._slice_bound(sl.start, dim, neg, False)
        stop = neg if sl.stop is None else self._slice_bound(sl.stop, dim, neg, False)
        span = idiv(iadd(isub(start, stop), -step.value - 1), -step.value)
        length = self.tmp(imax(span, 0), "len")
      if isinstance(length, Var):
        self.nonneg.add(length.uid)
      return self.tmp(start, "start"), step, length
    # step sign only known at run time
    step = self.tmp(step, "step")
    self.check(cmp("ne", step, 0), "ValueError", "slice step cannot be zero")
    pos = cmp("gt", step, 0)
    neg1 = Const(-1, I)
    if sl.start is None:
      start = ops.select(pos, ZERO, isub(dim, 1))
    else:
      start = ops.select(
        pos, self._slice_bound(sl.start, dim, ZERO, True), self._slice_bound(sl.start, dim, neg1, False)
      )
    if sl.stop is None:
      stop = ops.select(pos, dim, neg1)
    else:
      stop = ops.select(pos, self._slice_bound(sl.stop, dim, ZERO, True), self._slice_bound(sl.stop, dim, neg1, False))
    start, stop = self.tmp(start, "start"), self.tmp(stop, "stop")
    mag = self.tmp(ops.select(pos, step, isub(0, step)), "mag")
    dist = ops.select(pos, isub(stop, start), isub(start, stop))
    length = self.tmp(imax(idiv(iadd(dist, isub(mag, 1)), mag), 0), "len")
    if isinstance(length, Var):
      self.nonneg.add(length.uid)
    return start, step, length

  def _expand_index(self, arr: ArrayVal, index) -> list:
    items = list(index) if isinstance(index, tuple) else [index]
    n_real = sum(1 for it in items if it is not None and it is not Ellipsis and not self._is_mask(it))
    n_real += sum(it.rank for it in items if self._is_mask(it))
    if sum(1 for it in items if it is Ellipsis) > 1:
      raise self.error(StaticValueError, "an index can only have a single ellipsis ('...')")
    if n_real > arr.rank:
      raise self.error(
        StaticValueError,
        f"too many indices for array: array is {arr.rank}-dimensional, but {n_real} were indexed",
      )
    fill = [slice(None)] * (arr.rank - n_real)
    if Ellipsis in [it for it in items if it is Ellipsis]:
      k = next(i for i, it in enumerate(items) if it is Ellipsis)
      items[k : k + 1] = fill
    else:
      items += fill
    return items

  @staticmethod
  def _is_mask(it) -> bool:
    return isinstance(it, ArrayVal) and it.elt.kind == "b"

  def _basic_plan(self, arr: ArrayVal, items: list):
    shape: list[Expr] = []
    mapping = []
    k = 0  # source dimension
    for it in items:
      if it is None:
        shape.append(ONE)
        continue
      dim = arr.shape[k]
      if isinstance(it, slice):
        start, step, length = self.slice_params(it, dim)
        mapping.append((len(shape), start, step))
        shape.append(length)
      else:
        mapping.append((None, self.normalize_index(self.as_index(it, "indices"), dim, k), ZERO))
      k += 1
    return shape, mapping

  def getitem(self, arr: ArrayVal, index):
    items = self._expand_index(arr, index)
    items = [
      self.elem(it, ()) if isinstance(it, ArrayVal) and it.rank == 0 and it.elt.kind != "b" else it for it in items
    ]
    if any(isinstance(it, ArrayVal) for it in items):
      shape, src_index, base = self._fancy_plan(arr, items)
      cost = base.cost + 2
      index_roots = [self.get_roots(it) for it in items if isinstance(it, ArrayVal)]
      return self.delayed(
                arr.elt, shape, lambda idx: self.elem(base, src_index(idx)),
                roots=self.union_roots(self.get_roots(base), *index_roots), cost=cost,
            )  # fmt: skip
    shape, mapping = self._basic_plan(arr, items)
    if not shape and all(j is None for j, _, _ in mapping) and arr.rank > 0:
      idx = tuple(start for _, start, _ in mapping)
      if arr.buf is None:
        self.check_fresh(arr)
        if self.loop_depth > arr.depth:
          self.conflict(arr, "indexed inside a loop")
      return self.tmp(self.elem(arr, idx), "x")
    if arr.rank == 0 and not shape:
      return self.tmp(self.elem(arr, ()), "x")
    if (
      len(shape) == arr.rank
      and all(j == k and is_zero(start) and is_one(step) for k, (j, start, step) in enumerate(mapping))
      and all(a is b for a, b in zip(shape, arr.shape))
    ):
      return arr  # x[:], x[...]: same view
    return self.remap(arr, shape, mapping)

  def setitem(self, arr: ArrayVal, index, value) -> None:
    self.force_inplace(arr)
    items = self._expand_index(arr, index)
    items = [
      self.elem(it, ()) if isinstance(it, ArrayVal) and it.rank == 0 and it.elt.kind != "b" else it for it in items
    ]
    if any(isinstance(it, ArrayVal) for it in items):
      shape, src_index, base = self._fancy_plan(arr, items, for_store=True)
      if isinstance(value, ArrayVal):
        value = self.broadcast_to(self.as_concrete(value), shape, check=True, what="in indexed assignment")
      else:
        value = self.tmp(ops.cast(self.as_scalar(value, "assigned value"), arr.elt))
      self.record_store(arr)

      def body(idx):
        v = self.elem(value, idx) if isinstance(value, ArrayVal) else value
        self.emit(Store(base.buf, self.index_expr(base, src_index(idx)), ops.cast(v, arr.elt)))

      self.loop_nest(shape, body, parallel=False)
      return
    shape, mapping = self._basic_plan(arr, items)
    if not shape and (arr.rank == 0 or all(j is None for j, _, _ in mapping)):
      idx = tuple(start for _, start, _ in mapping)
      if isinstance(value, ArrayVal):
        if value.rank != 0:
          raise self.error(
            TypeInferenceError,
            f"cannot assign {describe(value)} to a single element (setting an array element with a sequence)",
          )
        value = self.elem(value, ())
      v = ops.cast(self.as_scalar(value, "assigned value"), arr.elt)
      pos = self.tmp(self.index_expr(arr, idx), "pos")
      tape = self.tape
      if tape is not None and not tape.muted and arr.elt.kind == "f" and tape.store_element(self, arr, pos, v):
        return
      self.emit(Store(arr.buf, pos, v))
      self.record_store(arr)
      return
    dest = self.remap(arr, shape, mapping)
    self.store_array(dest, value)

  def safe_inplace(self, dest: ArrayVal, src: ArrayVal) -> bool:
    """True if evaluating ``src`` element by element while writing ``dest``
    is safe: every array it reads either cannot alias ``dest`` or is the
    very same view."""
    droots = self.get_roots(dest)
    if not self.may_alias(self.get_roots(src), droots):
      return True
    reads = (src,) if src.buf is not None else src.reads
    if reads is None:
      return False
    for r in reads:
      if not self.may_alias(self.get_roots(r), droots):
        continue
      identical = (
                r.buf is dest.buf and r.rank == dest.rank and same(r.offset, dest.offset)
                and all(same(a, b) for a, b in zip(r.strides, dest.strides))
            )  # fmt: skip
      if not identical:
        return False
    return True

  def store_array(self, dest: ArrayVal, value) -> None:
    """``dest[...] = value`` with NumPy broadcasting and overlap semantics."""
    tape = self.tape
    if tape is not None and not tape.muted and dest.elt.kind == "f" and tape.store(self, dest, value):
      return
    self.store_array_plain(dest, value)

  def store_array_plain(self, dest: ArrayVal, value) -> None:
    if isinstance(value, ArrayVal):
      if value.rank > dest.rank:
        # NumPy allows leading length-1 dimensions on the right-hand side
        lead = value.rank - dest.rank
        for d in value.shape[:lead]:
          if isinstance(d, Const) and d.value != 1:
            raise self.error(
              TypeInferenceError,
              f"could not broadcast {describe(value)} into a {dest.rank}-dimensional destination",
            )
          self.check(cmp("eq", d, 1), "ValueError", "could not broadcast input array into the destination shape")
        value = self.remap(
                    value, value.shape[lead:],
                    [(None, ZERO, ZERO)] * lead + [(k, ZERO, ONE) for k in range(dest.rank)],
                )  # fmt: skip
      if not self.safe_inplace(dest, value):
        value = self.materialize(value, "tmp")
      else:
        self.consume(value)
      value = self.broadcast_to(value, dest.shape, consumed=True, check=True, what="in assignment")
    else:
      value = self.as_scalar(value, "assigned value")
    self.fill(dest, value)
    self.record_store(dest)

  def _fancy_plan(self, arr: ArrayVal, items: list, for_store: bool = False):
    """Plan integer-array / boolean-mask indexing.

    Returns ``(result_shape, src_index, base)`` where ``src_index(idx)`` maps
    a result index to a full index into the view ``base``.
    """
    # boolean masks -> integer index arrays
    expanded: list = []
    k = 0
    full_mask = False
    for it in items:
      if self._is_mask(it):
        if it.rank == 1:
          dim = arr.shape[k]
          if not self.dim_equal(it.shape[0], dim):
            self.check(
                            cmp("eq", it.shape[0], dim), "IndexError",
                            f"boolean index did not match indexed array along axis {k}; "
                            "size of axis is {1} but size of boolean index is {0}", it.shape[0], dim,
                        )  # fmt: skip
          expanded.append(self.nonzero1d(it))
          k += 1
        elif it.rank == arr.rank and len([x for x in items if x is not None]) == 1:
          for a, b in zip(it.shape, arr.shape):
            if not self.dim_equal(a, b):
              self.check(
                cmp("eq", a, b),
                "IndexError",
                "boolean index did not match indexed array; dimension is {1} but corresponding boolean dimension is {0}",
                a,
                b,
              )
          full_mask = True
          expanded.append(self.nonzero1d(self.flat_view(it)))
          k += it.rank
        else:
          raise self.error(
            UnsupportedError,
            f"boolean mask of rank {it.rank} indexing an array of rank {arr.rank}; "
            "masks must be 1-d or have the same shape as the array",
          )
      else:
        expanded.append(it)
        if it is not None:
          k += 1
    if full_mask:
      if for_store and not (arr.buf is not None and arr.contig):
        # element positions via explicit unravelling of the flat index
        idx_arr = expanded[0]
        shape_src = arr.shape

        def src_full(idx):
          flat = self.tmp(self.elem(idx_arr, idx), "flat")
          return self.unravel(flat, shape_src)

        return idx_arr.shape, src_full, arr
      arr = self.flat_view(arr)
      items = [expanded[0]]
    else:
      items = expanded
    arr_pos = [i for i, it in enumerate(items) if isinstance(it, ArrayVal)]
    first, last = arr_pos[0], arr_pos[-1]
    if arr_pos != list(range(first, last + 1)):
      raise self.error(
        UnsupportedError,
        "index arrays separated by slices or scalars are not supported; index in two steps instead",
      )
    for it in items:
      if isinstance(it, ArrayVal) and it.elt.kind not in "iu":
        raise self.error(
          TypeInferenceError, f"arrays used as indices must be of integer or boolean type (got {it.elt.name})"
        )

    def is_slice(it) -> bool:
      return isinstance(it, slice) or it is None

    # NumPy: when a scalar index is separated from the index arrays by a
    # slice, the indexed dimensions move to the front of the result
    scalar_pos = [i for i, it in enumerate(items) if not is_slice(it) and not isinstance(it, ArrayVal)]
    separated = any(any(is_slice(items[k]) for k in range(min(i, first) + 1, max(i, last))) for i in scalar_pos)
    basic_items = [slice(None) if isinstance(it, ArrayVal) else it for it in items]
    base_shape, mapping = self._basic_plan(arr, basic_items)
    base = self.remap(arr, base_shape, mapping)
    q = sum(1 for it in items[:first] if isinstance(it, slice) or it is None)
    index_arrays = [items[i] for i in arr_pos]
    n = len(index_arrays)
    ishape = self.broadcast_shapes([a.shape for a in index_arrays], "index arrays")
    index_arrays = [self.broadcast_to(self.as_concrete(a) if a.cost > CHEAP_COST else a, ishape) for a in index_arrays]
    for a in index_arrays:
      self.consume(a)
    r = len(ishape)
    dims = base.shape[q : q + n]
    if base.buf is None:
      self.consume(base)
    front = separated and q > 0

    def pick(sub):
      return tuple(self.normalize_index(self.elem(a, sub), dims[m], q + m) for m, a in enumerate(index_arrays))

    if front:
      shape = tuple(ishape) + tuple(base.shape[:q]) + tuple(base.shape[q + n :])

      def src_index(idx):
        return tuple(idx[r : r + q]) + pick(idx[:r]) + tuple(idx[r + q :])

    else:
      shape = tuple(base.shape[:q]) + tuple(ishape) + tuple(base.shape[q + n :])

      def src_index(idx):
        return tuple(idx[:q]) + pick(idx[q : q + r]) + tuple(idx[q + r :])

    return shape, src_index, base

  def nonzero1d(self, mask: ArrayVal) -> ArrayVal:
    """Positions of the true elements of a 1-d boolean array (int64 array)."""
    mask = self.as_concrete(mask) if mask.cost > CHEAP_COST else mask
    self.consume(mask)
    n = mask.shape[0]
    if self.can_parallel((n,)):
      return self._compact(mask, n)
    count = Var("count", I)
    self.mutables.add(count.uid)
    self.nonneg.add(count.uid)
    self.emit(Assign(count, ZERO))
    self.for_range(ZERO, n, lambda i: self.emit(Assign(count, iadd(count, ops.cast(self.elem(mask, (i,)), I)))))
    total = self.tmp(count, "nnz")
    if isinstance(total, Var):
      self.nonneg.add(total.uid)
    out = self.alloc_array(Int64, (total,), label="idx")
    pos = Var("pos", I)
    self.mutables.add(pos.uid)
    self.emit(Assign(pos, ZERO))

    def body(i):
      self.emit(
        If(
          ops.cast(self.elem(mask, (i,)), Bool),
          [Store(out.buf, pos, ops.cast(i, Int64)), Assign(pos, iadd(pos, 1))],
          [],
        )
      )

    self.for_range(ZERO, n, body)
    return out

  def _compact(self, mask: ArrayVal, n: Expr) -> ArrayVal:
    """Parallel compaction: count the true elements of every chunk, turn the
    counts into where each chunk writes, then let every chunk write."""
    nchunks = self.tmp(RtCall("num_chunks", (n, Const(True, Bool)), I), "nchunks")
    if isinstance(nchunks, Var):
      self.nonneg.add(nchunks.uid)
    starts = self.alloc_array(Int64, (nchunks,), label="starts")

    def count_chunk(chunk, start, stop):
      count = Var("count", I)
      self.mutables.add(count.uid)
      self.emit(Assign(count, ZERO))
      self.for_range(start, stop, lambda i: self.emit(Assign(count, iadd(count, ops.cast(self.elem(mask, (i,)), I)))))
      self.emit(Store(starts.buf, chunk, ops.cast(count, Int64)))

    self.par_loop(n, count_chunk, kind="reduce", nchunks=nchunks)
    running = Var("nnz", I)
    self.mutables.add(running.uid)
    self.nonneg.add(running.uid)
    self.emit(Assign(running, ZERO))

    def shift(c):
      here = self.copy_scalar(ops.cast(Load(starts.buf, c), I), "count")
      self.emit(Store(starts.buf, c, ops.cast(running, Int64)))
      self.emit(Assign(running, iadd(running, here)))

    self.for_range(ZERO, nchunks, shift)
    total = self.copy_scalar(running, "nnz")
    if isinstance(total, Var):
      self.nonneg.add(total.uid)
    out = self.alloc_array(Int64, (total,), label="idx")

    def write_chunk(chunk, start, stop):
      pos = Var("pos", I)
      self.mutables.add(pos.uid)
      self.emit(Assign(pos, ops.cast(Load(starts.buf, chunk), I)))

      def body(i):
        self.emit(
          If(
            ops.cast(self.elem(mask, (i,)), Bool),
            [Store(out.buf, pos, ops.cast(i, Int64)), Assign(pos, iadd(pos, 1))],
            [],
          )
        )

      self.for_range(start, stop, body)

    # not a map as far as backends are concerned: each chunk writes from its own start
    self.par_loop(n, write_chunk, kind="reduce", nchunks=nchunks)
    return out

  # ------------------------------------------------------------------ checks

  def check(self, cond: Expr, exc: str, fmt: str, *args: Expr) -> None:
    if isinstance(cond, Const):
      if cond.value:
        return
    self.emit(Check(cond, ErrorInfo(exc, fmt, self.loc()), tuple(ops._i(a) for a in args)))


def inner_work(el, block: list, skip: int) -> Expr:
  """Roughly how many inner loop iterations one element of a parallel loop
  costs: the trip counts of the loops below the ``skip`` levels that walk
  the chunk itself, where they can be computed before the loop starts."""
  from ..backends.cfamily import defined_vars
  from ..lib.core import outer_expr

  inner = set(defined_vars(block, into_parloops=True))

  def trips(s: ForRange) -> Expr:
    span = isub(s.stop, s.start)
    if isinstance(s.step, Const) and s.step.value > 1:
      span = idiv(span, s.step)
    known = outer_expr(el, span, inner)
    return imax(known, 1) if known is not None else Const(8, I)

  def cost(stmts: list, skip: int) -> Expr:
    total: Expr = ZERO
    for s in stmts:
      if isinstance(s, ForRange):
        w = cost(s.body, skip - 1) if skip > 0 else imul(trips(s), imax(cost(s.body, 0), 1))
      elif isinstance(s, While):
        w = cost(s.body, skip - 1) if skip > 0 else imul(Const(8, I), imax(cost(s.body, 0), 1))
      elif isinstance(s, If):
        w = imax(cost(s.then, skip), cost(s.orelse, skip))
      elif isinstance(s, ParLoop):
        w = cost(s.body, skip)
      else:
        continue
      total = iadd(total, w)
    return total

  return imax(cost(block, skip), 1)


def is_zero(e: Expr) -> bool:
  return isinstance(e, Const) and e.value == 0


def is_one(e: Expr) -> bool:
  return isinstance(e, Const) and e.value == 1
