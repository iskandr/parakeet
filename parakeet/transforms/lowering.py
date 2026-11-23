"""LowerAdverbs: turn the operator statements into loops.

``IndexMap`` becomes a chunked parallel loop (or a plain loop nest), and
``IndexReduce`` a fold per chunk whose partial results are combined
afterwards.  This is the last pass; everything after it is the loop IR the
backends print.
"""

from __future__ import annotations

from ..ndtypes import Bool, BufType
from ..syntax import ir, ops
from ..syntax.clone import Cloner, subst_expr
from ..syntax.ir import Alloc, Assign, Break, Const, Expr, ForRange, If, Load, ParLoop, RtCall, Scope, Store, Var, While
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, idiv, imax, imin, imul, iprod, isub


class Loops:
  """Builds loops into a block (no types to infer, no facts to track: that
  was all done when the operator bodies were elaborated)."""

  def __init__(self, block: list, defs: dict[int, Expr] | None = None, owner: Scope | None = None):
    self.block = block
    self.defs = defs if defs is not None else {}
    self.owner = owner  # the scope that releases what is allocated into this block

  def emit(self, s) -> None:
    self.block.append(s)

  def tmp(self, e, name: str = "t") -> Expr:
    if isinstance(e, int):
      return Const(e, I)
    if isinstance(e, (Const, Var)):
      return e
    v = Var(name, e.type)
    self.block.append(Assign(v, e))
    self.defs[v.uid] = e
    return v

  def sub(self, block: list, owner: Scope | None = None) -> Loops:
    return Loops(block, self.defs, owner or self.owner)

  def unravel(self, flat: Expr, shape, into=None) -> tuple[Expr, ...]:
    """Index tuple of flat position ``flat``; with ``into``, assigned to those variables."""
    idx: list[Expr] = []
    rest = flat
    for k in range(len(shape) - 1, 0, -1):
      q = self.tmp(idiv(rest, shape[k]), "q")
      idx.append(isub(rest, imul(q, shape[k])))
      rest = q
    idx.append(rest)
    out = tuple(reversed(idx))
    if into is None:
      return tuple(self.tmp(e, "i") for e in out)
    for v, e in zip(into, out):
      self.emit(Assign(v, e))
    return tuple(into)

  def chunk(self, shape, index: tuple[Var, ...], start: Expr, stop: Expr, body: list, scope: Scope, reduction=None):
    """Run ``body`` for the flat positions ``[start, stop)`` in order, with
    ``index`` bound to each position's index."""
    if len(shape) == 1:
      self.emit(ForRange(index[0], start, stop, ONE, body, scope, True, reduction))
      return
    inner = shape[-1]
    flat = Var("flat", I)
    self.emit(Assign(flat, start))
    blk: list = []
    lp = self.sub(blk)
    lp.emit(If(cmp("ge", flat, stop), [Break()], []))
    row = lp.tmp(idiv(flat, inner), "row")
    j0 = lp.tmp(isub(flat, imul(row, inner)), "j0")
    j1 = lp.tmp(imin(inner, iadd(j0, isub(stop, flat))), "j1")
    lp.unravel(row, shape[:-1], into=index[:-1])
    lp.emit(ForRange(index[-1], j0, j1, ONE, body, scope, True, reduction))
    lp.emit(Assign(flat, iadd(flat, isub(j1, j0))))
    self.emit(While(blk, Scope("loop")))

  def nest(self, shape, index: tuple[Var, ...], body: list, scope: Scope, reduction=None) -> None:
    """Plain loops over ``shape`` in index order."""
    blk, sc = body, scope
    for k in range(len(shape) - 1, -1, -1):
      last = k == len(shape) - 1
      loop = ForRange(index[k], ZERO, shape[k], ONE, blk, sc, last, reduction if last else None)
      blk, sc = [loop], Scope("loop")
    self.block.extend(blk)

  def parallel(
    self, total: Expr, body, kind: str = "map", nchunks: Expr | None = None, loc=None, work=None, heavy=None
  ):
    """A chunked parallel loop; ``body(loops, chunk, start, stop)`` fills one chunk."""
    if nchunks is None:
      nchunks = self.tmp(RtCall("num_chunks", (total, Const(kind == "reduce", Bool)), I), "nchunks")
    chunk, start, stop = Var("chunk", I), Var("start", I), Var("stop", I)
    blk: list = []
    scope = Scope("par")
    body(self.sub(blk, scope), chunk, start, stop)
    blk.append(ir.ReleaseScope(scope))
    if heavy is None:
      heavy = depth(blk) > 1
    self.emit(
      ParLoop(total, nchunks, chunk, start, stop, blk, scope, kind, loc, heavy, work if kind == "map" else None)
    )
    return nchunks


def depth(block: list) -> int:
  """Loop nesting of a block (operator statements count as the loops they become)."""
  best = 0
  for s in block:
    inner = max((depth(b) for b in ir.sub_blocks(s)), default=0)
    if isinstance(s, (ForRange, While)):
      inner += 1
    elif isinstance(s, (ir.IndexMap, ir.IndexReduce)):
      inner += len(s.shape)
    elif isinstance(s, ir.IndexFilter):
      inner += 1
    best = max(best, inner)
  return best


def outer_expr(defs: dict[int, Expr], e: Expr, inner: set[int], budget: int = 40) -> Expr | None:
  """``e`` rewritten without the variables in ``inner``, by substituting
  their (pure) definitions; None if that is not possible."""
  if isinstance(e, Const):
    return e
  if isinstance(e, Var):
    if e.uid not in inner:
      return e
    d = defs.get(e.uid)
    if d is None or budget <= 0:
      return None
    return outer_expr(defs, d, inner, budget - 1)
  if isinstance(e, ir.Prim):
    args = [outer_expr(defs, a, inner, budget - 1) for a in e.args]
    if any(a is None for a in args):
      return None
    return ops.prim(e.op, tuple(args), e.type)  # type: ignore[arg-type]
  if isinstance(e, ir.Cast):
    a = outer_expr(defs, e.arg, inner, budget - 1)
    return None if a is None else cast(a, e.type)
  return None


def work(defs: dict[int, Expr], block: list, bound: tuple[Var, ...] = ()) -> Expr:
  """Roughly how many inner iterations one run of ``block`` costs, as an
  expression that can be evaluated before the block's loop starts (so not in
  terms of ``bound``, the loop's own variables, or anything the block defines)."""
  inner = {v.uid for s in ir.walk_stmts(block) for v in ir.stmt_defs(s)} | {v.uid for v in bound}

  def known(e: Expr) -> Expr:
    out = outer_expr(defs, e, inner)
    return imax(out, 1) if out is not None else Const(8, I)

  def cost(stmts: list) -> Expr:
    total: Expr = ZERO
    for s in stmts:
      if isinstance(s, ForRange):
        span = isub(s.stop, s.start)
        if isinstance(s.step, Const) and s.step.value > 1:
          span = idiv(span, s.step)
        w = imul(known(span), imax(cost(s.body), 1))
      elif isinstance(s, While):
        w = imul(Const(8, I), imax(cost(s.body), 1))
      elif isinstance(s, If):
        w = imax(cost(s.then), cost(s.orelse))
      elif isinstance(s, (ir.IndexMap, ir.IndexReduce)):
        w = imul(known(iprod(s.shape)), imax(cost(s.body), 1))
      elif isinstance(s, ir.IndexFilter):
        w = imul(known(s.n), imax(cost(s.body), 1))
      elif isinstance(s, ParLoop):
        w = cost(s.body)
      else:
        continue
      total = iadd(total, w)
    return total

  return imax(cost(block), 1)


def pure_defs(fn_body: list) -> dict[int, Expr]:
  """Definitions of the integer variables that are assigned exactly once."""
  count: dict[int, int] = {}
  value: dict[int, Expr] = {}
  for s in ir.walk_stmts(fn_body):
    for v in ir.stmt_defs(s):
      count[v.uid] = count.get(v.uid, 0) + 1
    if isinstance(s, Assign) and not isinstance(s.var.type, BufType) and s.var.type.kind in "iu":
      value[s.var.uid] = s.value
  return {uid: e for uid, e in value.items() if count[uid] == 1}


# ---------------------------------------------------------------------------
# IndexMap
# ---------------------------------------------------------------------------


def lower_map(lp: Loops, s: ir.IndexMap) -> None:
  shape = tuple(lp.tmp(d, "n") for d in s.shape)
  if s.ordered or not s.parallel:
    lp.nest(shape, s.index, s.body, s.scope)
    return
  total = lp.tmp(iprod(shape), "total")
  heavy = depth(s.body) > 0
  est = lp.tmp(work(lp.defs, s.body, s.index), "work") if heavy else None
  lp.parallel(
    total,
    lambda inner, chunk, start, stop: inner.chunk(shape, s.index, start, stop, s.body, s.scope),
    loc=s.loc, work=est, heavy=heavy,
  )  # fmt: skip


# ---------------------------------------------------------------------------
# IndexReduce
# ---------------------------------------------------------------------------


def step(lp: Loops, f: ir.Fold, acc: Var, x: Expr, scratch: Scope) -> None:
  """``acc = combine(acc, x)``; ``scratch`` owns what the combine block allocates."""
  if not f.combine:
    lp.emit(Assign(acc, subst_expr(f.combined, {f.a.uid: acc, f.b.uid: x})))
    return
  c = Cloner({f.a.uid: acc, f.b.uid: lp.tmp(x, "x")})
  lp.block.extend(c.block(f.combine))
  lp.emit(Assign(acc, c.expr(f.combined)))
  lp.owner.owned.extend(c.scope(scratch).owned)  # this copy's buffers go with its surroundings


def first(lp: Loops, s: ir.IndexReduce, at: tuple[Expr, ...], accs: list[Var]) -> None:
  """Start every accumulator from the value at index ``at``."""
  c = Cloner({v.uid: e for v, e in zip(s.index, at)})
  lp.block.extend(c.block(s.body))
  for f, acc in zip(s.folds, accs):
    lp.emit(Assign(acc, cast(c.expr(f.value), acc.type)))
  if s.scope.owned:
    lp.owner.owned.extend(c.scope(s.scope).owned)  # what the copy allocated is released with its surroundings


def identity(op: str | None, t) -> Const | None:
  """The value that changes nothing when folded with ``op``, if there is one."""
  import numpy as np

  kind = t.kind
  if op in ("add", "or", "bitor", "bitxor"):
    return Const(False if kind == "b" else 0, t)
  if op in ("mul", "and"):
    return Const(True if kind == "b" else 1, t)
  if op == "bitand" and kind in "biu":
    return Const(True, t) if kind == "b" else Const(int(np.array(-1).astype(t.strong.name)), t)
  if op in ("min", "max"):
    top = op == "min"
    if kind == "f":
      return Const(float("inf") if top else float("-inf"), t)
    if kind == "b":
      return Const(top, t)
    if kind in "iu":
      info = np.iinfo(t.strong.name)
      return Const(int(info.max if top else info.min), t)
  return None


def hint(s: ir.IndexReduce, accs: list[Var]):
  """Tell backends the loop is a fold that may be vectorised."""
  if s.where is not None:
    return None
  op = s.folds[0].op
  if op in ("add", "mul") and all(f.op == op and not f.combine for f in s.folds):
    return (op, list(accs))
  return None


def lower_reduce(lp: Loops, s: ir.IndexReduce, scratch: list[Scope]) -> None:
  shape = tuple(lp.tmp(d, "n") for d in s.shape)
  total = lp.tmp(iprod(shape), "total")
  folds = s.folds
  seeded = folds[0].init is not None  # all folds of one statement agree on this
  accs = [Var("acc", f.a.type) for f in folds]
  body = list(s.body)

  def steps(inner: Loops, targets: list[Var]) -> list:
    blk = list(body)
    sub = inner.sub(blk, s.scope)
    for f, acc, sc in zip(folds, targets, scratch):
      step(sub, f, acc, f.value, sc)
    if s.where is not None:  # FilterReduce: the element counts only where the predicate holds
      blk = [*s.pre, If(s.where, blk, [])]
    blk.append(ir.ReleaseScope(s.scope))
    return blk

  def check_not_empty() -> None:
    for f in folds:
      if f.empty is not None:
        lp.emit(ir.Check(cmp("gt", total, 0), f.empty))
        return

  if s.parallel and all(f.op is not None for f in folds):
    nchunks = lp.tmp(RtCall("num_chunks", (total, Const(True, Bool)), I), "nchunks")
    partials = []
    for f in folds:
      buf = Var("partials", BufType(f.a.type.strong))
      lp.emit(Alloc(buf, nchunks))
      lp.owner.owned.append(buf)  # for the error path; released below as soon as they are combined
      partials.append(buf)

    def chunk_body(inner: Loops, chunk, start, stop):
      local = [Var("acc", f.a.type) for f in folds]
      if s.where is None:
        first(inner, s, inner.unravel(start, shape), local)
        inner.chunk(shape, s.index, iadd(start, 1), stop, steps(inner, local), s.scope, hint(s, local))
      else:  # the first element may not count: start from the identity instead
        for f, acc in zip(folds, local):
          inner.emit(Assign(acc, identity(f.op, f.a.type)))
        inner.chunk(shape, s.index, start, stop, steps(inner, local), s.scope)
      for f, buf, acc in zip(folds, partials, local):
        inner.emit(Store(buf, chunk, cast(acc, f.a.type.strong)))

    lp.parallel(total, chunk_body, kind="reduce", nchunks=nchunks, loc=s.loc, heavy=depth(body) > 0)

    def combine_partials(c: Expr) -> list:
      blk: list = []
      for f, buf, acc in zip(folds, partials, accs):
        blk.append(Assign(acc, cast(ops.prim(f.op, (acc, cast(Load(buf, c), acc.type)), acc.type), acc.type)))
      return blk

    c = Var("c", I)
    if seeded:
      for f, acc in zip(folds, accs):
        lp.emit(Assign(acc, cast(f.init, acc.type)))
      lp.emit(ForRange(c, ZERO, nchunks, ONE, combine_partials(c), Scope("loop")))
    else:
      check_not_empty()
      for buf, acc in zip(partials, accs):
        lp.emit(Assign(acc, cast(Load(buf, ZERO), acc.type)))
      lp.emit(ForRange(c, ONE, nchunks, ONE, combine_partials(c), Scope("loop")))
    for buf in partials:
      lp.emit(ir.Release(buf))
  elif seeded:
    for f, acc in zip(folds, accs):
      lp.emit(Assign(acc, cast(f.init, acc.type)))
    lp.nest(shape, s.index, steps(lp, accs), s.scope, hint(s, accs))
  else:
    check_not_empty()
    first(lp, s, lp.unravel(ZERO, shape), accs)
    lp.chunk(shape, s.index, ONE, total, steps(lp, accs), s.scope, hint(s, accs))
  for f, acc in zip(folds, accs):
    lp.emit(Assign(f.result, acc))


# ---------------------------------------------------------------------------
# IndexFilter
# ---------------------------------------------------------------------------


def lower_filter(lp: Loops, s: ir.IndexFilter) -> None:
  """Compaction: count, allocate, write.  In parallel: count the survivors
  of every chunk, turn the counts into where each chunk writes, then let
  every chunk write."""
  n = lp.tmp(s.n, "n")

  def counting(inner: Loops, start: Expr, stop: Expr, count: Var) -> None:
    c = Cloner()  # the body runs in both passes
    i = c.var(s.index)
    blk = c.block(s.body)
    blk.append(Assign(count, iadd(count, cast(c.expr(s.keep), I))))
    blk.append(ir.ReleaseScope(c.scope(s.scope)))
    inner.emit(Assign(count, ZERO))
    inner.emit(ForRange(i, start, stop, ONE, blk, c.scope(s.scope), True, ("add", [count])))

  def writing(inner: Loops, start: Expr, stop: Expr, pos: Var) -> None:
    blk = list(s.body)
    blk.append(If(s.keep, [Store(s.out, pos, cast(s.index, s.out.type.elt)), Assign(pos, iadd(pos, 1))], []))
    blk.append(ir.ReleaseScope(s.scope))
    inner.emit(ForRange(s.index, start, stop, ONE, blk, s.scope))

  if not s.parallel:
    count = Var("count", I)
    counting(lp, ZERO, n, count)
    lp.emit(Assign(s.count, count))
    lp.emit(Alloc(s.out, s.count))
    pos = Var("pos", I)
    lp.emit(Assign(pos, ZERO))
    writing(lp, ZERO, n, pos)
    return
  nchunks = lp.tmp(RtCall("num_chunks", (n, Const(True, Bool)), I), "nchunks")
  starts = Var("starts", BufType(I))
  lp.emit(Alloc(starts, nchunks))
  lp.owner.owned.append(starts)  # for the error path; released below

  def count_chunk(inner: Loops, chunk, start, stop):
    count = Var("count", I)
    counting(inner, start, stop, count)
    inner.emit(Store(starts, chunk, count))

  # neither loop is a map as far as backends are concerned: a chunk is a unit
  lp.parallel(n, count_chunk, kind="reduce", nchunks=nchunks, loc=s.loc)
  running, c, here = Var("nnz", I), Var("c", I), Var("count", I)
  lp.emit(Assign(running, ZERO))
  shift = [Assign(here, Load(starts, c)), Store(starts, c, running), Assign(running, iadd(running, here))]
  lp.emit(ForRange(c, ZERO, nchunks, ONE, shift, Scope("loop")))
  lp.emit(Assign(s.count, running))
  lp.emit(Alloc(s.out, s.count))

  def write_chunk(inner: Loops, chunk, start, stop):
    pos = Var("pos", I)
    inner.emit(Assign(pos, Load(starts, chunk)))
    writing(inner, start, stop, pos)

  lp.parallel(n, write_chunk, kind="reduce", nchunks=nchunks, loc=s.loc)
  lp.emit(ir.Release(starts))


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------


def lower_block(block: list, defs: dict[int, Expr], owner: Scope) -> list:
  """``owner`` is the scope whose release covers buffers allocated in ``block``."""
  out: list = []
  for s in block:
    if isinstance(s, ir.IndexReduce):
      s.body[:] = lower_block(s.body, defs, s.scope)
      s.pre[:] = lower_block(s.pre, defs, s.scope)
      scratch = [Scope("fold") for _ in s.folds]
      for f, sc in zip(s.folds, scratch):
        f.combine[:] = lower_block(f.combine, defs, sc)
      lower_reduce(Loops(out, defs, owner), s, scratch)
      continue
    if isinstance(s, ir.IndexMap):
      s.body[:] = lower_block(s.body, defs, s.scope)
      lower_map(Loops(out, defs, owner), s)
      continue
    if isinstance(s, ir.IndexFilter):
      s.body[:] = lower_block(s.body, defs, s.scope)
      lower_filter(Loops(out, defs, owner), s)
      continue
    if isinstance(s, If):
      s.then[:] = lower_block(s.then, defs, owner)
      s.orelse[:] = lower_block(s.orelse, defs, owner)
    elif isinstance(s, (ForRange, While, ParLoop)):
      s.body[:] = lower_block(s.body, defs, s.scope)
    out.append(s)
  return out


def lower(fn: ir.Function) -> None:
  """Replace every operator statement of ``fn`` by loops."""
  fn.body[:] = lower_block(fn.body, pure_defs(fn.body), fn.scope)


__all__ = ["Loops", "depth", "identity", "lower", "lower_block", "outer_expr", "pure_defs", "work"]
