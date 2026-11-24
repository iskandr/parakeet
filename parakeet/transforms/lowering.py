"""LowerAdverbs: turn the operator statements into loops.

``IndexMap`` becomes a chunked parallel loop (or a plain loop nest), and
``IndexReduce`` a fold per chunk whose partial results are combined
afterwards.  This is the last pass; everything after it is the loop IR the
backends print.
"""

from __future__ import annotations

from ..ndtypes import Bool, BufType
from ..syntax import ir, ops
from ..syntax.clone import Cloner
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

  def chunk(
    self,
    shape,
    index: tuple[Var, ...],
    start: Expr,
    stop: Expr,
    body: list,
    scope: Scope,
    reduction=None,
    simd: bool = True,
  ):
    """Run ``body`` for the flat positions ``[start, stop)`` in order, with
    ``index`` bound to each position's index."""
    if len(shape) == 1:
      self.emit(ForRange(index[0], start, stop, ONE, body, scope, simd, reduction))
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
    lp.emit(ForRange(index[-1], j0, j1, ONE, body, scope, simd, reduction))
    lp.emit(Assign(flat, iadd(flat, isub(j1, j0))))
    self.emit(While(blk, Scope("loop")))

  def nest(self, shape, index: tuple[Var, ...], body: list, scope: Scope, reduction=None, simd: bool = True) -> None:
    """Plain loops over ``shape`` in index order."""
    blk, sc = body, scope
    for k in range(len(shape) - 1, -1, -1):
      last = k == len(shape) - 1
      loop = ForRange(index[k], ZERO, shape[k], ONE, blk, sc, last and simd, reduction if last else None)
      blk, sc = [loop], Scope("loop")
    self.block.extend(blk)

  def parallel(
    self,
    total: Expr,
    body,
    kind: str = "map",
    nchunks: Expr | None = None,
    loc=None,
    work=None,
    heavy=None,
    shared_tile=None,
    monoidal_tile=None,
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
      ParLoop(
        total,
        nchunks,
        chunk,
        start,
        stop,
        blk,
        scope,
        kind,
        loc,
        heavy,
        work if kind == "map" else None,
        shared_tile=shared_tile,
        monoidal_tile=monoidal_tile,
      )
    )
    return nchunks


def depth(block: list) -> int:
  """Loop nesting of a block (operator statements count as the loops they become)."""
  best = 0
  for s in block:
    inner = max((depth(b) for b in ir.sub_blocks(s)), default=0)
    if isinstance(s, (ForRange, While)):
      inner += 1
    elif isinstance(s, (ir.IndexMap, ir.IndexReduce, ir.IndexScatter)):
      inner += len(s.shape)
    elif isinstance(s, ir.IndexTile):
      inner += sum(1 for d in s.tile_shape if d is not None)
    elif isinstance(s, (ir.IndexFilter, ir.IndexScan)):
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
      elif isinstance(s, (ir.IndexMap, ir.IndexReduce, ir.IndexScatter)):
        w = imul(known(iprod(s.shape)), imax(cost(s.body), 1))
      elif isinstance(s, ir.IndexTile):
        # Total tiles times inner tile work (approximated by inner body cost)
        tiled = [d for d, b in zip(s.shape, s.tile_shape) if b is not None]
        w = imul(known(iprod(tiled)) if tiled else ONE, imax(cost(s.body), 1))
      elif isinstance(s, (ir.IndexFilter, ir.IndexScan)):
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


def step(
  lp: Loops,
  f: ir.Fold,
  acc: Var,
  x: Expr,
  scratch: Scope,
  p_accs: list[Var] = (),  # type: ignore[assignment]
  p_xs: list[Expr] = (),  # type: ignore[assignment]
) -> None:
  """Fold ``x`` (and any carried states ``p_xs``) into ``acc`` (and ``p_accs``)."""
  subst: dict[int, Expr] = {f.a.uid: acc, f.b.uid: x if not f.combine else lp.tmp(x, "x")}
  for c, pa, px in zip(f.carries, p_accs, p_xs):
    subst[c.a.uid] = pa
    subst[c.b.uid] = px if not f.combine else lp.tmp(px, "px")
  c = Cloner(subst)
  lp.block.extend(c.block(f.combine))
  destinations = [acc, *p_accs]
  values = [c.expr(f.combined), *(c.expr(cy.combined) for cy in f.carries)]
  if f.carries:
    # Tuple assignment must also snapshot bare accumulator variables (e.g. swaps).
    saved = [Var("next", v.type) for v in values]
    lp.block.extend(Assign(tmp, v) for tmp, v in zip(saved, values))
    values = saved
  lp.block.extend(Assign(dst, val) for dst, val in zip(destinations, values))
  lp.owner.owned.extend(c.scope(scratch).owned)  # this copy's buffers go with its surroundings


def first(lp: Loops, s: ir.IndexReduce, at: tuple[Expr, ...], accs: list[Var], p_accs: list[list[Var]]) -> None:
  """Start every accumulator (and its carried states) from the value at index ``at``."""
  c = Cloner({v.uid: e for v, e in zip(s.index, at)})
  lp.block.extend(c.block(s.body))
  for f, acc, pas in zip(s.folds, accs, p_accs):
    lp.emit(Assign(acc, cast(c.expr(f.value), acc.type)))
    for cy, pa in zip(f.carries, pas):
      lp.emit(Assign(pa, cast(c.expr(cy.value), pa.type)))
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
  if op in ("add", "mul") and all(f.op == op and not f.combine and not f.carries for f in s.folds):
    return (op, list(accs))
  return None


def lower_reduce(lp: Loops, s: ir.IndexReduce, scratch: list[Scope]) -> None:
  shape = tuple(lp.tmp(d, "n") for d in s.shape)
  total = lp.tmp(iprod(shape), "total")
  folds = s.folds
  seeded = folds[0].init is not None  # all folds of one statement agree on this
  accs = [Var("acc", f.a.type) for f in folds]
  p_accs = [[Var("pacc", cy.a.type) for cy in f.carries] for f in folds]
  body = list(s.body)

  def steps(inner: Loops, targets: list[Var], p_targets: list[list[Var]]) -> list:
    blk = list(body)
    sub = inner.sub(blk, s.scope)
    for f, acc, pas, sc in zip(folds, targets, p_targets, scratch):
      step(sub, f, acc, f.value, sc, pas, [cy.value for cy in f.carries])
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
    p_partials: list[list[Var]] = []
    for f in folds:
      buf = Var("partials", BufType(f.a.type.strong))
      lp.emit(Alloc(buf, nchunks))
      lp.owner.owned.append(buf)  # for the error path; released below as soon as they are combined
      partials.append(buf)
      pbufs: list[Var] = []
      for cy in f.carries:
        pbuf = Var("p_partials", BufType(cy.a.type.strong))
        lp.emit(Alloc(pbuf, nchunks))
        lp.owner.owned.append(pbuf)
        pbufs.append(pbuf)
      p_partials.append(pbufs)

    def chunk_body(inner: Loops, chunk, start, stop):
      local = [Var("acc", f.a.type) for f in folds]
      p_local = [[Var("pacc", cy.a.type) for cy in f.carries] for f in folds]
      if s.where is None:
        first(inner, s, inner.unravel(start, shape), local, p_local)
        inner.chunk(shape, s.index, iadd(start, 1), stop, steps(inner, local, p_local), s.scope, hint(s, local))
      else:  # the first element may not count: start from the identity instead
        for f, acc in zip(folds, local):
          inner.emit(Assign(acc, identity(f.op, f.a.type)))
        inner.chunk(shape, s.index, start, stop, steps(inner, local, p_local), s.scope)
      for f, buf, acc, pbufs, pas in zip(folds, partials, local, p_partials, p_local):
        inner.emit(Store(buf, chunk, cast(acc, f.a.type.strong)))
        for cy, pbuf, pa in zip(f.carries, pbufs, pas):
          inner.emit(Store(pbuf, chunk, cast(pa, cy.a.type.strong)))

    lp.parallel(total, chunk_body, kind="reduce", nchunks=nchunks, loc=s.loc, heavy=depth(body) > 0)

    def combine_partials(c: Expr) -> list:
      blk: list = []
      sub = lp.sub(blk)
      for f, buf, acc, pbufs, pas, sc in zip(folds, partials, accs, p_partials, p_accs, scratch):
        loaded = cast(Load(buf, c), acc.type)
        if f.carries or f.combine or f.op == "structured":
          p_loaded = [cast(Load(pbuf, c), pa.type) for pbuf, pa in zip(pbufs, pas)]
          step(sub, f, acc, loaded, sc, pas, p_loaded)
        else:
          blk.append(Assign(acc, cast(ops.prim(f.op, (acc, loaded), acc.type), acc.type)))
      return blk

    c = Var("c", I)
    if seeded:
      for f, acc in zip(folds, accs):
        lp.emit(Assign(acc, cast(f.init, acc.type)))
      lp.emit(ForRange(c, ZERO, nchunks, ONE, combine_partials(c), Scope("loop")))
    else:
      check_not_empty()
      for buf, acc, pbufs, pas in zip(partials, accs, p_partials, p_accs):
        lp.emit(Assign(acc, cast(Load(buf, ZERO), acc.type)))
        for pbuf, pa in zip(pbufs, pas):
          lp.emit(Assign(pa, cast(Load(pbuf, ZERO), pa.type)))
      lp.emit(ForRange(c, ONE, nchunks, ONE, combine_partials(c), Scope("loop")))
    for buf in partials:
      lp.emit(ir.Release(buf))
    for pbufs in p_partials:
      for pbuf in pbufs:
        lp.emit(ir.Release(pbuf))
  elif seeded:
    for f, acc in zip(folds, accs):
      lp.emit(Assign(acc, cast(f.init, acc.type)))
    lp.nest(shape, s.index, steps(lp, accs, p_accs), s.scope, hint(s, accs))
  else:
    check_not_empty()
    first(lp, s, lp.unravel(ZERO, shape), accs, p_accs)
    lp.chunk(shape, s.index, ONE, total, steps(lp, accs, p_accs), s.scope, hint(s, accs))
  for f, acc, pas in zip(folds, accs, p_accs):
    lp.emit(Assign(f.result, acc))
    for cy, pa in zip(f.carries, pas):
      lp.emit(Assign(cy.result, pa))


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
# IndexScan
# ---------------------------------------------------------------------------


def lower_scan(lp: Loops, s: ir.IndexScan, scratch: Scope) -> None:
  """A loop that carries the accumulator; when the combiner is associative,
  two parallel passes instead: fold every chunk, turn the chunk totals into
  each chunk's starting value, then scan the chunks."""
  f = s.fold
  n = lp.tmp(s.n, "n")
  seeded = f.init is not None
  strong = f.a.type.strong

  def value_at(inner: Loops, at: Expr, acc: Var, emit: bool, join: Expr | None = None, unless: Expr | None = None):
    """``acc`` = the value at ``at`` (folded onto ``join`` if given, except
    where ``unless`` holds), from a copy of the body."""
    c = Cloner({s.index.uid: at})
    result = c.var(f.result)
    inner.block.extend(c.block(s.body))
    x = inner.tmp(c.expr(f.value), "x")
    if join is None:
      inner.emit(Assign(acc, cast(x, acc.type)))
    else:
      joined: list = [Assign(acc, join)]
      step(inner.sub(joined), f, acc, x, scratch)
      inner.block.extend(joined if unless is None else [If(unless, [Assign(acc, cast(x, acc.type))], joined)])
    if emit:
      inner.emit(Assign(result, acc))
      inner.block.extend(c.stmt(t) for t in s.emit)
    inner.owner.owned.extend(c.scope(s.scope).owned)

  def rest(inner: Loops, start: Expr, stop: Expr, acc: Var, emit: bool, fresh: bool) -> None:
    """Fold (and with ``emit``, emit) the indices ``[start, stop)`` onto ``acc``."""
    c = Cloner() if fresh else None
    index = s.index if c is None else c.var(s.index)
    scope = s.scope if c is None else c.scope(s.scope)
    blk = list(s.body) if c is None else c.block(s.body)
    step(inner.sub(blk, scope), f, acc, f.value if c is None else c.expr(f.value), scratch)
    if emit:
      result = f.result if c is None else c.var(f.result)
      blk.append(Assign(result, acc))
      blk.extend(s.emit if c is None else [c.stmt(t) for t in s.emit])
    blk.append(ir.ReleaseScope(scope))
    inner.emit(ForRange(index, start, stop, ONE, blk, scope))

  if not s.parallel:
    acc = Var("acc", f.a.type)
    if seeded:
      lp.emit(Assign(acc, cast(f.init, acc.type)))
      rest(lp, ZERO, n, acc, True, False)
    else:
      head: list = []
      value_at(lp.sub(head), ZERO, acc, True)
      lp.emit(If(cmp("gt", n, 0), head, []))
      rest(lp, ONE, n, acc, True, False)
    return

  nchunks = lp.tmp(RtCall("num_chunks", (n, Const(True, Bool)), I), "nchunks")
  carries = Var("carries", BufType(strong))
  lp.emit(Alloc(carries, nchunks))
  lp.owner.owned.append(carries)  # for the error path; released below

  def fold_chunk(inner: Loops, chunk, start, stop):
    acc = Var("acc", f.a.type)
    value_at(inner, start, acc, False)
    rest(inner, iadd(start, 1), stop, acc, False, True)
    inner.emit(Store(carries, chunk, cast(acc, strong)))

  # neither loop is a map as far as backends are concerned: a chunk is a unit
  lp.parallel(n, fold_chunk, kind="reduce", nchunks=nchunks, loc=s.loc)
  # carries[c] becomes the fold of everything before chunk c
  carry, c, total = Var("carry", f.a.type), Var("c", I), Var("total", f.a.type)
  shift: list = [Assign(total, cast(Load(carries, c), total.type)), Store(carries, c, cast(carry, strong))]
  step(lp.sub(shift), f, carry, total, scratch)
  if seeded:
    lp.emit(Assign(carry, cast(f.init, carry.type)))
    lp.emit(ForRange(c, ZERO, nchunks, ONE, shift, Scope("loop")))
  else:
    some = [Assign(carry, cast(Load(carries, ZERO), carry.type)), ForRange(c, ONE, nchunks, ONE, shift, Scope("loop"))]
    lp.emit(If(cmp("gt", nchunks, 0), some, []))

  def scan_chunk(inner: Loops, chunk, start, stop):
    acc = Var("acc", f.a.type)
    before = cast(Load(carries, chunk), acc.type)
    # without an init the first chunk has nothing before it
    value_at(inner, start, acc, True, join=before, unless=None if seeded else cmp("eq", chunk, 0))
    rest(inner, iadd(start, 1), stop, acc, True, False)

  lp.parallel(n, scan_chunk, kind="reduce", nchunks=nchunks, loc=s.loc)
  lp.emit(ir.Release(carries))


# ---------------------------------------------------------------------------
# IndexTile
# ---------------------------------------------------------------------------


def row_major_index(index: tuple[Expr, ...], shape: tuple[Expr, ...]) -> Expr:
  """Flat row-major offset of ``index`` inside ``shape``."""
  if not index:
    return ZERO
  acc = index[0]
  for i, d in zip(index[1:], shape[1:]):
    acc = iadd(imul(acc, d), i)
  return acc


def emit_tile_stage(lp: Loops, st: ir.TileStage, defs: dict[int, Expr], owner: Scope) -> None:
  """Fill a tile-local staged buffer ``st.buf`` over ``st.shape``."""
  shape = tuple(lp.tmp(d, "sn") for d in st.shape)
  count = lp.tmp(iprod(shape), "scount")
  lp.emit(Alloc(st.buf, count))
  owner.owned.append(st.buf)
  body = lower_block(list(st.body), defs, st.scope)
  body.append(Store(st.buf, row_major_index(st.index, shape), cast(st.value, st.buf.type.elt)))
  body.append(ir.ReleaseScope(st.scope))
  lp.nest(shape, st.index, body, st.scope)


def jam_register_tile(lp: Loops, s: ir.IndexTile, defs: dict[int, Expr], owner: Scope) -> bool:
  """Lower a compile-time ``level="register"`` N-D micro-tile whose body is
  ``before; IndexReduce; after`` by jamming all ``prod(tile_shape)`` copies of
  the inner reduction into one multi-fold ``IndexReduce`` (generalized N-D
  register blocking)."""
  import itertools

  from ..syntax.clone import defined

  if s.stages or s.scope.owned:
    return False
  body = [t for t in s.body if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
  # Body must be an inner IndexMap over ``s.extent`` whose body is before; IndexReduce; after
  if len(body) != 1 or not isinstance(body[0], ir.IndexMap) or body[0].ordered or body[0].scope.owned:
    return False
  imap = body[0]
  if len(imap.shape) != len(s.shape):
    return False
  factors: list[int] = []
  for b in s.tile_shape:
    if b is None:
      factors.append(1)
    elif isinstance(b, Const) and b.value >= 1:
      factors.append(int(b.value))
    else:
      return False
  if all(f == 1 for f in factors):
    return False
  ibody = [t for t in imap.body if not (isinstance(t, ir.ReleaseScope) and t.scope is imap.scope)]
  where = [k for k, t in enumerate(ibody) if isinstance(t, ir.IndexReduce)]
  if len(where) != 1:
    return False
  red = ibody[where[0]]
  if red.where is not None:
    return False
  before, after = ibody[: where[0]], ibody[where[0] + 1 :]
  if any(isinstance(t, ir.Store) for t in before):
    return False

  # Reduction extent must not depend on the jammed element coordinates
  elem_vars = {v.uid for v in (*s.origin, *s.extent, *imap.index)} | set(defined(before))
  if any(outer_expr(defs, d, elem_vars) is None for d in red.shape):
    return False

  shape = tuple(lp.tmp(d, "rn") for d in s.shape)

  def emit_single(inner: Loops, coords: tuple[Expr, ...]) -> None:
    # Single element at global-within-tile ``coords``: origin = coords, extent = 1, imap.index = 0
    subst: dict[int, Expr] = {}
    for o, e, v, c in zip(s.origin, s.extent, imap.index, coords):
      subst[o.uid] = c
      subst[e.uid] = ONE
      subst[v.uid] = ZERO
    cl = Cloner(subst)
    inner.block.extend(lower_block(cl.block(before + [red] + after), defs, owner))

  def emit_jammed(inner: Loops, base_coords: tuple[Expr, ...], block_sizes: tuple[int, ...]) -> None:
    """Jam all ``prod(block_sizes)`` elements starting at ``base_coords`` into one IndexReduce."""
    k_index = tuple(Var("k", I) for _ in red.index)
    r_scope = Scope("loop")
    r_body: list = []
    r_folds: list = []
    tails: list[list] = []
    local_defs = defined(before + [red] + after).values()
    offsets = list(itertools.product(*(range(b) for b in block_sizes)))
    for off in offsets:
      subst: dict[int, Expr] = {}
      for o, e, v, bc, d in zip(s.origin, s.extent, imap.index, base_coords, off):
        subst[o.uid] = iadd(bc, d) if d else bc
        subst[e.uid] = ONE
        subst[v.uid] = ZERO
      for kv, nk in zip(red.index, k_index):
        subst[kv.uid] = nk
      cl = Cloner(subst)
      for v in local_defs:
        cl.var(v)
      inner.block.extend(lower_block([cl.stmt(t) for t in before], defs, owner))
      r_body.extend(cl.stmt(t) for t in red.body)
      r_folds.extend(cl.fold(f) for f in red.folds)
      r_scope.owned.extend(cl.scope(red.scope).owned)
      tails.append([cl.stmt(t) for t in after])
    c0 = Cloner({o.uid: bc for o, bc in zip(s.origin, base_coords)})
    r_extent = tuple(c0.expr(d) for d in red.shape)
    fused = ir.IndexReduce(r_extent, k_index, r_body, r_folds, r_scope, red.home, False, red.loc)
    inner.block.extend(lower_block([fused], defs, owner))
    for tail in tails:
      inner.block.extend(lower_block(tail, defs, owner))

  # Partition the N-D box ``shape`` into disjoint hyper-rectangles: each axis k
  # with factor f > 1 splits into [0, full_k) (step f, block size f) and
  # [full_k, shape[k]) (step 1, block size 1). Every leaf with prod(block_sizes) > 1
  # jams its full sub-block (including partial remainder faces along full prefix axes).
  fulls = [lp.tmp(imul(idiv(d, f), f), "jam_end") if f > 1 else d for d, f in zip(shape, factors)]

  def recurse(
    inner: Loops,
    axis: int,
    coords: tuple[Expr, ...],
    block_sizes: tuple[int, ...],
    all_full: bool,
  ) -> None:
    if axis == len(shape):
      if any(b > 1 for b in block_sizes):
        emit_jammed(inner, coords, block_sizes)
      else:
        emit_single(inner, coords)
      return
    f = factors[axis]
    if f == 1 or not all_full:
      v = Var("i", I)
      blk: list = []
      sc = Scope("loop")
      recurse(inner.sub(blk, sc), axis + 1, (*coords, v), (*block_sizes, 1), all_full)
      inner.emit(ForRange(v, ZERO, shape[axis], ONE, blk, sc))
      return
    # Full micro-tile range [0, fulls[axis]) with step f
    v_full = Var("i", I)
    blk_full: list = []
    sc_full = Scope("loop")
    recurse(inner.sub(blk_full, sc_full), axis + 1, (*coords, v_full), (*block_sizes, f), True)
    inner.emit(ForRange(v_full, ZERO, fulls[axis], Const(f, I), blk_full, sc_full))
    # Remainder [fulls[axis], shape[axis]) with step 1
    v_rem = Var("i", I)
    blk_rem: list = []
    sc_rem = Scope("loop")
    recurse(inner.sub(blk_rem, sc_rem), axis + 1, (*coords, v_rem), (*block_sizes, 1), False)
    inner.emit(ForRange(v_rem, fulls[axis], shape[axis], ONE, blk_rem, sc_rem))

  recurse(lp, 0, (), (), True)
  return True


def lower_tile(lp: Loops, s: ir.IndexTile, defs: dict[int, Expr], owner: Scope) -> None:
  """Lower an N-dimensional :class:`~parakeet.syntax.ir.IndexTile` at any level
  (``"cache"``, ``"shared"``, or ``"register"``)."""
  if s.level == "register" and jam_register_tile(lp, s, defs, owner):
    return

  shape = tuple(lp.tmp(d, "tn") for d in s.shape)
  tile_sizes = tuple(None if b is None else lp.tmp(b, "tile") for b in s.tile_shape)
  tiled_axes = [k for k, b in enumerate(tile_sizes) if b is not None]

  def emit_tile_body(inner: Loops, tile_indices: tuple[Expr, ...], tile_owner: Scope) -> None:
    ti_iter = iter(tile_indices)
    for o, e, dim, b in zip(s.origin, s.extent, shape, tile_sizes):
      if b is None:
        inner.emit(Assign(o, ZERO))
        inner.emit(Assign(e, dim))
      else:
        t_idx = next(ti_iter)
        if isinstance(b, Const) and b.value == 1:
          inner.emit(Assign(o, t_idx))
          inner.emit(Assign(e, ONE))
        else:
          orig = inner.tmp(imul(t_idx, b), "t0")
          inner.emit(Assign(o, orig))
          inner.emit(Assign(e, imin(b, isub(dim, orig))))
    for st in s.stages:
      emit_tile_stage(inner, st, defs, tile_owner)
    inner.block.extend(lower_block(list(s.body), defs, s.scope))
    for st in s.stages:
      inner.emit(ir.Release(st.buf))
    inner.emit(ir.ReleaseScope(s.scope))

  if not tiled_axes:
    emit_tile_body(lp, (), owner)
    return

  ntiles = tuple(
    shape[k]
    if (isinstance(tile_sizes[k], Const) and tile_sizes[k].value == 1)
    else lp.tmp(idiv(isub(iadd(shape[k], tile_sizes[k]), 1), tile_sizes[k]), "ntiles")  # type: ignore[arg-type]
    for k in tiled_axes
  )

  if not s.parallel:
    # Sequential N-D loop nest over tiles
    t_vars = tuple(Var("t", I) for _ in tiled_axes)
    inner_blk: list = []
    emit_tile_body(lp.sub(inner_blk, owner), t_vars, owner)
    lp.nest(ntiles, t_vars, inner_blk, Scope("loop"))
    return

  total_tiles = lp.tmp(iprod(ntiles), "total_tiles")
  heavy = depth(s.body) > 0
  est = lp.tmp(work(defs, s.body, (*s.origin, *s.extent)), "work") if heavy else None

  def chunk_body(inner: Loops, _chunk, start: Expr, stop: Expr) -> None:
    flat = Var("tile", I)
    one_blk: list = []
    one_lp = inner.sub(one_blk, inner.owner)
    t_coords = (flat,) if len(ntiles) == 1 else one_lp.unravel(flat, ntiles)
    emit_tile_body(one_lp, t_coords, inner.owner)
    inner.emit(ForRange(flat, start, stop, ONE, one_blk, Scope("loop")))

  work_blk: list = []
  lp.sub(work_blk, owner).parallel(
    total_tiles,
    chunk_body,
    loc=s.loc,
    work=est,
    heavy=heavy,
    shared_tile=s.shared_tile,
    monoidal_tile=s.monoidal_tile,
  )
  lp.emit(If(cmp("gt", total_tiles, 0), work_blk, []))


PRIVATIZE_MAX = 65536  # max destination elements for thread-privatized parallel scatter
PRIVATIZE_MIN_WORK = 4096  # minimum scatter iterations before parallelizing


def lower_scatter(lp: Loops, s: ir.IndexScatter, target=None) -> None:
  """Lower an :class:`~parakeet.syntax.ir.IndexScatter` (``ReduceByIndex``).

  On CUDA targets, associative ``add`` scatters for native atomic element types
  (``float32``, ``float64``, ``int32``, ``uint32``) lower directly to a single
  parallel loop using hardware ``atomicAdd`` with zero scratch memory.
  Otherwise, when ``s.parallel`` is True and ``s.op`` is associative with an
  identity (``add``, ``mul``, ``min``, ``max``), small-to-medium destinations
  (``<= PRIVATIZE_MAX``) run in parallel via thread-privatized partial buffers
  combined into ``s.buf``. Larger or sequential scatters lower to a tight
  sequential loop nest."""
  shape = tuple(lp.tmp(d, "sn") for d in s.shape)
  total = lp.tmp(iprod(shape), "total")
  dests = [u.buf for u in s.updates]

  def iteration(cl, dests, base=ZERO, atomic=False):
    body = cl.block(s.body)
    pos = cl.expr(s.target)
    pos = pos if base is ZERO else iadd(base, pos)
    stores = []
    for u, dst in zip(s.updates, dests, strict=True):
      elt = dst.type.elt
      val = cast(cl.expr(u.value), elt)
      if not atomic and u.op != "assign":
        val = ops.prim(u.op, (Load(dst, pos), val), elt)
      stores.append(Store(dst, pos, val, atomic="add" if atomic else None))
    body.extend(stores if s.where is None else [If(cl.expr(s.where), stores, [])])
    body.append(ir.ReleaseScope(cl.scope(s.scope)))
    return body

  atomic_types = {"float32", "float64", "int32", "uint32"}
  if (
    target is not None
    and target.kind == "cuda"
    and s.parallel
    and all(u.op == "add" and u.buf.type.elt.name in atomic_types for u in s.updates)
  ):
    cl = Cloner()
    indices = tuple(cl.var(v) for v in s.index)
    body = iteration(cl, dests, atomic=True)
    heavy = depth(s.body) > 0
    est = lp.tmp(work(lp.defs, s.body, s.index), "work") if heavy else None
    lp.parallel(
      total,
      lambda sub, ch, a, b: sub.chunk(shape, indices, a, b, body, cl.scope(s.scope), simd=False),
      loc=s.loc,
      work=est,
      heavy=heavy,
    )
    return

  seq = []
  cl = Cloner()
  indices = tuple(cl.var(v) for v in s.index)
  lp.sub(seq).nest(shape, indices, iteration(cl, dests), cl.scope(s.scope), simd=False)
  if not s.parallel or any(identity(u.op, u.buf.type.elt) is None for u in s.updates):
    lp.block.extend(seq)
    return

  dsize = lp.tmp(s.dest_size, "dsize")
  can_priv = ops.land(
    cmp("ge", total, Const(PRIVATIZE_MIN_WORK, I)),
    ops.land(cmp("gt", dsize, ZERO), cmp("le", dsize, Const(PRIVATIZE_MAX // len(dests), I))),
  )
  par = []
  sub = lp.sub(par)
  nchunks = sub.tmp(RtCall("num_chunks", (total, Const(True, Bool)), I), "nchunks")
  size = sub.tmp(imul(nchunks, dsize), "priv_n")
  private = [Var("priv", dst.type) for dst in dests]
  for u, buf in zip(s.updates, private, strict=True):
    sub.emit(Alloc(buf, size, zero=u.op == "add"))
    lp.owner.owned.append(buf)
    if u.op != "add":
      j = Var("j", I)
      init = Store(buf, j, identity(u.op, buf.type.elt))
      sub.parallel(
        size, lambda inner, ch, a, b: inner.emit(ForRange(j, a, b, ONE, [init], Scope("loop"), simd=True)), loc=s.loc
      )

  def scatter_chunk(inner, chunk, start, stop):
    cl = Cloner()
    indices = tuple(cl.var(v) for v in s.index)
    base = inner.tmp(imul(chunk, dsize), "pbase")
    inner.chunk(shape, indices, start, stop, iteration(cl, private, base), cl.scope(s.scope), simd=False)

  sub.parallel(total, scatter_chunk, kind="reduce", nchunks=nchunks, loc=s.loc, heavy=depth(s.body) > 0)

  def combine_chunk(inner, chunk, start, stop):
    j, c = Var("j", I), Var("c", I)
    init, steps, stores = [], [], []
    for u, buf in zip(s.updates, private, strict=True):
      acc = Var("acc", buf.type.elt)
      init.append(Assign(acc, Load(u.buf, j)))
      steps.append(Assign(acc, ops.prim(u.op, (acc, Load(buf, iadd(imul(c, dsize), j))), acc.type)))
      stores.append(Store(u.buf, j, acc))
    body = [*init, ForRange(c, ZERO, nchunks, ONE, steps, Scope("loop")), *stores]
    inner.emit(ForRange(j, start, stop, ONE, body, Scope("loop"), simd=True))

  sub.parallel(dsize, combine_chunk, kind="map", loc=s.loc)
  par.extend(ir.Release(buf) for buf in private)
  lp.emit(If(can_priv, par, seq))


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------


def lower_block(block: list, defs: dict[int, Expr], owner: Scope, target=None) -> list:
  """``owner`` is the scope whose release covers buffers allocated in ``block``."""
  out: list = []
  for s in block:
    if isinstance(s, ir.IndexTile):
      lower_tile(Loops(out, defs, owner), s, defs, owner)
      continue
    if isinstance(s, ir.IndexScatter):
      s.body[:] = lower_block(s.body, defs, s.scope, target)
      lower_scatter(Loops(out, defs, owner), s, target)
      continue
    if isinstance(s, ir.IndexReduce):
      s.body[:] = lower_block(s.body, defs, s.scope, target)
      s.pre[:] = lower_block(s.pre, defs, s.scope, target)
      scratch = [Scope("fold") for _ in s.folds]
      for f, sc in zip(s.folds, scratch):
        f.combine[:] = lower_block(f.combine, defs, sc, target)
      lower_reduce(Loops(out, defs, owner), s, scratch)
      continue
    if isinstance(s, ir.IndexMap):
      s.body[:] = lower_block(s.body, defs, s.scope, target)
      lower_map(Loops(out, defs, owner), s)
      continue
    if isinstance(s, ir.IndexFilter):
      s.body[:] = lower_block(s.body, defs, s.scope, target)
      lower_filter(Loops(out, defs, owner), s)
      continue
    if isinstance(s, ir.IndexScan):
      s.body[:] = lower_block(s.body, defs, s.scope, target)
      s.emit[:] = lower_block(s.emit, defs, s.scope, target)
      scratch = Scope("fold")
      s.fold.combine[:] = lower_block(s.fold.combine, defs, scratch, target)
      lower_scan(Loops(out, defs, owner), s, scratch)
      continue
    if isinstance(s, If):
      s.then[:] = lower_block(s.then, defs, owner, target)
      s.orelse[:] = lower_block(s.orelse, defs, owner, target)
    elif isinstance(s, (ForRange, While, ParLoop)):
      s.body[:] = lower_block(s.body, defs, s.scope, target)
    out.append(s)
  return out


def lower(fn: ir.Function, target=None) -> None:
  """Replace every operator statement of ``fn`` by loops."""
  fn.body[:] = lower_block(fn.body, pure_defs(fn.body), fn.scope, target)


__all__ = [
  "PRIVATIZE_MAX",
  "PRIVATIZE_MIN_WORK",
  "Loops",
  "depth",
  "emit_tile_stage",
  "identity",
  "jam_register_tile",
  "lower",
  "lower_block",
  "lower_scatter",
  "lower_tile",
  "outer_expr",
  "pure_defs",
  "row_major_index",
  "work",
]
