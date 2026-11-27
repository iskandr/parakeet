"""LowerAdverbs: turn the operator statements into loops.

``IndexMap`` becomes a chunked parallel loop (or a plain loop nest), and
``IndexReduce`` a fold per chunk whose partial results are combined
afterwards.  This is the last pass; everything after it is the loop IR the
backends print.
"""

from __future__ import annotations

from ..ndtypes import Bool, BufType, Int64
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
    warp_reduce=None,
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
        warp_reduce=warp_reduce,
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
    elif isinstance(s, (ir.IndexFilter, ir.IndexExpand, ir.IndexScan)):
      inner += 1
    elif isinstance(s, ir.IndexWavefront):
      inner += 2
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
      elif isinstance(s, ir.IndexExpand):
        w = imul(known(s.n), imax(iadd(cost(s.count_body), cost(s.emit_body)), 1))
      elif isinstance(s, ir.IndexWavefront):
        span = imin(known(s.n), imul(Const(2, I), known(s.band))) if s.band is not None else known(s.n)
        w = imul(imul(known(s.m), span), imax(cost(s.body), 1))
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
    loc=s.loc, work=est, heavy=heavy, warp_reduce=s.warp_reduce,
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
# IndexExpand (1-to-K flatmap / ragged compaction)
# ---------------------------------------------------------------------------


def lower_expand(lp: Loops, s: ir.IndexExpand, target=None) -> None:
  """Two-pass 1-to-K ragged compaction:
  1. Compute ``count[i]`` into ``offsets[i + 1]`` in parallel.
  2. In-place parallel inclusive prefix sum on ``offsets[1..n]`` -> CSR offsets + ``total``.
  3. Allocate output buffers to ``total`` and emit ``k in [0, count[i])`` in parallel.
  When ``s.values`` is empty (count-only DCE), step 3 is skipped completely."""
  n = lp.tmp(s.n, "n")
  n1 = lp.tmp(iadd(n, ONE), "n1")
  lp.emit(Alloc(s.offsets, n1, zero=True))

  # Pass 1: count[i] -> offsets[i + 1]
  def count_range(inner: Loops, start: Expr, stop: Expr) -> None:
    blk = list(s.count_body)
    blk.append(Store(s.offsets, iadd(s.index, ONE), cast(s.count, Int64)))
    blk.append(ir.ReleaseScope(s.count_scope))
    inner.emit(ForRange(s.index, start, stop, ONE, blk, s.count_scope, True))

  if not s.parallel:
    count_range(lp, ZERO, n)
    # Serial in-place inclusive scan of offsets[1..n]
    run = Var("run", Int64)
    lp.emit(Assign(run, Const(0, Int64)))
    ci = Var("ci", I)
    sblk = [
      Assign(run, ops.prim("add", (run, Load(s.offsets, ci)), Int64)),
      Store(s.offsets, ci, run),
    ]
    lp.emit(ForRange(ci, ONE, n1, ONE, sblk, Scope("loop")))
  else:
    w_count = work(lp.defs, s.count_body, (s.index,))
    lp.parallel(n, lambda inner, _c, st, sp: count_range(inner, st, sp), loc=s.loc, work=w_count)

    # Parallel in-place inclusive scan on offsets[1..n]
    is_gpu = target is not None and target.is_gpu
    nchunks = lp.tmp(RtCall("num_chunks", (n, Const(True, Bool)), I), "nchunks")
    carries = Var("carries", BufType(Int64))
    lp.emit(Alloc(carries, nchunks))
    lp.owner.owned.append(carries)

    def sum_chunk(inner: Loops, chunk: Var, st: Expr, sp: Expr) -> None:
      acc = Var("acc", Int64)
      i = Var("i", I)
      inner.emit(Assign(acc, Const(0, Int64)))
      blk = [Assign(acc, ops.prim("add", (acc, Load(s.offsets, iadd(i, ONE))), Int64))]
      inner.emit(ForRange(i, st, sp, ONE, blk, Scope("loop")))
      inner.emit(Store(carries, chunk, acc))

    lp.parallel(n, sum_chunk, kind="reduce", nchunks=nchunks, loc=s.loc)

    def scan_carries(inner: Loops) -> None:
      run = Var("run", Int64)
      c = Var("c", I)
      cur = Var("cur", Int64)
      inner.emit(Assign(run, Const(0, Int64)))
      cblk = [
        Assign(cur, Load(carries, c)),
        Store(carries, c, run),
        Assign(run, ops.prim("add", (run, cur), Int64)),
      ]
      inner.emit(ForRange(c, ZERO, nchunks, ONE, cblk, Scope("loop")))

    if is_gpu:
      lp.parallel(nchunks, lambda inner, _c, _st, _sp: scan_carries(inner), kind="reduce", nchunks=ONE, loc=s.loc)
    else:
      scan_carries(lp)

    def add_chunk(inner: Loops, chunk: Var, st: Expr, sp: Expr) -> None:
      run = Var("run", Int64)
      i = Var("i", I)
      inner.emit(Assign(run, Load(carries, chunk)))
      blk = [
        Assign(run, ops.prim("add", (run, Load(s.offsets, iadd(i, ONE))), Int64)),
        Store(s.offsets, iadd(i, ONE), run),
      ]
      inner.emit(ForRange(i, st, sp, ONE, blk, Scope("loop")))

    lp.parallel(n, add_chunk, kind="reduce", nchunks=nchunks, loc=s.loc)
    lp.emit(ir.Release(carries))

  lp.emit(Assign(s.total, cast(Load(s.offsets, n), I)))

  # Pass 2: emit values (skipped when count-only DCE cleared s.values)
  if not s.values:
    return
  for buf, _ in s.values:
    lp.emit(Alloc(buf, s.total))

  def emit_range(inner: Loops, start: Expr, stop: Expr) -> None:
    ei = Var("ei", I)
    base = Var("base", I)
    end = Var("end", I)
    cnt = Var("cnt", I)
    cl = Cloner({s.index.uid: ei})
    k = cl.var(s.emit_index)
    eblk = cl.block(s.emit_body)
    for buf, val in s.values:
      eblk.append(Store(buf, iadd(base, k), cl.expr(val)))
    eblk.append(ir.ReleaseScope(cl.scope(s.emit_scope)))
    outer_blk = [
      Assign(base, cast(Load(s.offsets, ei), I)),
      Assign(end, cast(Load(s.offsets, iadd(ei, ONE)), I)),
      Assign(cnt, isub(end, base)),
      ForRange(k, ZERO, cnt, ONE, eblk, cl.scope(s.emit_scope)),
    ]
    inner.emit(ForRange(ei, start, stop, ONE, outer_blk, Scope("loop")))

  if not s.parallel:
    emit_range(lp, ZERO, n)
  else:
    w_emit = imul(Const(4, I), work(lp.defs, s.emit_body, (s.index, s.emit_index)))
    lp.parallel(n, lambda inner, _c, st, sp: emit_range(inner, st, sp), loc=s.loc, heavy=True, work=w_emit)


# ---------------------------------------------------------------------------
# IndexScan
# ---------------------------------------------------------------------------


GPU_SINGLE_SCAN_MAX = 2048  # max elements for a 1-launch GPU scan without carries round-trip


def lower_scan(lp: Loops, s: ir.IndexScan, scratch: Scope, target=None) -> None:
  """A loop that carries the accumulator; when the combiner is associative,
  two parallel passes instead: fold every chunk, turn the chunk totals into
  each chunk's starting value, then scan the chunks.

  On GPU targets, small scans (``n <= GPU_SINGLE_SCAN_MAX``) run in a single
  1-chunk device kernel with zero ``carries`` buffer or host sync, and larger
  scans prefix ``carries`` in a 1-chunk device kernel so no intermediate
  host-device round-trip occurs."""
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

  if target is not None and target.is_gpu:
    single_blk: list = []

    def single_scan(inner: Loops, _chunk, start, stop):
      acc = Var("acc", f.a.type)
      if seeded:
        inner.emit(Assign(acc, cast(f.init, acc.type)))
        rest(inner, start, stop, acc, True, True)
      else:
        value_at(inner, start, acc, True)
        rest(inner, iadd(start, 1), stop, acc, True, True)

    single_run: list = []
    lp.sub(single_run).parallel(n, single_scan, kind="reduce", nchunks=ONE, loc=s.loc, heavy=depth(s.body) > 0)
    single_blk.append(If(cmp("gt", n, ZERO), single_run, []))

    multi_blk: list = []
    mlp = lp.sub(multi_blk)
    nchunks = mlp.tmp(RtCall("num_chunks", (n, Const(True, Bool)), I), "nchunks")
    carries = Var("carries", BufType(strong))
    mlp.emit(Alloc(carries, nchunks))
    lp.owner.owned.append(carries)

    def fold_chunk(inner: Loops, chunk, start, stop):
      acc = Var("acc", f.a.type)
      value_at(inner, start, acc, False)
      rest(inner, iadd(start, 1), stop, acc, False, True)
      inner.emit(Store(carries, chunk, cast(acc, strong)))

    mlp.parallel(n, fold_chunk, kind="reduce", nchunks=nchunks, loc=s.loc)

    def prefix_carries(inner: Loops, _chunk, start, stop):
      carry, c, total_v = Var("carry", f.a.type), Var("c", I), Var("total", f.a.type)
      shift: list = [Assign(total_v, cast(Load(carries, c), total_v.type)), Store(carries, c, cast(carry, strong))]
      step(inner.sub(shift), f, carry, total_v, scratch)
      if seeded:
        inner.emit(Assign(carry, cast(f.init, carry.type)))
        inner.emit(ForRange(c, start, stop, ONE, shift, Scope("loop")))
      else:
        inner.emit(Assign(carry, cast(Load(carries, start), carry.type)))
        inner.emit(ForRange(c, iadd(start, 1), stop, ONE, shift, Scope("loop")))

    prefix_run: list = []
    mlp.sub(prefix_run).parallel(nchunks, prefix_carries, kind="reduce", nchunks=ONE, loc=s.loc)
    mlp.emit(If(cmp("gt", nchunks, ZERO), prefix_run, []))

    def scan_chunk(inner: Loops, chunk, start, stop):
      acc = Var("acc", f.a.type)
      before = cast(Load(carries, chunk), acc.type)
      value_at(inner, start, acc, True, join=before, unless=None if seeded else cmp("eq", chunk, 0))
      rest(inner, iadd(start, 1), stop, acc, True, False)

    mlp.parallel(n, scan_chunk, kind="reduce", nchunks=nchunks, loc=s.loc)
    mlp.emit(ir.Release(carries))
    lp.emit(If(cmp("le", n, Const(GPU_SINGLE_SCAN_MAX, I)), single_blk, multi_blk))
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
# IndexWavefront (2D / banded anti-diagonal semiring DP)
# ---------------------------------------------------------------------------


def lower_wavefront(lp: Loops, s: ir.IndexWavefront) -> None:
  """Lower 2D anti-diagonal DP over ``(m, n)`` using 3 rolling anti-diagonal
  ring buffers of length ``m`` (O(m) memory, SIMD-vectorizable inner loop)."""
  m = lp.tmp(s.m, "wf_m")
  n = lp.tmp(s.n, "wf_n")
  k_fields = len(s.results)
  band = lp.tmp(s.band, "wf_band") if s.band is not None else None

  # Default outputs when m == 0 or n == 0
  for rvar, ic in zip(s.results, s.init_corner):
    lp.emit(Assign(rvar, cast(ic, rvar.type)))
  if s.track_max and s.max_val is not None:
    lp.emit(Assign(s.max_val, cast(s.init_corner[0], s.max_val.type)))
    lp.emit(Assign(s.best_i, Const(-1, s.best_i.type)))
    lp.emit(Assign(s.best_j, Const(-1, s.best_j.type)))

  ring_size = lp.tmp(imul(Const(3, I), imax(m, ONE)), "wf_rsz")
  rings = [Var(f"wf_r{f}", BufType(rvar.type.strong)) for f, rvar in enumerate(s.results)]
  for rb in rings:
    lp.emit(Alloc(rb, ring_size))
    lp.owner.owned.append(rb)

  non_empty_blk: list = []
  ne = lp.sub(non_empty_blk)
  num_diags = ne.tmp(isub(iadd(m, n), ONE), "wf_ndiag")
  d = Var("wf_d", I)
  d_body: list = []
  db = ne.sub(d_body)
  cur_slot = db.tmp(ops.imod(d, Const(3, I)), "wf_cs")
  prev1_slot = db.tmp(ops.imod(iadd(d, Const(2, I)), Const(3, I)), "wf_p1")
  prev2_slot = db.tmp(ops.imod(iadd(d, ONE), Const(3, I)), "wf_p2")
  cur_base = db.tmp(imul(cur_slot, m), "wf_cb")
  p1_base = db.tmp(imul(prev1_slot, m), "wf_p1b")
  p2_base = db.tmp(imul(prev2_slot, m), "wf_p2b")

  i_lo = imax(ZERO, isub(d, isub(n, ONE)))
  i_hi = imin(isub(m, ONE), d)
  if band is not None:
    i_lo = imax(i_lo, idiv(iadd(isub(d, band), ONE), Const(2, I)))
    i_hi = imin(i_hi, idiv(iadd(d, band), Const(2, I)))
  i_lo_v = db.tmp(i_lo, "wf_ilo")
  i_hi_v = db.tmp(iadd(i_hi, ONE), "wf_ihi")

  inner_blk: list = []
  ib = db.sub(inner_blk)
  iv = s.i_var
  jv = s.j_var
  ib.emit(Assign(jv, isub(d, iv)))

  has_top = cmp("gt", iv, ZERO)
  has_left = cmp("gt", jv, ZERO)
  if band is not None:
    has_top = ops.land(has_top, cmp("le", isub(jv, isub(iv, ONE)), band))
    has_left = ops.land(has_left, cmp("le", isub(iv, isub(jv, ONE)), band))
  has_top_v = ib.tmp(has_top, "wf_ht")
  has_left_v = ib.tmp(has_left, "wf_hl")
  has_diag_v = ib.tmp(ops.land(cmp("gt", iv, ZERO), cmp("gt", jv, ZERO)), "wf_hd")
  safe_im1 = ib.tmp(imax(isub(iv, ONE), ZERO), "wf_im1")

  for f in range(k_fields):
    rb = rings[f]
    ft = s.results[f].type.strong
    top_load = Load(rb, iadd(p1_base, safe_im1))
    left_load = Load(rb, iadd(p1_base, iv))
    diag_load = Load(rb, iadd(p2_base, safe_im1))
    diag_fallback = ops.select(
      cmp("gt", jv, ZERO),
      cast(s.init_top[f], ft),
      ops.select(cmp("gt", iv, ZERO), cast(s.init_left[f], ft), cast(s.init_corner[f], ft)),
    )
    ib.emit(Assign(s.top_vars[f], ops.select(has_top_v, top_load, cast(s.init_top[f], ft))))
    ib.emit(Assign(s.left_vars[f], ops.select(has_left_v, left_load, cast(s.init_left[f], ft))))
    ib.emit(Assign(s.diag_vars[f], ops.select(has_diag_v, diag_load, diag_fallback)))

  inner_blk.extend(s.body)
  for f in range(k_fields):
    val_f = ib.tmp(cast(s.next_vals[f], s.results[f].type.strong), f"wf_nv{f}")
    ib.emit(Store(rings[f], iadd(cur_base, iv), val_f))
    if f == 0 and s.track_max and s.max_val is not None:
      better = cmp("gt", val_f, s.max_val)
      ib.emit(
        If(
          better,
          [
            Assign(s.max_val, val_f),
            Assign(s.best_i, cast(iv, s.best_i.type)),
            Assign(s.best_j, cast(jv, s.best_j.type)),
          ],
          [],
        )
      )
  inner_blk.append(ir.ReleaseScope(s.scope))

  db.emit(ForRange(iv, i_lo_v, i_hi_v, ONE, inner_blk, s.scope, simd=not s.track_max))
  ne.emit(ForRange(d, ZERO, num_diags, ONE, d_body, Scope("loop")))

  # Read final cell (m - 1, n - 1) on anti-diagonal (m + n - 2) if in band
  last_d = ne.tmp(isub(num_diags, ONE), "wf_ld")
  last_base = ne.tmp(imul(ops.imod(last_d, Const(3, I)), m), "wf_lb")
  last_pos = ne.tmp(iadd(last_base, isub(m, ONE)), "wf_lp")
  read_last = [Assign(rvar, cast(Load(rings[f], last_pos), rvar.type)) for f, rvar in enumerate(s.results)]
  if s.band is not None:
    ne.emit(If(cmp("le", imax(isub(m, n), isub(n, m)), s.band), read_last, []))
  else:
    ne.block.extend(read_last)

  lp.emit(If(ops.land(cmp("gt", m, ZERO), cmp("gt", n, ZERO)), non_empty_blk, []))
  for rb in rings:
    lp.emit(ir.Release(rb))


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
    if isinstance(s, ir.IndexExpand):
      s.count_body[:] = lower_block(s.count_body, defs, s.count_scope, target)
      s.emit_body[:] = lower_block(s.emit_body, defs, s.emit_scope, target)
      lower_expand(Loops(out, defs, owner), s, target)
      continue
    if isinstance(s, ir.IndexScan):
      s.body[:] = lower_block(s.body, defs, s.scope, target)
      s.emit[:] = lower_block(s.emit, defs, s.scope, target)
      scratch = Scope("fold")
      s.fold.combine[:] = lower_block(s.fold.combine, defs, scratch, target)
      lower_scan(Loops(out, defs, owner), s, scratch, target)
      continue
    if isinstance(s, ir.IndexWavefront):
      s.body[:] = lower_block(s.body, defs, s.scope, target)
      lower_wavefront(Loops(out, defs, owner), s)
      continue
    if isinstance(s, If):
      s.then[:] = lower_block(s.then, defs, owner, target)
      s.orelse[:] = lower_block(s.orelse, defs, owner, target)
    elif isinstance(s, (ForRange, While, ParLoop)):
      s.body[:] = lower_block(s.body, defs, s.scope, target)
    out.append(s)
  return out


def _replace_var_in_expr(e: Expr, old_uid: int, new_var: Var) -> Expr:
  if isinstance(e, Var):
    return new_var if e.uid == old_uid else e
  if isinstance(e, ir.Prim):
    return ir.Prim(e.op, tuple(_replace_var_in_expr(a, old_uid, new_var) for a in e.args), e.type)
  if isinstance(e, ir.Cast):
    return ir.Cast(_replace_var_in_expr(e.arg, old_uid, new_var), e.type)
  if isinstance(e, Load):
    return Load(e.buf, _replace_var_in_expr(e.index, old_uid, new_var))
  if isinstance(e, RtCall):
    return RtCall(e.name, tuple(_replace_var_in_expr(a, old_uid, new_var) for a in e.args), e.type, e.pure)
  return e


def _replace_var_in_block(block: list, old_uid: int, new_var: Var) -> None:
  for idx, s in enumerate(block):
    if isinstance(s, Assign):
      block[idx] = Assign(s.var, _replace_var_in_expr(s.value, old_uid, new_var))
    elif isinstance(s, Store):
      block[idx] = Store(
        s.buf,
        _replace_var_in_expr(s.index, old_uid, new_var),
        _replace_var_in_expr(s.value, old_uid, new_var),
        s.atomic,
      )
    elif isinstance(s, ir.Check):
      block[idx] = ir.Check(
        _replace_var_in_expr(s.cond, old_uid, new_var),
        s.err,
        tuple(_replace_var_in_expr(a, old_uid, new_var) for a in s.args),
      )
    elif isinstance(s, Alloc):
      s.count = _replace_var_in_expr(s.count, old_uid, new_var)
    elif isinstance(s, If):
      s.cond = _replace_var_in_expr(s.cond, old_uid, new_var)
      _replace_var_in_block(s.then, old_uid, new_var)
      _replace_var_in_block(s.orelse, old_uid, new_var)
    elif isinstance(s, ForRange):
      s.start = _replace_var_in_expr(s.start, old_uid, new_var)
      s.stop = _replace_var_in_expr(s.stop, old_uid, new_var)
      s.step = _replace_var_in_expr(s.step, old_uid, new_var)
      _replace_var_in_block(s.body, old_uid, new_var)
    elif isinstance(s, While):
      _replace_var_in_block(s.body, old_uid, new_var)
    elif isinstance(s, ParLoop):
      s.total = _replace_var_in_expr(s.total, old_uid, new_var)
      s.nchunks = _replace_var_in_expr(s.nchunks, old_uid, new_var)
      if s.work is not None:
        s.work = _replace_var_in_expr(s.work, old_uid, new_var)
      _replace_var_in_block(s.body, old_uid, new_var)


def _count_var_in_stmts(stmts: list, uid: int) -> int:
  n = 0
  for s in ir.walk_stmts(stmts):
    for e in ir.stmt_exprs(s):
      for x in ir.walk_expr(e):
        if isinstance(x, Var) and x.uid == uid:
          n += 1
  return n


def sink_gpu_scalar_loads(fn: ir.Function, target=None) -> None:
  """Move host scalar loads ``v = buf[idx]`` into consuming GPU ``ParLoop``s when
  every use of ``v`` is inside those loops and ``buf`` is not mutated in between.

  Prevents unnecessary host-side buffer downloads for scalars like
  ``g_tot = g_left[nbins - 1]`` or ``x - x[-1]``."""
  from .analysis import BufferInfo

  binfo = BufferInfo.analyze(fn)
  max_bufs = 28 if (target is not None and getattr(target, "name", None) == "metal") else None
  def_count: dict[int, int] = {}
  use_count: dict[int, int] = {}
  for s in ir.walk_stmts(fn.body):
    for v in ir.stmt_defs(s):
      def_count[v.uid] = def_count.get(v.uid, 0) + 1
    for e in ir.stmt_exprs(s):
      for x in ir.walk_expr(e):
        if isinstance(x, Var):
          use_count[x.uid] = use_count.get(x.uid, 0) + 1
  for r in fn.results:
    if isinstance(r, Expr):
      for x in ir.walk_expr(r):
        if isinstance(x, Var):
          use_count[x.uid] = use_count.get(x.uid, 0) + 1
    else:
      for name in getattr(type(r), "__dataclass_fields__", ()):
        val = getattr(r, name)
        if isinstance(val, Expr):
          for x in ir.walk_expr(val):
            if isinstance(x, Var):
              use_count[x.uid] = use_count.get(x.uid, 0) + 1
        elif isinstance(val, (tuple, list)):
          for item in val:
            if isinstance(item, Expr):
              for x in ir.walk_expr(item):
                if isinstance(x, Var):
                  use_count[x.uid] = use_count.get(x.uid, 0) + 1

  def _is_pure_invariant_index(e: Expr) -> bool:
    for x in ir.walk_expr(e):
      if isinstance(x, Load):
        return False
      if isinstance(x, RtCall) and not x.pure:
        return False
      if isinstance(x, Var) and def_count.get(x.uid, 0) > 1:
        return False
    return True

  def _touches_buf_or_impure(st: ir.Stmt, buf_uid: int) -> bool:
    written: set[int] = set()
    for sub in ir.walk_stmts([st]):
      if isinstance(sub, Store):
        written.add(sub.buf.uid)
      elif isinstance(sub, ir.Release):
        written.add(sub.var.uid)
      elif isinstance(sub, Assign) and any(
        isinstance(x, RtCall) and not x.pure and x.name not in ("num_chunks", "num_threads", "rng_key")
        for x in ir.walk_expr(sub.value)
      ):
        return True
    return bool(written) and binfo.may_alias({buf_uid}, written)

  def _loop_buf_uids(pl: ParLoop) -> set[int]:
    uids: set[int] = set()
    for sub in ir.walk_stmts(pl.body):
      if isinstance(sub, Store):
        uids.add(sub.buf.uid)
      for e in ir.stmt_exprs(sub):
        for x in ir.walk_expr(e):
          if isinstance(x, Var) and isinstance(x.type, BufType):
            uids.add(x.uid)
    return uids

  def walk_block(block: list) -> None:
    for s in block:
      for b in ir.sub_blocks(s):
        walk_block(b)
    drop: set[int] = set()
    for i, s in enumerate(block):
      if not (isinstance(s, Assign) and isinstance(s.value, Load) and not isinstance(s.var.type, BufType)):
        continue
      v = s.var
      if def_count.get(v.uid, 0) != 1 or use_count.get(v.uid, 0) <= 0:
        continue
      buf = s.value.buf
      idx_e = s.value.index
      if not _is_pure_invariant_index(idx_e):
        continue
      consumers: list[ParLoop] = []
      covered = 0
      for j in range(i + 1, len(block)):
        nxt = block[j]
        if _touches_buf_or_impure(nxt, buf.uid):
          break
        if (
          isinstance(nxt, ParLoop) and nxt.shared_tile is None and nxt.monoidal_tile is None and nxt.warp_reduce is None
        ):
          hdr_uses = sum(
            1
            for h in (nxt.total, nxt.nchunks, *(() if nxt.work is None else (nxt.work,)))
            for x in ir.walk_expr(h)
            if isinstance(x, Var) and x.uid == v.uid
          )
          if hdr_uses > 0:
            break
          body_uses = _count_var_in_stmts(nxt.body, v.uid)
          if body_uses > 0:
            consumers.append(nxt)
            covered += body_uses
      if consumers and covered == use_count[v.uid]:
        if max_bufs is not None and any(len(_loop_buf_uids(pl) | {buf.uid}) > max_bufs for pl in consumers):
          continue
        drop.add(i)
        for pl in consumers:
          local_v = Var(v.name, v.type)
          _replace_var_in_block(pl.body, v.uid, local_v)
          pl.body.insert(0, Assign(local_v, Load(buf, idx_e)))
    if drop:
      block[:] = [s for i, s in enumerate(block) if i not in drop]

  walk_block(fn.body)


def lower(fn: ir.Function, target=None) -> None:
  """Replace every operator statement of ``fn`` by loops."""
  fn.body[:] = lower_block(fn.body, pure_defs(fn.body), fn.scope, target)
  if target is not None and target.is_gpu:
    sink_gpu_scalar_loads(fn, target)


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
