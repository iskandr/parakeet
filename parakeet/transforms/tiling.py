"""Tiling and reduction interchange for map-of-reduction operators.

*reduction interchange* (``PermuteReductions``)::

    map i.., j over (.., m):
      reduce k.. over (K..):
        fold(op, x[.., k, j])        # unit stride in j, strided in k

When the inner reduction walks memory with a large stride while the map's last
index ``j`` has unit stride (``np.sum(X, axis=0)``, ``np.mean(X, axis=0)``),
interchanging the loops over L1-sized tiles of ``j`` turns every load and
accumulator update into a contiguous SIMD loop.

*tiling and jamming* (``TiledMap``)::

    map i.., j over (.., m):          # any rank >= 2
      straight-line code
      reduce k.. over (K..):          # one reduction, of any rank,
        fold(...), fold(...)          #   with any number of folds
      straight-line code              # the "epilogue": sqrt, tanh, a store, ...

Evaluated row by row, every row of the result sweeps over all of the ``j``
operand, which stops fitting in cache long before the arrays are large.
Cutting ``j`` into cache-sized tiles and jamming a 2x4 block of reductions
into one multi-fold ``IndexReduce`` keeps the ``j`` tile in cache and reuses
every loaded value.
"""

from __future__ import annotations

import math

from ..ndtypes import Bool, BufType
from ..syntax import ir, ops
from ..syntax.clone import Cloner, defined
from ..syntax.ir import Alloc, Assign, Const, Expr, ForRange, If, IndexMap, IndexReduce, Load, RtCall, Scope, Store, Var
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, idiv, imax, imin, imul, iprod, isub
from .lowering import Loops, outer_expr, pure_defs

UNROLL_I = 2
UNROLL_J = 4
MIN_STATIC_SIZE = 256  # maps known to be smaller than this are not worth tiling
INTERCHANGE_TILE = 256  # L1-sized strip of contiguous output columns (256 * 8B = 2 KiB)
COMMUTATIVE_OPS = frozenset({"add", "mul", "min", "max", "and", "or", "bitand", "bitor", "bitxor"})

STRAIGHT = (ir.Assign, ir.Store, ir.Check, ir.Comment)


class Shape:
  """What :func:`match` found: the map is ``before; reduce; after``."""

  __slots__ = ("after", "before", "extent", "reduce")


def match(s: ir.Stmt, defs: dict[int, Expr], min_rank: int = 2) -> Shape | None:
  """Is ``s`` a map whose element is one reduction and straight-line code,
  over an index space that does not depend on the element?"""
  if not isinstance(s, IndexMap) or s.ordered or len(s.shape) < min_rank or s.scope.owned:
    return None
  if (
    min_rank >= 2
    and all(isinstance(d, Const) for d in s.shape)
    and math.prod(d.value for d in s.shape) < MIN_STATIC_SIZE
  ):
    return None
  body = [t for t in s.body if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
  where = [k for k, t in enumerate(body) if isinstance(t, IndexReduce)]
  if len(where) != 1 or not all(isinstance(t, STRAIGHT) for k, t in enumerate(body) if k != where[0]):
    return None
  red = body[where[0]]
  if red.where is not None:
    return None
  if any(isinstance(t.var.type, ir.BufType) for t in body if isinstance(t, Assign)):
    return None
  if any(isinstance(t, ir.Store) for t in body[: where[0]]):
    return None
  inner = set(defined(s.body)) | {v.uid for v in s.index}
  extent = [outer_expr(defs, d, inner) for d in red.shape]
  if any(d is None for d in extent):
    return None  # ragged: elements fold different numbers of terms
  m = Shape()
  m.before, m.reduce, m.after, m.extent = body[: where[0]], red, body[where[0] + 1 :], tuple(extent)
  return m


# ---------------------------------------------------------------------------
# reduction interchange (PermuteReductions)
# ---------------------------------------------------------------------------


def linear_stride(e: Expr, v: Var, defs: dict[int, Expr], budget: int = 20) -> Expr | None:
  """Return the affine coefficient of ``v`` in integer expression ``e``, or ``None``."""
  if budget <= 0:
    return None
  if isinstance(e, Const):
    return ZERO
  if isinstance(e, Var):
    if e is v:
      return ONE
    d = defs.get(e.uid)
    return ZERO if d is None else linear_stride(d, v, defs, budget - 1)
  if isinstance(e, ir.Cast):
    return linear_stride(e.arg, v, defs, budget - 1)
  if isinstance(e, ir.Prim):
    if e.op in ("add", "sub") and len(e.args) == 2:
      a = linear_stride(e.args[0], v, defs, budget - 1)
      b = linear_stride(e.args[1], v, defs, budget - 1)
      if a is None or b is None:
        return None
      return iadd(a, b) if e.op == "add" else isub(a, b)
    if e.op == "mul" and len(e.args) == 2:
      a = linear_stride(e.args[0], v, defs, budget - 1)
      b = linear_stride(e.args[1], v, defs, budget - 1)
      if a is None or b is None:
        return None
      if isinstance(a, Const) and a.value == 0:
        return imul(e.args[0], b)
      if isinstance(b, Const) and b.value == 0:
        return imul(a, e.args[1])
      return None
    if e.op == "neg" and len(e.args) == 1:
      a = linear_stride(e.args[0], v, defs, budget - 1)
      return None if a is None else isub(ZERO, a)
  return None


def is_const(e: Expr | None, val: int) -> bool:
  """True if ``e`` is a constant equal to ``val``."""
  return isinstance(e, Const) and e.value == val


def should_interchange(s: IndexMap, m: Shape, defs: dict[int, Expr]) -> bool:
  """True when the reduction is associative+commutative and walks memory
  strided along its own inner index while ``s.index[-1]`` has unit stride."""
  red = m.reduce
  if red.scope.owned:
    return False
  if isinstance(s.shape[-1], Const) and s.shape[-1].value < 4:
    return False
  if not all(f.op in COMMUTATIVE_OPS and not f.combine for f in red.folds):
    return False
  if len({f.init is not None for f in red.folds}) != 1:
    return False
  if not all(isinstance(t, (Assign, ir.Comment)) for t in m.before + red.body):
    return False
  inner = set(defined(s.body)) | {v.uid for v in s.index}
  if any(f.init is not None and outer_expr(defs, f.init, inner) is None for f in red.folds):
    return False
  local_defs = dict(defs)
  for t in m.before + red.body:
    if isinstance(t, Assign) and not isinstance(t.var.type, BufType) and t.var.type.kind in "iu":
      local_defs[t.var.uid] = t.value
  map_inner, reduce_inner = s.index[-1], red.index[-1]
  unit_map_strided_reduce = False
  for t in ir.walk_stmts(red.body):
    for e in ir.stmt_exprs(t):
      for x in ir.walk_expr(e):
        if isinstance(x, Load):
          reduce_stride = linear_stride(x.index, reduce_inner, local_defs)
          map_stride = linear_stride(x.index, map_inner, local_defs)
          if is_const(reduce_stride, 1):
            return False  # already contiguous along the reduction axis
          if is_const(map_stride, 1) and reduce_stride is not None and not is_const(reduce_stride, 0):
            unit_map_strided_reduce = True
  return unit_map_strided_reduce


def interchanged(s: IndexMap, m: Shape, defs: dict[int, Expr], owner: Scope) -> list:
  """Swap ``m.reduce`` outside an L1-cached strip of ``s.index[-1]`` so the
  innermost loop walks both the input and the tile accumulator with stride 1."""
  out: list = []
  lp = Loops(out, defs, owner)
  red = m.reduce
  folds = red.folds
  seeded = folds[0].init is not None
  lead = tuple(lp.tmp(d, "n") for d in s.shape[:-1])
  ncols = lp.tmp(s.shape[-1], "n_j")
  n_rows = lp.tmp(iprod(lead), "rows") if lead else ONE
  tile_width = Const(INTERCHANGE_TILE, I)
  n_tiles = lp.tmp(idiv(iadd(ncols, INTERCHANGE_TILE - 1), tile_width), "ntiles")
  total = lp.tmp(imul(n_rows, n_tiles), "total")
  extent = tuple(lp.tmp(d, "k") for d in m.extent)
  total_k = lp.tmp(iprod(extent), "k_total")

  def emit_tile(inner: Loops, row_coords: tuple[Expr, ...], j0: Expr, tile_len: Expr, base: Expr, acc_bufs: list[Var]):
    """Compute one ``tile_len``-wide strip of columns for ``row_coords``."""

    def emit_tile_step(step_lp: Loops, t: Var, k_coords: tuple[Expr, ...], init_only: bool) -> None:
      """Evaluate the reduction body at ``(row_coords, j0 + t, k_coords)`` and
      either initialize or fold into ``acc_bufs``."""
      j = iadd(j0, t)
      subst = {v.uid: e for v, e in zip(s.index, (*row_coords, j))}
      subst.update({v.uid: e for v, e in zip(red.index, k_coords)})
      cloner = Cloner(subst)
      for v in defined(m.before + red.body).values():
        cloner.var(v)
      step_lp.block.extend(cloner.stmt(st) for st in m.before)
      step_lp.block.extend(cloner.stmt(st) for st in red.body)
      pos = iadd(base, t)
      for f, buf in zip(folds, acc_bufs):
        val = cast(cloner.expr(f.value), f.a.type)
        if init_only:
          step_lp.emit(Store(buf, pos, cast(val, f.a.type.strong)))
        else:
          cur = cast(Load(buf, pos), f.a.type)
          step_lp.emit(Store(buf, pos, cast(ops.prim(f.op, (cur, val), f.a.type), f.a.type.strong)))

    # 1. Initialize tile accumulators
    init_body: list = []
    t_init = Var("t", I)
    if seeded:
      for f, buf in zip(folds, acc_bufs):
        init_body.append(Store(buf, iadd(base, t_init), cast(f.init, f.a.type.strong)))
    else:
      emit_tile_step(inner.sub(init_body), t_init, (ZERO,) * len(extent), init_only=True)
    inner.emit(ForRange(t_init, ZERO, tile_len, ONE, init_body, Scope("loop"), simd=True))

    # 2. Sweep the reduction space, vectorizing across the contiguous tile of j
    def step_k(k_lp: Loops, k_coords: tuple[Expr, ...]) -> None:
      t_step = Var("t", I)
      step_body: list = []
      emit_tile_step(k_lp.sub(step_body), t_step, k_coords, init_only=False)
      k_lp.emit(ForRange(t_step, ZERO, tile_len, ONE, step_body, Scope("loop"), simd=True))

    if seeded:
      k_vars = tuple(Var("k", I) for _ in extent)
      body_k: list = []
      step_k(inner.sub(body_k), k_vars)
      inner.nest(extent, k_vars, body_k, Scope("loop"))
    else:
      k_flat = Var("kf", I)
      body_k = []
      k_lp = inner.sub(body_k)
      k_coords = (k_flat,) if len(extent) == 1 else k_lp.unravel(k_flat, extent)
      step_k(k_lp, k_coords)
      inner.emit(ForRange(k_flat, ONE, total_k, ONE, body_k, Scope("loop")))

    # 3. Epilogue (store / mean division / sqrt / ...)
    t_end = Var("t", I)
    epilogue: list = []
    subst = {v.uid: e for v, e in zip(s.index, (*row_coords, iadd(j0, t_end)))}
    epilogue_cloner = Cloner(subst)
    before_defs = defined(m.before)
    after_uses = {
      x.uid for st in ir.walk_stmts(m.after) for e in ir.stmt_exprs(st) for x in ir.walk_expr(e) if isinstance(x, Var)
    }
    if before_defs.keys() & after_uses:
      for v in before_defs.values():
        epilogue_cloner.var(v)
      epilogue.extend(epilogue_cloner.stmt(st) for st in m.before)
    for v in defined(m.after).values():
      epilogue_cloner.var(v)
    for f, buf in zip(folds, acc_bufs):
      epilogue.append(Assign(epilogue_cloner.var(f.result), cast(Load(buf, iadd(base, t_end)), f.result.type)))
    epilogue.extend(epilogue_cloner.stmt(st) for st in m.after)
    inner.emit(
      ForRange(
        t_end, ZERO, tile_len, ONE, epilogue, Scope("loop"), simd=not any(isinstance(st, ir.Check) for st in m.after)
      )
    )

  def run_range(inner: Loops, start: Expr, stop: Expr, base: Expr, acc_bufs: list[Var]) -> None:
    def one_tile(body_lp: Loops, flat: Var) -> None:
      if lead:
        row = body_lp.tmp(idiv(flat, n_tiles), "row")
        tile_idx = body_lp.tmp(isub(flat, imul(row, n_tiles)), "ti")
        row_coords = (row,) if len(lead) == 1 else body_lp.unravel(row, lead)
      else:
        tile_idx = flat
        row_coords = ()
      j0 = body_lp.tmp(imul(tile_idx, tile_width), "j0")
      tile_len = body_lp.tmp(imin(tile_width, isub(ncols, j0)), "tlen")
      emit_tile(body_lp, row_coords, j0, tile_len, base, acc_bufs)

    loop(inner, start, stop, one_tile, name="tile")

  work: list = []
  work_lp = lp.sub(work)
  if not seeded:
    for f in folds:
      if f.empty is not None:
        work_lp.emit(ir.Check(cmp("gt", total_k, 0), f.empty))
        break
  if s.parallel:
    nchunks = work_lp.tmp(RtCall("num_chunks", (total, Const(True, Bool)), I), "nchunks")
    acc_bufs = []
    for f in folds:
      buf = Var("tile_acc", BufType(f.a.type.strong))
      work_lp.emit(Alloc(buf, imul(nchunks, tile_width)))
      owner.owned.append(buf)
      acc_bufs.append(buf)
    work_lp.parallel(
      total,
      lambda inner, chunk, start, stop: run_range(
        inner, start, stop, inner.tmp(imul(chunk, tile_width), "base"), acc_bufs
      ),
      kind="reduce",
      nchunks=nchunks,
      loc=s.loc,
      heavy=True,
    )
    for buf in acc_bufs:
      work_lp.emit(ir.Release(buf))
  else:
    acc_bufs = []
    for f in folds:
      buf = Var("tile_acc", BufType(f.a.type.strong))
      work_lp.emit(Alloc(buf, tile_width))
      owner.owned.append(buf)
      acc_bufs.append(buf)
    run_range(work_lp, ZERO, total, ZERO, acc_bufs)
    for buf in acc_bufs:
      work_lp.emit(ir.Release(buf))
  lp.emit(If(cmp("gt", total, 0), work, []))
  return out


def interchange_block(block: list, defs: dict[int, Expr], owner: Scope) -> list:
  """Rewrite eligible strided map-of-reduction statements throughout ``block``."""
  out: list = []
  for s in block:
    if isinstance(s, If):
      s.then[:] = interchange_block(s.then, defs, owner)
      s.orelse[:] = interchange_block(s.orelse, defs, owner)
    elif isinstance(s, IndexReduce):
      s.body[:] = interchange_block(s.body, defs, s.scope)
    elif isinstance(s, (IndexMap, ForRange, ir.While, ir.ParLoop)):
      s.body[:] = interchange_block(s.body, defs, s.scope)
    m = match(s, defs, min_rank=1)
    if m is not None and should_interchange(s, m, defs):  # type: ignore[arg-type]
      out.extend(interchanged(s, m, defs, owner))  # type: ignore[arg-type]
    else:
      out.append(s)
  return out


def interchange(fn: ir.Function, options) -> None:
  """PermuteReductions: interchange strided map-of-reduction loops into
  L1-tiled contiguous SIMD loops."""
  if options.tiling:
    fn.body[:] = interchange_block(fn.body, pure_defs(fn.body), fn.scope)


# ---------------------------------------------------------------------------
# tiling and jamming (TiledMap)
# ---------------------------------------------------------------------------


def element(lp: Loops, s: IndexMap, m: Shape, at: tuple[Expr, ...]) -> None:
  """One element of the map, as it was written."""
  c = Cloner({v.uid: e for v, e in zip(s.index, at)})
  lp.block.extend(c.block(m.before + [m.reduce] + m.after))


def jam(lp: Loops, s: IndexMap, m: Shape, ats: list[tuple[Expr, ...]]) -> None:
  """The elements at ``ats`` with their reductions fused into one."""
  red = m.reduce
  index = tuple(Var("k", I) for _ in red.index)
  scope = Scope("loop")
  body: list = []
  folds: list = []
  tails: list = []
  for at in ats:
    c = Cloner({v.uid: e for v, e in zip(s.index, at)})
    c.m.update({v.uid: k for v, k in zip(red.index, index)})
    for v in defined(m.before + [red] + m.after).values():
      c.var(v)
    lp.block.extend(c.stmt(t) for t in m.before)
    body.extend(c.stmt(t) for t in red.body)
    folds.extend(c.fold(f) for f in red.folds)
    scope.owned.extend(c.scope(red.scope).owned)
    tails.append([c.stmt(t) for t in m.after])
  lp.emit(IndexReduce(m.extent, index, body, folds, scope, red.home, False, red.loc))
  for tail in tails:
    lp.block.extend(tail)


def loop(lp: Loops, start: Expr, stop: Expr, fill, step: Expr = ONE, name: str = "i") -> None:
  v = Var(name, I)
  blk: list = []
  scope = Scope("loop")
  fill(lp.sub(blk, scope), v)
  lp.emit(ForRange(v, start, stop, step, blk, scope))


def tile_size(lp: Loops, length: Expr, itemsize: int, tile_bytes: int) -> Expr:
  """How many ``j`` rows of ``length`` elements make one tile."""
  budget = max(tile_bytes // max(itemsize, 1), 1)
  rows = idiv(Const(budget, I), imax(length, 1))
  whole = imul(idiv(rows, UNROLL_J), UNROLL_J)
  size = lp.tmp(imax(whole, UNROLL_J), "tile")
  if isinstance(size, Const):  # the printed IR says so either way
    v = Var("tile", I)
    lp.emit(Assign(v, size))
    size = v
  return size


def tiled(s: IndexMap, m: Shape, defs: dict[int, Expr], owner: Scope, tile_bytes: int) -> list:
  """The loops of ``s``, in tiles of its last dimension.  The dimensions
  before it are one long row index."""
  out: list = []
  lp = Loops(out, defs, owner)
  lead = tuple(lp.tmp(d, "n") for d in s.shape[:-1])
  n_j = lp.tmp(s.shape[-1], "n")
  n_i = lp.tmp(iprod(lead), "rows")
  length = lp.tmp(iprod(m.extent), "len")
  itemsize = m.reduce.folds[0].a.type.bits // 8
  tile = tile_size(lp, length, itemsize, tile_bytes)

  def at(inner: Loops, i: Expr, j: Expr) -> tuple[Expr, ...]:
    return ((inner.tmp(i, "i"),) if len(lead) == 1 else inner.unravel(inner.tmp(i, "i"), lead)) + (inner.tmp(j, "j"),)

  def single(inner: Loops, i: Expr, j: Expr) -> None:
    element(inner, s, m, at(inner, i, j))

  def span(inner: Loops, row: Expr, lo: Expr, hi: Expr) -> None:
    """Columns ``[lo, hi)`` of one row, an element at a time."""
    loop(inner, lo, hi, lambda body, j: single(body, row, j), name="j")

  def rows(inner: Loops, r0: Expr, r1: Expr) -> None:
    """Complete rows ``[r0, r1)``, a tile of columns at a time."""
    i_jam = inner.tmp(iadd(r0, imul(idiv(isub(r1, r0), UNROLL_I), UNROLL_I)), "i_jam")

    def tile_body(t: Loops, j0: Var) -> None:
      j1 = t.tmp(imin(n_j, iadd(j0, tile)), "j1")
      j_jam = t.tmp(iadd(j0, imul(idiv(isub(j1, j0), UNROLL_J), UNROLL_J)), "j_jam")

      def block_rows(b: Loops, i: Var) -> None:
        def block(e: Loops, j: Var) -> None:
          jam(e, s, m, [at(e, iadd(i, p), iadd(j, q)) for p in range(UNROLL_I) for q in range(UNROLL_J)])

        def leftover(e: Loops, j: Var) -> None:
          for p in range(UNROLL_I):
            single(e, iadd(i, p), j)

        loop(b, j0, j_jam, block, step=Const(UNROLL_J, I), name="j")
        loop(b, j_jam, j1, leftover, name="j")

      loop(t, r0, i_jam, block_rows, step=Const(UNROLL_I, I), name="i")
      loop(t, i_jam, r1, lambda b, i: span(b, i, j0, j1), name="i")

    loop(inner, ZERO, n_j, tile_body, step=tile, name="j0")

  def chunk(inner: Loops, _chunk, start: Expr, stop: Expr) -> None:
    """Flat positions ``[start, stop)``: the end of a row, whole rows, the
    beginning of a row."""
    r0 = inner.tmp(idiv(start, n_j), "r0")
    c0 = inner.tmp(isub(start, imul(r0, n_j)), "c0")
    lead_end = inner.tmp(ops.select(cmp("gt", c0, 0), imin(stop, imul(iadd(r0, 1), n_j)), start), "lead_end")
    span(inner, r0, c0, iadd(c0, isub(lead_end, start)))
    first_row = inner.tmp(ops.select(cmp("gt", c0, 0), iadd(r0, 1), r0), "first_row")
    last_row = inner.tmp(idiv(stop, n_j), "last_row")  # may be one past the end
    # a chunk inside a single row has no complete rows (and first_row > last_row)
    rows(inner, first_row, inner.tmp(imax(first_row, last_row), "rows_end"))
    tail_start = inner.tmp(imax(lead_end, imul(last_row, n_j)), "tail_start")
    span(inner, last_row, isub(tail_start, imul(last_row, n_j)), isub(stop, imul(last_row, n_j)))

  total = lp.tmp(imul(n_i, n_j), "total")
  work: list = []
  # where it cannot run in parallel it is still a loop of one chunk: the
  # backend compiles that as a function of its own
  lp.sub(work).parallel(total, chunk, nchunks=None if s.parallel else ONE, loc=s.loc, work=length, heavy=True)
  # an empty row would make the row arithmetic divide by zero; there is nothing to do then anyway
  lp.emit(If(cmp("gt", total, 0), work, []))
  return out


def tile_block(block: list, defs: dict[int, Expr], owner: Scope, tile_bytes: int) -> list:
  out: list = []
  for s in block:
    if isinstance(s, If):
      s.then[:] = tile_block(s.then, defs, owner, tile_bytes)
      s.orelse[:] = tile_block(s.orelse, defs, owner, tile_bytes)
    elif isinstance(s, IndexReduce):
      s.body[:] = tile_block(s.body, defs, s.scope, tile_bytes)
    elif isinstance(s, (IndexMap, ForRange, ir.While, ir.ParLoop)):
      s.body[:] = tile_block(s.body, defs, s.scope, tile_bytes)
    m = match(s, defs)
    if m is None:
      out.append(s)
    else:
      out.extend(tiled(s, m, defs, owner, tile_bytes))  # type: ignore[arg-type]
  return out


def tiling(fn: ir.Function, options) -> None:
  """TiledMap: tile and jam every matmul-shaped map of ``fn``."""
  if options.tiling:
    fn.body[:] = tile_block(fn.body, pure_defs(fn.body), fn.scope, options.tile_bytes)


__all__ = [
  "INTERCHANGE_TILE",
  "UNROLL_I",
  "UNROLL_J",
  "interchange",
  "interchange_block",
  "interchanged",
  "jam",
  "linear_stride",
  "match",
  "should_interchange",
  "tile_block",
  "tile_size",
  "tiled",
  "tiling",
]
