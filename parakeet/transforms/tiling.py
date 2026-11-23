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
from ..syntax.ir import (
  Alloc,
  Assign,
  Const,
  Expr,
  ForRange,
  If,
  IndexMap,
  IndexReduce,
  IndexTile,
  Load,
  RtCall,
  Scope,
  Store,
  Var,
)
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, idiv, imax, imin, imul, iprod, isub
from .lowering import Loops, outer_expr, pure_defs

UNROLL_I = 2
UNROLL_J = 4
ROW_TILE = 32  # cache-tile row strip height for multi-core rectangular tiling
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
# tiling and jamming (IndexTile)
# ---------------------------------------------------------------------------


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


def extract_shared_tile_plan(
  s: IndexMap, m: Shape, shape: tuple[Expr, ...], extent: tuple[Expr, ...]
) -> ir.SharedTilePlan | None:
  """Analyze a map-of-reduction ``(s, m)`` and extract a general
  :class:`~parakeet.syntax.ir.SharedTilePlan` for GPU ``__shared__`` /
  ``threadgroup`` block tiling.

  Automatically discovers every maximal pure sub-expression in the inner
  reduction that loads from memory along ``(batch..., row, k)``,
  ``(batch..., col, k)``, or ``(k,)`` — including fused prologues like
  ``sin(a) @ cos(b)``, ``allpairs``, multi-input contractions, and batched 3D+
  matmul — and lifts each into a :class:`~parakeet.syntax.ir.TileStage`.
  """
  red = m.reduce
  if not s.parallel or len(shape) < 2 or len(extent) != 1:
    return None
  if red.scope.owned or red.where is not None:
    return None
  if any(isinstance(t, ir.Check) for t in red.body):
    return None

  i_var, j_var = s.index[-2], s.index[-1]
  k_var = red.index[0]
  i_uid, j_uid, k_uid = i_var.uid, j_var.uid, k_var.uid
  k_dim = extent[0]

  all_assigns = [t for t in m.before + red.body if isinstance(t, Assign)]
  def_map: dict[int, Assign] = {t.var.uid: t for t in all_assigns}
  order_map: dict[int, int] = {t.var.uid: idx for idx, t in enumerate(all_assigns)}
  mask_cache: dict[int, tuple[bool, bool, bool, bool, bool]] = {}

  def expr_mask(e: Expr) -> tuple[bool, bool, bool, bool, bool]:
    """Return ``(uses_i, uses_j, uses_k, has_load, is_pure)``."""
    if isinstance(e, Var):
      uid = e.uid
      if uid == i_uid:
        return (True, False, False, False, True)
      if uid == j_uid:
        return (False, True, False, False, True)
      if uid == k_uid:
        return (False, False, True, False, True)
      if uid in mask_cache:
        return mask_cache[uid]
      d = def_map.get(uid)
      if d is None:
        return (False, False, False, False, True)
      res = expr_mask(d.value)
      mask_cache[uid] = res
      return res
    if isinstance(e, Const):
      return (False, False, False, False, True)
    if isinstance(e, Load):
      ui, uj, uk, _, p = expr_mask(e.index)
      return (ui, uj, uk, True, p)
    if isinstance(e, ir.Prim):
      ui = uj = uk = hl = False
      p = True
      for a in e.args:
        ai, aj, ak, al, ap = expr_mask(a)
        ui |= ai
        uj |= aj
        uk |= ak
        hl |= al
        p &= ap
      return (ui, uj, uk, hl, p)
    if isinstance(e, ir.Cast):
      return expr_mask(e.arg)
    if isinstance(e, ir.RtCall):
      ui = uj = uk = hl = False
      p = e.pure
      for a in e.args:
        ai, aj, ak, al, ap = expr_mask(a)
        ui |= ai
        uj |= aj
        uk |= ak
        hl |= al
        p &= ap
      return (ui, uj, uk, hl, p)
    return (False, False, False, False, False)

  def stage_side(e: Expr) -> str | None:
    if isinstance(e.type, BufType):
      return None
    ui, uj, uk, hl, p = expr_mask(e)
    if not (p and hl and uk):
      return None
    if ui and not uj:
      return "row"
    if uj and not ui:
      return "col"
    if not ui and not uj:
      return "k"
    return None

  def expr_key(e: Expr):
    if isinstance(e, Var):
      return ("v", e.uid)
    if isinstance(e, Const):
      return ("c", e.value, e.type)
    if isinstance(e, Load):
      return ("l", e.buf.uid, expr_key(e.index))
    if isinstance(e, ir.Prim):
      return ("p", e.op, e.type, tuple(expr_key(a) for a in e.args))
    if isinstance(e, ir.Cast):
      return ("t", e.type, expr_key(e.arg))
    if isinstance(e, ir.RtCall):
      return ("r", e.name, e.type, tuple(expr_key(a) for a in e.args))
    return ("?", id(e))

  def collect_deps(e: Expr, seen: set[int]) -> None:
    for x in ir.walk_expr(e):
      if isinstance(x, Var) and x.uid in def_map and x.uid not in seen:
        seen.add(x.uid)
        collect_deps(def_map[x.uid].value, seen)

  stages: list[ir.TileStage] = []
  stage_sides: list[str] = []
  stage_vars: list[Var] = []
  staged_by_key: dict[object, Var] = {}

  def make_stage(e: Expr, side: str) -> Var:
    ek = (side, expr_key(e))
    existing = staged_by_key.get(ek)
    if existing is not None:
      return existing
    dep_uids: set[int] = set()
    collect_deps(e, dep_uids)
    dep_stmts = [def_map[u] for u in sorted(dep_uids, key=lambda u: order_map[u])]
    st_k = Var("sk", I)
    if side == "row":
      st_i = Var("si", I)
      st_index: tuple[Var, ...] = (st_i, st_k)
      st_shape: tuple[Expr, ...] = (shape[-2], k_dim)
      subst: dict[int, Expr] = {i_uid: st_i, k_uid: st_k}
    elif side == "col":
      st_j = Var("sj", I)
      st_index = (st_j, st_k)
      st_shape = (shape[-1], k_dim)
      subst = {j_uid: st_j, k_uid: st_k}
    else:
      st_index = (st_k,)
      st_shape = (k_dim,)
      subst = {k_uid: st_k}
    cl = Cloner(subst)
    for st in dep_stmts:
      cl.var(st.var)
    st_body = [cl.stmt(st) for st in dep_stmts]
    st_val = cl.expr(e)
    idx = len(stages)
    sbuf = Var(f"sh_{idx}", BufType(e.type.strong))
    svar = Var(f"sv_{idx}", e.type)
    stages.append(ir.TileStage(sbuf, st_shape, st_index, st_body, st_val, Scope("stage")))
    stage_sides.append(side)
    stage_vars.append(svar)
    staged_by_key[ek] = svar
    return svar

  def rewrite_expr(e: Expr) -> Expr:
    side = stage_side(e)
    if side is not None:
      return make_stage(e, side)
    if isinstance(e, ir.Prim):
      return ir.Prim(e.op, tuple(rewrite_expr(a) for a in e.args), e.type)
    if isinstance(e, ir.Cast):
      return ir.Cast(rewrite_expr(e.arg), e.type)
    if isinstance(e, Load):
      return Load(e.buf, rewrite_expr(e.index))
    if isinstance(e, ir.RtCall):
      return ir.RtCall(e.name, tuple(rewrite_expr(a) for a in e.args), e.type, e.pure)
    return e

  # Rewrite only mixed statements in red.body and fold values; single-side assigns
  # are subsumed by maximal staged expressions unless still referenced by mixed code.
  mixed_body: list[ir.Stmt] = []
  emitted_uids: set[int] = set()
  for st in red.body:
    if isinstance(st, ir.Comment):
      continue
    if isinstance(st, Assign) and stage_side(st.value) is not None:
      continue
    if isinstance(st, Assign):
      mixed_body.append(Assign(st.var, rewrite_expr(st.value)))
      emitted_uids.add(st.var.uid)
    else:
      return None

  mixed_folds: list[ir.Fold] = []
  for f in red.folds:
    new_payloads = [ir.Payload(p.result, rewrite_expr(p.value), p.a, p.b, p.combined) for p in f.payloads]
    mixed_folds.append(
      ir.Fold(f.result, rewrite_expr(f.value), f.a, f.b, f.combine, f.combined, f.init, f.op, f.empty, new_payloads)
    )

  if not any(side in ("row", "col") for side in stage_sides):
    return None

  # Collect uniform (i,j,k-invariant, load-free) checks from m.before; reject any non-uniform ones
  pre_checks: list[ir.Check] = []
  local_uids = set(def_map) | {i_uid, j_uid, k_uid}
  for t in m.before:
    if isinstance(t, ir.Check):
      for e in (t.cond, *t.args):
        ui, uj, uk, hl, p = expr_mask(e)
        if ui or uj or uk or hl or not p or any(isinstance(x, Var) and x.uid in local_uids for x in ir.walk_expr(e)):
          return None
      pre_checks.append(t)

  # Pull in any remaining non-staged local Assigns from m.before + red.body still
  # referenced by mixed_body / mixed_folds (e.g. a non-k invariant or 3D mixed load).
  needed: set[int] = set()
  for st in mixed_body:
    for e in ir.stmt_exprs(st):
      collect_deps(e, needed)
  for f in mixed_folds:
    collect_deps(f.value, needed)
    for p in f.payloads:
      collect_deps(p.value, needed)
  prefix_deps = [def_map[u] for u in sorted(needed - emitted_uids, key=lambda u: order_map[u])]
  full_mixed_body = prefix_deps + mixed_body
  has_mixed_loads = any(
    isinstance(x, Load) for st in full_mixed_body for e in ir.stmt_exprs(st) for x in ir.walk_expr(e)
  ) or any(
    isinstance(x, Load) for f in mixed_folds for e in (f.value, *(p.value for p in f.payloads)) for x in ir.walk_expr(e)
  )
  return ir.SharedTilePlan(
    shape=shape,
    index=s.index,
    before=list(m.before),
    reduce=ir.IndexReduce(extent, red.index, red.body, red.folds, red.scope, red.home, False, red.loc),
    after=list(m.after),
    stages=stages,
    stage_sides=tuple(stage_sides),
    stage_vars=tuple(stage_vars),
    mixed_body=full_mixed_body,
    mixed_folds=mixed_folds,
    has_mixed_loads=has_mixed_loads,
    pre_checks=pre_checks,
  )


def tiled(s: IndexMap, m: Shape, defs: dict[int, Expr], owner: Scope, tile_bytes: int) -> list:
  """Rewrite a matmul-shaped ``IndexMap`` into nested :class:`IndexTile`
  operators: an outer L2 ``level="cache"`` tile over rows and columns wrapping
  an inner ``level="register"`` micro-tile (``UNROLL_I x UNROLL_J``), with an
  optional :class:`~parakeet.syntax.ir.SharedTilePlan` for GPU shared-memory
  block tiling."""
  out: list = []
  lp = Loops(out, defs, owner)
  rank = len(s.shape)
  shape = tuple(lp.tmp(d, "n") for d in s.shape)
  extent = tuple(lp.tmp(d, "k") for d in m.extent)
  length = lp.tmp(iprod(extent), "len")
  itemsize = m.reduce.folds[0].a.type.bits // 8
  col_tile = tile_size(lp, length, itemsize, tile_bytes)
  shared_plan = extract_shared_tile_plan(s, m, shape, extent)

  # Outer cache tile: step leading batch axes 1-by-1, tile rows by ROW_TILE and columns by col_tile
  cache_tile_shape: tuple[Expr | None, ...] = (
    *(ONE,) * (rank - 2),
    Const(ROW_TILE, I),
    col_tile,
  )
  cache_origin = tuple(Var("t0", I) for _ in range(rank))
  cache_extent = tuple(Var("te", I) for _ in range(rank))

  # Inner register micro-tile over the cache tile's clamped extent
  reg_tile_shape: tuple[Expr | None, ...] = (
    *(None,) * (rank - 2),
    Const(UNROLL_I, I),
    Const(UNROLL_J, I),
  )
  reg_origin = tuple(Var("r0", I) for _ in range(rank))
  reg_extent = tuple(Var("re", I) for _ in range(rank))

  # Single-element IndexMap inside the register tile (coords = cache_origin + reg_origin + elem_idx)
  elem_index = tuple(Var("u", I) for _ in range(rank))
  subst = {v.uid: iadd(iadd(co, ro), u) for v, co, ro, u in zip(s.index, cache_origin, reg_origin, elem_index)}
  cloner = Cloner(subst)
  elem_body = cloner.block(m.before + [m.reduce] + m.after)
  elem_scope = Scope("loop")
  elem_body.append(ir.ReleaseScope(elem_scope))
  elem_map = IndexMap(reg_extent, elem_index, elem_body, elem_scope, False, False, s.loc)

  reg_scope = Scope("loop")
  reg_tile = IndexTile(
    cache_extent,
    reg_tile_shape,
    reg_origin,
    reg_extent,
    [],
    [elem_map, ir.ReleaseScope(reg_scope)],
    reg_scope,
    level="register",
    parallel=False,
    loc=s.loc,
  )
  cache_scope = Scope("loop")
  cache_tile = IndexTile(
    shape,
    cache_tile_shape,
    cache_origin,
    cache_extent,
    [],
    [reg_tile, ir.ReleaseScope(cache_scope)],
    cache_scope,
    level="cache",
    parallel=s.parallel,
    loc=s.loc,
    shared_tile=shared_plan,
  )
  out.append(cache_tile)
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
  "ROW_TILE",
  "UNROLL_I",
  "UNROLL_J",
  "extract_shared_tile_plan",
  "interchange",
  "interchange_block",
  "interchanged",
  "linear_stride",
  "match",
  "should_interchange",
  "tile_block",
  "tile_size",
  "tiled",
  "tiling",
]
