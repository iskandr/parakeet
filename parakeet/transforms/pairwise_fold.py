"""Distribute a map of structured folds over rectangular tiles.

A query strip owns its summaries. For each consecutive reference tile: evaluate
the score map in register tiles, fold each row into a partial summary, then
merge that partial on the right. The score and merge are cloned from the IR.
"""

from ..syntax import ir, ops
from ..syntax.clone import Cloner, defined
from ..syntax.ops import ZERO, I, cmp, iadd, imul
from .analysis import BufferInfo, buffer_accesses, is_pure_expr
from .tiling import UNROLL_I, UNROLL_J

ROWS = 16
COLUMNS = 128


def _depends(exprs, ids):
  return any(isinstance(v, ir.Var) and v.uid in ids for e in exprs for v in ir.walk_expr(e))


def _split(block, kind):
  hits = [k for k, s in enumerate(block) if isinstance(s, kind)]
  if len(hits) == 1:
    k = hits[0]
    return block[:k], block[k], block[k + 1 :]
  return None


def _scalar_code(block):
  return all(
    isinstance(s, ir.Assign) and not isinstance(s.var.type, ir.BufType) and is_pure_expr(s.value) for s in block
  )


def _match(rows, info):
  """Find a rectangular, single-use score and a regroupable row reduction."""
  if len(rows.shape) != 1 or rows.ordered or rows.scope.owned:
    return None
  parts = _split(
    [s for s in rows.body if not (isinstance(s, ir.ReleaseScope) and s.scope is rows.scope)], ir.IndexReduce
  )
  if parts is None:
    return None
  before, refs, outputs = parts
  if len(refs.shape) != 1 or refs.pre or refs.where is not None or refs.scope.owned or len(refs.folds) != 1:
    return None
  fold = refs.folds[0]
  if not fold.carries or not fold.op or fold.init is not None or fold.empty is None or not _scalar_code(fold.combine):
    return None
  score = _split(refs.body, ir.IndexReduce)
  if score is None:
    return None
  pre, features, post = score
  if (
    features.pre
    or features.where is not None
    or features.scope.owned
    or not _scalar_code([*pre, *features.body, *post])
    or any(f.init is None or f.combine or f.carries or not f.op for f in features.folds)
  ):
    return None
  # Hoist only uniform shape checks. Keep row-offset definitions in the point program.
  row_defs = [s for s in before if isinstance(s, ir.Assign)]
  checks = [s for s in before if not isinstance(s, ir.Assign)]
  row_ids = {rows.index[0].uid, *defined(row_defs)}
  if (
    not _scalar_code(row_defs)
    or _depends(refs.shape, row_ids)
    or any(
      not isinstance(s, (ir.If, ir.Check))
      or _depends(ir.stmt_exprs(s), row_ids)
      or any(isinstance(v, (ir.Load, ir.RtCall)) for e in ir.stmt_exprs(s) for v in ir.walk_expr(e))
      for s in ir.walk_stmts(checks)
    )
  ):
    return None
  if not outputs or any(not isinstance(s, ir.Store) or s.atomic or s.buf.uid not in info.local_allocs for s in outputs):
    return None
  writes = {s.buf.uid for s in outputs}
  reads, _ = buffer_accesses(rows.body)
  if info.may_alias(reads, writes):
    return None
  points = set(defined(refs.body))
  if _depends([c.value for c in fold.carries], points) or _depends(
    features.shape, row_ids | {refs.index[0].uid} | set(defined(pre))
  ):
    return None
  # The tile merge can depend on the query, but has no distinguished reference.
  merge_exprs = [e for stmt in fold.combine for e in ir.stmt_exprs(stmt)] + [f.combined for f in (fold, *fold.carries)]
  if _depends(merge_exprs, points | {refs.index[0].uid}):
    return None
  if not all(is_pure_expr(e) for s in ir.walk_stmts([refs, *outputs]) for e in ir.stmt_exprs(s)):
    return None

  return refs, fold, row_defs, checks, outputs


def _tile(rows, refs, fold, row_defs, checks, outputs):
  row, col = rows.index[0], refs.index[0]
  base, height = ir.Var("query_base", I), ir.Var("query_count", I)
  scope = ir.Scope("loop")
  fields = [fold, *fold.carries]
  scores = ir.Var("score_tile", ir.BufType(fold.value.type.strong))
  states = [ir.Var("row_summary", ir.BufType(f.result.type.strong)) for f in fields]
  scope.owned.extend([scores, *states])
  body = [*checks, ir.Check(cmp("gt", refs.shape[0], ZERO), fold.empty)]
  body += [ir.Alloc(scores, ir.Const(ROWS * COLUMNS, I)), *(ir.Alloc(s, ir.Const(ROWS, I)) for s in states)]

  tile = ir.Var("ref_tile", I)
  offset, width = ir.Var("ref_base", I), ir.Var("ref_count", I)
  ref_body = [
    ir.Assign(offset, imul(tile, COLUMNS)),
    ir.Assign(width, ops.imin(COLUMNS, ops.isub(refs.shape[0], offset))),
  ]
  # The existing register tile schedule jams the feature folds for adjacent pairs.
  axes = (height, width)
  origin = tuple(ir.Var("micro_base", I) for _ in axes)
  extent = tuple(ir.Var("micro_count", I) for _ in axes)
  index = tuple(ir.Var("point", I) for _ in axes)
  r, j = (iadd(o, i) for o, i in zip(origin, index))
  cl = Cloner({row.uid: iadd(base, r), col.uid: iadd(offset, j)}, home=scope)
  point = cl.block([*row_defs, *refs.body])
  point.append(ir.Store(scores, iadd(imul(r, COLUMNS), j), cl.expr(fold.value)))
  point_map = ir.IndexMap(extent, index, point, ir.Scope("loop"), parallel=False)
  ref_body.append(
    ir.IndexTile(
      axes,
      (ir.Const(UNROLL_I, I), ir.Const(UNROLL_J, I)),
      origin,
      extent,
      [],
      [point_map],
      ir.Scope("loop"),
      level="register",
      parallel=False,
    )
  )

  r, j = ir.Var("r", I), ir.Var("j", I)
  row_cl = Cloner({row.uid: iadd(base, r), col.uid: iadd(offset, j)})
  update = row_cl.block(row_defs)
  partial = row_cl.fold(fold)
  partial.value = ir.Load(scores, iadd(imul(r, COLUMNS), j))
  update.append(ir.IndexReduce((width,), (j,), [], [partial], ir.Scope("loop"), scope))
  pieces = [p.result for p in (partial, *partial.carries)]
  previous = [ir.Var("previous", f.a.type) for f in fields]
  merge = [ir.Assign(v, ir.Load(buf, r)) for v, buf in zip(previous, states)]
  bindings = dict(row_cl.m)
  for f, a, b in zip(fields, previous, pieces):
    bindings.update({f.a.uid: a, f.b.uid: b})
  combine = Cloner(bindings)
  merge.extend(combine.block(fold.combine))
  merge.extend(ir.Store(buf, r, combine.expr(f.combined)) for buf, f in zip(states, fields))
  first = [ir.Store(buf, r, v) for buf, v in zip(states, pieces)]
  update.append(ir.If(cmp("eq", tile, ZERO), first, merge))
  ref_body.append(ir.IndexMap((height,), (r,), update, ir.Scope("loop"), parallel=False))

  blocks = iadd(ops.idiv(refs.shape[0], COLUMNS), ops.cast(cmp("ne", ops.imod(refs.shape[0], COLUMNS), ZERO), I))
  body.append(ir.IndexMap((blocks,), (tile,), ref_body, ir.Scope("loop"), ordered=True, parallel=False))
  r = ir.Var("r", I)
  out = Cloner({row.uid: iadd(base, r), **{f.result.uid: ir.Load(buf, r) for f, buf in zip(fields, states)}})
  body.append(ir.IndexMap((height,), (r,), out.block([*row_defs, *outputs]), ir.Scope("loop"), parallel=False))
  return ir.IndexTile(
    rows.shape, (ir.Const(ROWS, I),), (base,), (height,), [], body, scope, parallel=rows.parallel, loc=rows.loc
  )


def tile_pairwise_folds(fn, options):
  if not (options.tiling and options.fusion) or options.target.is_gpu:
    return
  info = BufferInfo.analyze(fn)

  def visit(block):
    for i, s in enumerate(block):
      match = _match(s, info) if isinstance(s, ir.IndexMap) else None
      if match is not None:
        block[i] = _tile(s, *match)
      else:
        for child in ir.sub_blocks(s):
          visit(child)

  visit(fn.body)
