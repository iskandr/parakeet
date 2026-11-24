"""Distribute a map of structured folds over rectangular tiles.

A query strip owns its summaries. For each consecutive reference tile: evaluate
the score map in register tiles, fold each row into a partial summary, then
merge that partial on the right. The score and merge are cloned from the IR.
"""

from dataclasses import dataclass

from ..syntax import ir, ops
from ..syntax.clone import Cloner, defined
from ..syntax.ops import ZERO, I, cmp, iadd, imul
from ..target import choose_tile_config
from . import lowering, tiling
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


@dataclass
class ScoreProgram:
  body: list
  value: ir.Expr
  lift: list | None

  def tile_fold(self, fold, cloner, tile_value):
    """Read the cached score and reconstruct all dependent state fields."""
    if self.lift is None:
      partial = cloner.fold(fold)
      partial.value = tile_value
      return [], partial
    cloner.m[self.value.uid] = tile_value
    body = cloner.block(self.lift)
    return body, cloner.fold(fold)


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
  scalar_merge = all(isinstance(s, ir.If) or _scalar_code([s]) for s in ir.walk_stmts(fold.combine))
  if not fold.carries and fold.op != "structured":
    return None
  if not fold.op or fold.init is not None or fold.empty is None or not scalar_merge:
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
  stores = [s for s in outputs if isinstance(s, ir.Store)]
  if not stores or any(s.atomic or s.buf.uid not in info.local_allocs for s in stores):
    return None
  if not _scalar_code([s for s in outputs if not isinstance(s, ir.Store)]):
    return None
  writes = {s.buf.uid for s in stores}
  reads, _ = buffer_accesses(rows.body)
  if info.may_alias(reads, writes):
    return None
  points = set(defined(refs.body))
  if _depends(features.shape, row_ids | {refs.index[0].uid} | set(defined(pre))):
    return None
  # The tile merge can depend on the query, but has no distinguished reference.
  merge_exprs = [e for stmt in ir.walk_stmts(fold.combine) for e in ir.stmt_exprs(stmt)] + [
    f.combined for f in (fold, *fold.carries)
  ]
  if _depends(merge_exprs, points | {refs.index[0].uid}):
    return None
  if not all(is_pure_expr(e) for s in ir.walk_stmts([refs, *outputs]) for e in ir.stmt_exprs(s)):
    return None

  if _depends([c.value for c in fold.carries], points):
    # Cache the feature reduction once. Its epilogue can lift the scalar
    # into any number of dependent fields inside the consuming row fold.
    if len(features.folds) != 1:
      return None
    program = ScoreProgram([*pre, features], features.folds[0].result, [*pre, *post])
  else:
    program = ScoreProgram(refs.body, fold.value, None)
  return refs, fold, row_defs, checks, outputs, program


def _tile(rows, refs, fold, row_defs, checks, outputs, program):
  row, col = rows.index[0], refs.index[0]
  base, height = ir.Var("query_base", I), ir.Var("query_count", I)
  scope = ir.Scope("loop")
  fields = [fold, *fold.carries]
  scores = ir.Var("score_tile", ir.BufType(program.value.type.strong))
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
  point = cl.block([*row_defs, *program.body])
  point.append(ir.Store(scores, iadd(imul(r, COLUMNS), j), cl.expr(program.value)))
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
  lift, partial = program.tile_fold(fold, row_cl, ir.Load(scores, iadd(imul(r, COLUMNS), j)))
  update.append(ir.IndexReduce((width,), (j,), lift, [partial], ir.Scope("loop"), scope))
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


def _gpu_tiles(rows, refs, fold, row_defs, checks, outputs, program, owner, target):
  """Produce one structured partial per rectangular score tile, then reduce its rows in reference order."""
  if not rows.parallel:
    return None
  row, col = rows.index[0], refs.index[0]
  prefix = []
  lp = lowering.Loops(prefix, {}, owner)
  shape = tuple(lp.tmp(d, "pair_extent") for d in (rows.shape[0], refs.shape[0]))
  point = ir.IndexMap(shape, (row, col), [*row_defs, *program.body], ir.Scope("loop"))
  match = tiling.match(point, {}, min_rank=1)
  if match is None:
    return None
  # Register accumulators are initialized before tile coordinates are bound.
  # Only a uniform seed is valid for every point, including padded edge lanes.
  bound = set(defined(point.body)) | {row.uid, col.uid}
  if _depends([f.init for f in match.reduce.folds], bound):
    return None
  plan = tiling.extract_shared_tile_plan(point, match, shape, match.extent)
  if plan is None:
    return None
  scores = ir.Var("score_tile", ir.BufType(program.value.type.strong))
  cfg = choose_tile_config(plan, profile=target, score_bytes=scores.type.elt.itemsize)
  if cfg is None:
    return None
  bm, bn = cfg[:2]
  nq, nr = shape
  blocks = lp.tmp(iadd(ops.idiv(nr, bn), ops.cast(cmp("ne", ops.imod(nr, bn), ZERO), I)), "ref_tiles")
  fields = (fold, *fold.carries)
  states = [ir.Var("tile_summary", ir.BufType(f.result.type.strong)) for f in fields]
  owner.owned.extend(states)
  count = lp.tmp(imul(nq, blocks), "summary_count")
  prefix.append(ir.If(cmp("gt", nq, ZERO), [*checks, ir.Check(cmp("gt", nr, ZERO), fold.empty)], []))
  prefix.extend(ir.Alloc(buf, count) for buf in states)

  q0, j0 = ir.Var("query_base", I), ir.Var("ref_base", I)
  height, width = ir.Var("query_count", I), ir.Var("ref_count", I)
  r, j = ir.Var("r", I), ir.Var("j", I)
  q, ref = iadd(q0, r), iadd(j0, j)
  tile_scope = ir.Scope("loop")
  tile_scope.owned.append(scores)
  # Both the generic tile body and the shared-memory schedule compute the same
  # point program. Only the location and lifetime of its tile differ.
  plan.after.append(ir.Store(scores, iadd(imul(ops.imod(row, bm), bn), ops.imod(col, bn)), program.value))
  score_copy = Cloner({row.uid: q, col.uid: ref}, home=tile_scope)
  score_body = score_copy.block(point.body)
  score_body.append(ir.Store(scores, iadd(imul(r, bn), j), score_copy.expr(program.value)))
  score_map = ir.IndexMap((height, width), (r, j), score_body, ir.Scope("loop"), parallel=False)

  row_copy = Cloner({row.uid: q, col.uid: ref}, home=tile_scope)
  row_body = row_copy.block(row_defs)
  lift, local = program.tile_fold(fold, row_copy, ir.Load(scores, iadd(imul(r, bn), j)))
  local.empty = None  # Every dispatched reference tile has positive width.
  row_body.append(ir.IndexReduce((width,), (j,), lift, [local], ir.Scope("loop"), tile_scope))
  dest = iadd(imul(q, blocks), ops.idiv(j0, bn))
  row_body.extend(ir.Store(buf, dest, f.result) for buf, f in zip(states, (local, *local.carries)))
  partial_rows = ir.IndexMap((height,), (r,), row_body, ir.Scope("loop"), parallel=False)
  plan.row_fold = ir.TileRowFold(scores, (q0, j0), width, partial_rows, cfg)
  tile = ir.IndexTile(
    shape,
    (ir.Const(bm, I), ir.Const(bn, I)),
    (q0, j0),
    (height, width),
    [],
    [
      ir.Alloc(scores, ir.Const(bm * bn, I)),
      score_map,
      partial_rows,
    ],
    tile_scope,
    level="shared",
    loc=rows.loc,
    shared_tile=plan,
  )

  tile_id = ir.Var("ref_tile", I)
  merge_copy = Cloner(home=rows.scope)
  merge_body = merge_copy.block(row_defs)
  final = merge_copy.fold(fold)
  at = iadd(imul(row, blocks), tile_id)
  for f, buf in zip((final, *final.carries), states):
    f.value = ir.Load(buf, at)
  final.empty = None  # Checked before the first stage.
  merge_body.append(ir.IndexReduce((blocks,), (tile_id,), [], [final], ir.Scope("loop"), rows.scope))
  merge_body.extend(merge_copy.block(outputs))
  merge = ir.IndexMap((nq,), (row,), merge_body, rows.scope, loc=rows.loc)
  return [*prefix, tile, merge, *(ir.Release(buf) for buf in states)]


def tile_pairwise_folds(fn, options):
  if not (options.tiling and options.fusion):
    return
  info = BufferInfo.analyze(fn)

  def visit(block, owner):
    out = []
    for s in block:
      match = _match(s, info) if isinstance(s, ir.IndexMap) else None
      if match is not None:
        if not options.target.is_gpu:
          out.append(_tile(s, *match))
          continue
        tiles = _gpu_tiles(s, *match, owner, options.target)
        if tiles is not None:
          out.extend(tiles)
          continue
      for child in ir.sub_blocks(s):
        visit(child, getattr(s, "scope", owner))
      out.append(s)
    block[:] = out

  visit(fn.body, fn.scope)
