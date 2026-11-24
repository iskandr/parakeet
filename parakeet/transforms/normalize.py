"""Pre-fusion operator normalization passes.

Runs right after dead-code elimination so every downstream pass (fusion,
reduction interchange, tiling, and GPU shared-memory extraction) sees clean,
canonical operator statements:

* :func:`hoist_invariant_checks`: lifts loop-invariant ``Check`` statements
  out of ``IndexMap`` / ``IndexReduce`` / ``IndexFilter`` bodies, guarded by
  ``if prod(shape) > 0`` to preserve exact empty-array exception semantics.
* :func:`squeeze_unit_map`: drops ``Const(1)`` axes from ``IndexMap`` (turning
  ``keepdims=True`` ``(M, 1)`` maps into ``(M,)`` so they fuse cleanly).
* :func:`split_invariant_map`: hoists inner ``IndexReduce``s out of any
  ``IndexMap`` axes they do not depend on (leading, middle, or trailing),
  eliminating ``O(M * N^2)`` / ``O(M^2 * N)`` redundant recomputation in
  ``softmax``, ``layer_norm``, and column variance.
"""

from __future__ import annotations

from ..syntax import ir
from ..syntax.clone import Cloner, defined
from ..syntax.ir import (
  Assign,
  Check,
  Const,
  Expr,
  If,
  IndexFilter,
  IndexMap,
  IndexReduce,
  IndexScatter,
  Load,
  Scope,
  Var,
)
from ..syntax.ops import ZERO, cast, cmp, iprod
from .analysis import IndexFacts, is_pure_expr


def _ekey(e: Expr):
  if isinstance(e, Var):
    return ("v", e.uid)
  if isinstance(e, Const):
    return ("c", e.type, type(e.value), e.value)
  if isinstance(e, ir.Prim):
    return ("p", e.op, e.type, tuple(_ekey(a) for a in e.args))
  if isinstance(e, ir.Cast):
    return ("t", e.type, _ekey(e.arg))
  if isinstance(e, Load):
    return ("l", e.buf.uid, _ekey(e.index))
  if isinstance(e, ir.RtCall):
    return ("r", e.name, e.pure, e.type, tuple(_ekey(a) for a in e.args))
  return ("o", id(e))


def hoist_checks_from_op(
  s: IndexMap | IndexReduce | IndexFilter | IndexScatter, facts: IndexFacts
) -> tuple[list[ir.Stmt], bool]:
  """Lift loop-invariant :class:`Check` statements from the leading straight-line
  prefix of ``s`` before ``s``, guarded by ``if prod(shape) > 0``."""
  if isinstance(s, IndexMap):
    if s.ordered:
      return [s], False
    shape = s.shape
    bound = {v.uid for v in s.index}
    body = s.body
  elif isinstance(s, IndexReduce):
    if s.where is not None:
      return [s], False
    shape = s.shape
    bound = {v.uid for v in s.index}
    body = s.body
  elif isinstance(s, IndexFilter):
    shape = (s.n,)
    bound = {s.index.uid}
    body = s.body
  elif isinstance(s, IndexScatter):
    shape = s.shape
    bound = {v.uid for v in s.index}
    body = s.body
  else:
    return [s], False

  inner_defs = set(defined(body)) | bound
  hoisted: list[Check] = []
  kept: list[ir.Stmt] = []
  can_hoist = True
  for t in body:
    if can_hoist and isinstance(t, Check) and is_pure_expr(t.cond) and all(is_pure_expr(a) for a in t.args):
      cond = facts.lift_outside(t.cond, inner_defs)
      args = [facts.lift_outside(a, inner_defs) for a in t.args]
      if cond is not None and all(a is not None for a in args):
        hoisted.append(Check(cond, t.err, tuple(args)))  # type: ignore[arg-type]
        continue
      # A non-hoistable Check may raise earlier: do not hoist later checks past it
      can_hoist = False
    elif isinstance(t, (ir.Store, If, ir.ForRange, ir.While, ir.ParLoop, *ir.OPERATORS)):
      can_hoist = False
    elif isinstance(t, Assign) and not is_pure_expr(t.value):
      can_hoist = False
    kept.append(t)

  if not hoisted:
    return [s], False

  body[:] = kept
  # Only run hoisted checks when the operator actually has >= 1 iteration
  if all(isinstance(d, Const) and d.value > 0 for d in shape):
    return [*hoisted, s], True
  total = iprod(shape)
  return [If(cmp("gt", total, ZERO), list(hoisted), []), s], True


def squeeze_unit_map(s: IndexMap) -> IndexMap | None:
  """Drop ``Const(1)`` axes from an ``IndexMap`` by substituting ``0``.

  Turns ``keepdims=True`` maps like ``map i, j over (M, 1)`` into ``map i over (M,)``,
  enabling vertical and sibling fusion with ``(M,)`` consumers (e.g. ``layer_norm``)."""
  if len(s.shape) < 2 or not any(isinstance(d, Const) and d.value == 1 for d in s.shape):
    return None
  kept_shape: list[Expr] = []
  kept_index: list[Var] = []
  subst: dict[int, Expr] = {}
  for d, v in zip(s.shape, s.index):
    if isinstance(d, Const) and d.value == 1:
      subst[v.uid] = Const(0, v.type)
    else:
      kept_shape.append(d)
      kept_index.append(v)
  if not kept_shape:
    return None
  cl = Cloner(subst)
  for v in defined(s.body).values():
    cl.var(v)
  new_body = [cl.stmt(t) for t in s.body]
  return IndexMap(tuple(kept_shape), tuple(kept_index), new_body, cl.scope(s.scope), s.ordered, s.parallel, s.loc)


def split_invariant_map(s: IndexMap) -> list[ir.Stmt] | None:
  """Hoist an ``IndexReduce`` (and its setup) out of any ``IndexMap`` axes
  that it does not depend on (leading, middle, or trailing).

  Fixes ``O(M * N^2)`` / ``O(M^2 * N)`` redundant recomputation in expressions
  like ``x - np.max(x, axis=1, keepdims=True)`` (``softmax``, ``layer_norm``)
  and ``x - np.mean(x, axis=0, keepdims=True)`` (column variance / batch norm)."""
  if s.ordered or s.scope.owned or len(s.shape) < 2:
    return None
  body = [t for t in s.body if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
  red_positions = [k for k, t in enumerate(body) if isinstance(t, IndexReduce)]
  if not red_positions:
    return None
  last_red = red_positions[-1]
  prefix = body[: last_red + 1]
  tail = body[last_red + 1 :]
  if not tail or any(isinstance(t, ir.Store) for t in prefix):
    return None
  used_uids = {
    x.uid for st in ir.walk_stmts(prefix) for e in ir.stmt_exprs(st) for x in ir.walk_expr(e) if isinstance(x, Var)
  }
  used_axes = [k for k, idx_var in enumerate(s.index) if idx_var.uid in used_uids]
  inv_axes = [k for k, idx_var in enumerate(s.index) if idx_var.uid not in used_uids]
  if not inv_axes:
    return None  # prefix depends on all axes (e.g. matmul)

  prefix_can_raise = any(
    isinstance(x, Check) or (isinstance(x, IndexReduce) and any(f.empty is not None for f in x.folds))
    for x in ir.walk_stmts(prefix)
  )
  inv_shape = tuple(s.shape[k] for k in inv_axes)
  inv_index = tuple(s.index[k] for k in inv_axes)
  tail_scope = Scope("loop")
  inner_map = IndexMap(
    inv_shape,
    inv_index,
    tail + [ir.ReleaseScope(tail_scope)],
    tail_scope,
    False,
    s.parallel if not used_axes else False,
    s.loc,
  )
  if not used_axes:
    if prefix_can_raise:
      return [If(cmp("gt", iprod(s.shape), ZERO), prefix + [inner_map], [])]
    return prefix + [inner_map]

  inner_block: list[ir.Stmt] = prefix + [inner_map]
  if prefix_can_raise:
    inner_block = [If(cmp("gt", iprod(inv_shape), ZERO), inner_block, [])]
  outer_map = IndexMap(
    tuple(s.shape[k] for k in used_axes),
    tuple(s.index[k] for k in used_axes),
    inner_block + [ir.ReleaseScope(s.scope)],
    s.scope,
    False,
    s.parallel,
    s.loc,
  )
  return [outer_map]


def _inline_pure_expr(e: Expr, defs: dict[int, Expr], depth: int = 16) -> Expr:
  """Substitute straight-line pure local definitions into ``e``."""
  if depth <= 0:
    return e
  if isinstance(e, Var):
    d = defs.get(e.uid)
    return _inline_pure_expr(d, defs, depth - 1) if d is not None else e
  if isinstance(e, ir.Prim):
    return ir.Prim(e.op, tuple(_inline_pure_expr(a, defs, depth - 1) for a in e.args), e.type)
  if isinstance(e, ir.Cast):
    return ir.Cast(_inline_pure_expr(e.arg, defs, depth - 1), e.type)
  if isinstance(e, Load):
    return Load(e.buf, _inline_pure_expr(e.index, defs, depth - 1))
  if isinstance(e, ir.RtCall):
    return ir.RtCall(e.name, tuple(_inline_pure_expr(a, defs, depth - 1) for a in e.args), e.type, e.pure)
  return e


def _expr_matches_indexed(e: Expr, template: Expr, idx_uid: int, arg_aliases: set[int]) -> bool:
  """True if ``e`` is identical to ``template`` with ``idx_uid`` replaced by a var in ``arg_aliases``."""
  if isinstance(template, Var) and template.uid == idx_uid:
    return isinstance(e, Var) and e.uid in arg_aliases
  if type(e) is not type(template):
    return False
  if isinstance(e, Var):
    return e.uid == template.uid  # type: ignore[union-attr]
  if isinstance(e, Const):
    return e.type == template.type and e.value == template.value and type(e.value) is type(template.value)  # type: ignore[union-attr]
  if isinstance(e, Load):
    return e.buf.uid == template.buf.uid and _expr_matches_indexed(e.index, template.index, idx_uid, arg_aliases)  # type: ignore[union-attr]
  if isinstance(e, ir.Cast):
    return e.type == template.type and _expr_matches_indexed(e.arg, template.arg, idx_uid, arg_aliases)  # type: ignore[union-attr]
  if isinstance(e, ir.Prim):
    return (
      e.op == template.op  # type: ignore[union-attr]
      and e.type == template.type
      and len(e.args) == len(template.args)  # type: ignore[union-attr]
      and all(_expr_matches_indexed(a, b, idx_uid, arg_aliases) for a, b in zip(e.args, template.args))  # type: ignore[union-attr]
    )
  if isinstance(e, ir.RtCall):
    return (
      e.name == template.name  # type: ignore[union-attr]
      and e.pure
      and template.pure  # type: ignore[union-attr]
      and e.type == template.type
      and len(e.args) == len(template.args)  # type: ignore[union-attr]
      and all(_expr_matches_indexed(a, b, idx_uid, arg_aliases) for a, b in zip(e.args, template.args))  # type: ignore[union-attr]
    )
  return False


def _is_inbounds_check_for(cond: Expr, aliases: set[int], shape_k) -> bool:
  """True for ``(a >= 0) and (a < shape)`` where ``a`` is in ``aliases``."""
  if not (isinstance(cond, ir.Prim) and cond.op == "and" and len(cond.args) == 2):
    return False
  lo, hi = cond.args
  if not (isinstance(lo, ir.Prim) and lo.op == "ge" and isinstance(lo.args[0], Var) and lo.args[0].uid in aliases):
    return False
  if not (isinstance(lo.args[1], Const) and lo.args[1].value == 0):
    return False
  if not (isinstance(hi, ir.Prim) and hi.op == "lt" and isinstance(hi.args[0], Var) and hi.args[0].uid in aliases):
    return False
  return _ekey(hi.args[1]) == shape_k


def reuse_extrema_values(block: list[ir.Stmt]) -> tuple[list[ir.Stmt], bool]:
  """Replace ``f(arg)`` after a pure 1-D ``argmin``/``argmax`` reduction ``(best, arg)``
  with ``best`` and drop the redundant bounds check on ``arg``.

  Eliminates redundant recomputation of patterns like ``i = np.argmax(gain); return i, gain[i]``
  and avoids host-side array downloads when ``gain`` lives on a GPU."""
  # Active records: list of dicts with keys idx_uid, shape_k, val_expr, best_var, read_bufs, read_vars, aliases
  active: list[dict] = []
  out: list[ir.Stmt] = []
  changed = False

  def invalidate_for_stmt(st: ir.Stmt) -> None:
    if not active:
      return
    if isinstance(st, (If, ir.ForRange, ir.While, ir.ParLoop, ir.IndexTile)):
      active.clear()
      return
    written_bufs: set[int] = set()
    written_vars: set[int] = set()
    impure = False
    for sub in ir.walk_stmts([st]):
      if isinstance(sub, ir.Store):
        written_bufs.add(sub.buf.uid)
      elif isinstance(sub, IndexScatter):
        written_bufs.update(u.buf.uid for u in sub.updates)
      elif isinstance(sub, Assign):
        written_vars.add(sub.var.uid)
        if any(isinstance(x, ir.RtCall) and not x.pure for x in ir.walk_expr(sub.value)):
          impure = True
      elif isinstance(sub, ir.Release):
        written_bufs.add(sub.var.uid)
    if impure:
      active.clear()
      return
    active[:] = [
      rec for rec in active if not (rec["read_bufs"] & written_bufs) and not (rec["read_vars"] & written_vars)
    ]

  for s in block:
    if isinstance(s, IndexReduce):
      invalidate_for_stmt(s)
      out.append(s)
      if len(s.shape) == 1 and s.where is None and not s.scope.owned:
        local_defs: dict[int, Expr] = {}
        pure_body = True
        for t in s.body:
          if isinstance(t, (ir.Comment, ir.ReleaseScope)):
            continue
          if isinstance(t, Assign) and is_pure_expr(t.value):
            local_defs[t.var.uid] = t.value
          else:
            pure_body = False
            break
        if pure_body:
          idx_v = s.index[0]
          shape_k = _ekey(s.shape[0])
          for f in s.folds:
            if f.op in ("argmin", "argmax") and f.init is None and f.empty is not None and len(f.carries) == 1:
              cy = f.carries[0]
              cy_val = _inline_pure_expr(cy.value, local_defs)
              while isinstance(cy_val, ir.Cast):
                cy_val = cy_val.arg
              if isinstance(cy_val, Var) and cy_val.uid == idx_v.uid:
                val_expr = _inline_pure_expr(f.value, local_defs)
                if is_pure_expr(val_expr):
                  rbufs = {x.buf.uid for x in ir.walk_expr(val_expr) if isinstance(x, Load)}
                  rvars = {x.uid for x in ir.walk_expr(val_expr) if isinstance(x, Var) and x.uid != idx_v.uid}
                  active.append(
                    {
                      "idx_uid": idx_v.uid,
                      "shape_k": shape_k,
                      "val_expr": val_expr,
                      "best_var": f.result,
                      "read_bufs": rbufs,
                      "read_vars": rvars,
                      "aliases": {cy.result.uid},
                    }
                  )
      continue

    if active and isinstance(s, Check):
      if any(_is_inbounds_check_for(s.cond, rec["aliases"], rec["shape_k"]) for rec in active):
        changed = True
        continue

    if active and isinstance(s, Assign):
      val = s.value
      # Track pure copies / integer casts / negative-index wraps of arg
      for rec in active:
        aliases = rec["aliases"]
        if isinstance(val, Var) and val.uid in aliases:
          aliases.add(s.var.uid)
        elif isinstance(val, ir.Cast) and isinstance(val.arg, Var) and val.arg.uid in aliases and val.type.kind in "iu":
          aliases.add(s.var.uid)
        elif (
          isinstance(val, ir.Prim)
          and val.op == "select"
          and len(val.args) == 3
          and isinstance(val.args[0], ir.Prim)
          and val.args[0].op == "lt"
          and isinstance(val.args[0].args[0], Var)
          and val.args[0].args[0].uid in aliases
          and isinstance(val.args[0].args[1], Const)
          and val.args[0].args[1].value == 0
          and isinstance(val.args[1], ir.Prim)
          and val.args[1].op == "add"
          and isinstance(val.args[1].args[0], Var)
          and val.args[1].args[0].uid in aliases
          and _ekey(val.args[1].args[1]) == rec["shape_k"]
          and isinstance(val.args[2], Var)
          and val.args[2].uid in aliases
        ):
          aliases.add(s.var.uid)
      # Replace re-evaluation of the reduced expression at arg with best_var
      replaced = False
      for rec in active:
        if _expr_matches_indexed(val, rec["val_expr"], rec["idx_uid"], rec["aliases"]):
          best = rec["best_var"]
          if s.var.type == best.type or (s.var.type.kind == best.type.kind and s.var.type.bits == best.type.bits):
            out.append(Assign(s.var, cast(best, s.var.type)))
            changed = True
            replaced = True
            break
      invalidate_for_stmt(s)
      if not replaced:
        out.append(s)
      continue

    invalidate_for_stmt(s)
    out.append(s)
  return out, changed


def normalize_block(block: list[ir.Stmt], facts: IndexFacts) -> tuple[list[ir.Stmt], bool]:
  """Recursively normalize every operator in ``block``."""
  changed = False
  block, reused = reuse_extrema_values(block)
  changed |= reused
  local = facts.fork()
  out: list[ir.Stmt] = []
  for s in block:
    local.record_stmt(s)
    for b in ir.sub_blocks(s):
      b[:], inner = normalize_block(b, local)
      changed |= inner
    if isinstance(s, IndexMap):
      squeezed = squeeze_unit_map(s)
      if squeezed is not None:
        s = squeezed
        changed = True
      split_out = split_invariant_map(s)
      if split_out is not None:
        for t in split_out:
          if isinstance(t, (IndexMap, IndexReduce, IndexFilter)):
            hoisted, h = hoist_checks_from_op(t, local)
            out.extend(hoisted)
            changed |= h
          else:
            out.append(t)
        changed = True
        continue
    if isinstance(s, (IndexMap, IndexReduce, IndexFilter, IndexScatter)):
      hoisted, h = hoist_checks_from_op(s, local)
      out.extend(hoisted)
      changed |= h
      continue
    out.append(s)
  return out, changed


def hoist_invariant_checks(fn: ir.Function) -> None:
  """Lift loop-invariant ``Check`` statements out of operator bodies across ``fn``."""
  facts = IndexFacts.from_function(fn)

  def walk(block: list[ir.Stmt], f: IndexFacts) -> list[ir.Stmt]:
    loc = f.fork()
    res: list[ir.Stmt] = []
    for s in block:
      loc.record_stmt(s)
      for b in ir.sub_blocks(s):
        b[:] = walk(b, loc)
      if isinstance(s, (IndexMap, IndexReduce, IndexFilter)):
        hoisted, _ = hoist_checks_from_op(s, loc)
        res.extend(hoisted)
      else:
        res.append(s)
    return res

  fn.body[:] = walk(fn.body, facts)


def normalize(fn: ir.Function, options) -> None:
  """Pre-fusion canonicalization: hoist invariant checks, squeeze unit map axes,
  and split reductions out of invariant map axes."""
  if not options.fusion:
    return
  for _ in range(3):
    fn.body[:], changed = normalize_block(fn.body, IndexFacts.from_function(fn))
    if not changed:
      break


__all__ = [
  "hoist_checks_from_op",
  "hoist_invariant_checks",
  "normalize",
  "normalize_block",
  "split_invariant_map",
  "squeeze_unit_map",
]
