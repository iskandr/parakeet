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
from ..syntax.ir import Assign, Check, Const, Expr, If, IndexFilter, IndexMap, IndexReduce, Scope, Var
from ..syntax.ops import ZERO, cmp, iprod
from .analysis import IndexFacts, is_pure_expr


def hoist_checks_from_op(s: IndexMap | IndexReduce | IndexFilter, facts: IndexFacts) -> tuple[list[ir.Stmt], bool]:
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


def normalize_block(block: list[ir.Stmt], facts: IndexFacts) -> tuple[list[ir.Stmt], bool]:
  """Recursively normalize every operator in ``block``."""
  changed = False
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
    if isinstance(s, (IndexMap, IndexReduce, IndexFilter)):
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
