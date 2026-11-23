"""IR verifier for Parakeet's operator and loop representations.

Continuing Parakeet 0.x's ``verify`` pass, :func:`verify` checks that every
function (before passes, after each rewrite, and after lowering) satisfies the
invariants the middle end and backends depend on:

* every variable read is in scope (params, enclosing definitions, or the
  current operator/loop's bindings; ``ParLoop`` and operator-body locals never
  escape into sibling loops);
* every buffer operation (``Load``, ``Store``, ``Alloc``, ``Incref``,
  ``Release``) operates on a ``BufType`` with integer indices/counts;
* every loop/operator bound and index is integer-typed, and every condition is
  a scalar;
* every buffer listed in a ``Scope.owned`` is actually allocated in the
  function (catching dangling releases after fusion or dead-code removal).
"""

from __future__ import annotations

from ..errors import InternalError
from ..ndtypes import BufType, ScalarType
from ..syntax import ir
from ..syntax.ir import (
  Alloc,
  ArrayParam,
  ArrayResult,
  Assign,
  Check,
  Expr,
  ForRange,
  Function,
  If,
  Incref,
  IndexFilter,
  IndexMap,
  IndexReduce,
  IndexScan,
  Load,
  ParLoop,
  Release,
  ReleaseScope,
  Store,
  Var,
  While,
)


def _fail(fn: Function, phase: str, msg: str) -> None:
  where = f" ({phase})" if phase else ""
  raise InternalError(f"IR verification failed for {fn.name}{where}: {msg}")


def _is_int(t) -> bool:
  return isinstance(t, ScalarType) and t.kind in "iu"


def verify(fn: Function, phase: str = "") -> Function:
  """Verify structural, scoping, typing, and ownership invariants of ``fn``.

  Returns ``fn`` so it can be used inline; raises :class:`InternalError` on any
  violation."""
  # Collect every buffer that is allocated anywhere in ``fn`` (Alloc or IndexFilter.out)
  # plus buffer parameters so dangling scope.owned entries are caught immediately.
  allocated_bufs: set[int] = set()
  param_vars: set[int] = set()
  for p in fn.params:
    if isinstance(p, ArrayParam):
      param_vars.add(p.buf.uid)
      allocated_bufs.add(p.buf.uid)
      param_vars.add(p.offset.uid)
      for v in p.shape:
        param_vars.add(v.uid)
      if p.strides is not None:
        for v in p.strides:
          param_vars.add(v.uid)
    else:
      param_vars.add(p.uid)
      if isinstance(p.type, BufType):
        allocated_bufs.add(p.uid)

  for s in ir.walk_stmts(fn.body):
    if isinstance(s, Alloc):
      allocated_bufs.add(s.var.uid)
    elif isinstance(s, IndexFilter):
      allocated_bufs.add(s.out.uid)
    elif isinstance(s, Assign) and isinstance(s.var.type, BufType):
      allocated_bufs.add(s.var.uid)

  def check_scope(scope: ir.Scope) -> None:
    for v in scope.owned:
      if not isinstance(v.type, BufType):
        _fail(fn, phase, f"scope owns non-buffer variable {v.name}: {v.type}")
      if v.uid not in allocated_bufs:
        _fail(fn, phase, f"scope owns buffer {v.name} that is never allocated in {fn.name}")

  def check_expr(e: Expr, visible: set[int], context: str) -> None:
    for node in ir.walk_expr(e):
      if isinstance(node, Var):
        if node.uid not in visible:
          _fail(fn, phase, f"variable {node.name} used in {context} is not in scope")
      elif isinstance(node, Load):
        if not isinstance(node.buf.type, BufType):
          _fail(fn, phase, f"Load from non-buffer {node.buf.name}: {node.buf.type}")
        if not _is_int(node.index.type):
          _fail(fn, phase, f"Load index on {node.buf.name} has non-integer type {node.index.type}")

  def check_block(block: list[ir.Stmt], visible: set[int]) -> None:
    for s in block:
      if isinstance(s, Assign):
        check_expr(s.value, visible, f"assignment to {s.var.name}")
        visible.add(s.var.uid)
      elif isinstance(s, Store):
        if s.buf.uid not in visible:
          _fail(fn, phase, f"Store target {s.buf.name} is not in scope")
        if not isinstance(s.buf.type, BufType):
          _fail(fn, phase, f"Store into non-buffer {s.buf.name}: {s.buf.type}")
        if not _is_int(s.index.type):
          _fail(fn, phase, f"Store index on {s.buf.name} has non-integer type {s.index.type}")
        if not isinstance(s.value.type, ScalarType):
          _fail(fn, phase, f"Store value into {s.buf.name} has non-scalar type {s.value.type}")
        check_expr(s.index, visible, f"Store index on {s.buf.name}")
        check_expr(s.value, visible, f"Store value on {s.buf.name}")
      elif isinstance(s, Alloc):
        if not isinstance(s.var.type, BufType):
          _fail(fn, phase, f"Alloc target {s.var.name} has non-buffer type {s.var.type}")
        if not _is_int(s.count.type):
          _fail(fn, phase, f"Alloc count for {s.var.name} has non-integer type {s.count.type}")
        check_expr(s.count, visible, f"Alloc {s.var.name}")
        visible.add(s.var.uid)
      elif isinstance(s, Incref):
        if not isinstance(s.buf.type, BufType):
          _fail(fn, phase, f"Incref on non-buffer expression of type {s.buf.type}")
        check_expr(s.buf, visible, "Incref")
      elif isinstance(s, Release):
        # Loop-carried buffer holders (``keep_N``) are NULL-initialised at function entry
        # and released at the top of a loop before their first assignment.
        if s.var.uid not in visible and s.var.uid not in allocated_bufs:
          _fail(fn, phase, f"Release of {s.var.name} not in scope")
        if not isinstance(s.var.type, BufType):
          _fail(fn, phase, f"Release of non-buffer {s.var.name}: {s.var.type}")
      elif isinstance(s, ReleaseScope):
        check_scope(s.scope)
      elif isinstance(s, Check):
        if not isinstance(s.cond.type, ScalarType):
          _fail(fn, phase, f"Check condition has non-scalar type {s.cond.type}")
        check_expr(s.cond, visible, "Check condition")
        for arg in s.args:
          check_expr(arg, visible, "Check argument")
      elif isinstance(s, If):
        if not isinstance(s.cond.type, ScalarType):
          _fail(fn, phase, f"If condition has non-scalar type {s.cond.type}")
        check_expr(s.cond, visible, "If condition")
        then_vis = set(visible)
        else_vis = set(visible)
        check_block(s.then, then_vis)
        check_block(s.orelse, else_vis)
        # Python variables assigned in an if-branch live in the enclosing function scope
        visible.update(then_vis)
        visible.update(else_vis)
      elif isinstance(s, ForRange):
        for label, expr in (("start", s.start), ("stop", s.stop), ("step", s.step)):
          if not _is_int(expr.type):
            _fail(fn, phase, f"ForRange {label} has non-integer type {expr.type}")
          check_expr(expr, visible, f"ForRange {label}")
        if not _is_int(s.var.type):
          _fail(fn, phase, f"ForRange index {s.var.name} has non-integer type {s.var.type}")
        check_scope(s.scope)
        body_vis = set(visible)
        body_vis.add(s.var.uid)
        check_block(s.body, body_vis)
        # Loop-assigned locals (other than the loop index) remain visible in Python function scope
        visible.update(body_vis - {s.var.uid})
      elif isinstance(s, While):
        check_scope(s.scope)
        body_vis = set(visible)
        check_block(s.body, body_vis)
        visible.update(body_vis)
      elif isinstance(s, ParLoop):
        for label, expr in (("total", s.total), ("nchunks", s.nchunks)):
          if not _is_int(expr.type):
            _fail(fn, phase, f"ParLoop {label} has non-integer type {expr.type}")
          check_expr(expr, visible, f"ParLoop {label}")
        if s.work is not None:
          check_expr(s.work, visible, "ParLoop work")
        check_scope(s.scope)
        # ParLoop is outlined into its own function: body locals never leak out
        body_vis = set(visible)
        body_vis.update((s.chunk.uid, s.start.uid, s.stop.uid))
        check_block(s.body, body_vis)
      elif isinstance(s, IndexMap):
        if len(s.shape) != len(s.index):
          _fail(fn, phase, f"IndexMap rank mismatch: {len(s.shape)} dims vs {len(s.index)} indices")
        for d in s.shape:
          if not _is_int(d.type):
            _fail(fn, phase, f"IndexMap shape dim has non-integer type {d.type}")
          check_expr(d, visible, "IndexMap shape")
        for v in s.index:
          if not _is_int(v.type):
            _fail(fn, phase, f"IndexMap index {v.name} has non-integer type {v.type}")
        check_scope(s.scope)
        body_vis = set(visible)
        body_vis.update(v.uid for v in s.index)
        check_block(s.body, body_vis)
      elif isinstance(s, IndexReduce):
        if len(s.shape) != len(s.index):
          _fail(fn, phase, f"IndexReduce rank mismatch: {len(s.shape)} dims vs {len(s.index)} indices")
        for d in s.shape:
          if not _is_int(d.type):
            _fail(fn, phase, f"IndexReduce shape dim has non-integer type {d.type}")
          check_expr(d, visible, "IndexReduce shape")
        for v in s.index:
          if not _is_int(v.type):
            _fail(fn, phase, f"IndexReduce index {v.name} has non-integer type {v.type}")
        for f in s.folds:
          if f.init is not None:
            check_expr(f.init, visible, f"IndexReduce init for {f.result.name}")
        check_scope(s.scope)
        body_vis = set(visible)
        body_vis.update(v.uid for v in s.index)
        if s.pre:
          check_block(s.pre, body_vis)
        if s.where is not None:
          if not isinstance(s.where.type, ScalarType):
            _fail(fn, phase, f"IndexReduce where guard has non-scalar type {s.where.type}")
          check_expr(s.where, body_vis, "IndexReduce where")
        check_block(s.body, body_vis)
        for f in s.folds:
          check_expr(f.value, body_vis, f"IndexReduce fold value for {f.result.name}")
          comb_vis = set(body_vis)
          comb_vis.update((f.a.uid, f.b.uid))
          check_block(f.combine, comb_vis)
          check_expr(f.combined, comb_vis, f"IndexReduce combiner for {f.result.name}")
        for f in s.folds:
          visible.add(f.result.uid)
      elif isinstance(s, IndexFilter):
        if not _is_int(s.n.type):
          _fail(fn, phase, f"IndexFilter length has non-integer type {s.n.type}")
        if not _is_int(s.index.type):
          _fail(fn, phase, f"IndexFilter index {s.index.name} has non-integer type {s.index.type}")
        if not isinstance(s.out.type, BufType):
          _fail(fn, phase, f"IndexFilter output {s.out.name} has non-buffer type {s.out.type}")
        if not _is_int(s.count.type):
          _fail(fn, phase, f"IndexFilter count {s.count.name} has non-integer type {s.count.type}")
        check_expr(s.n, visible, "IndexFilter length")
        check_scope(s.scope)
        body_vis = set(visible)
        body_vis.add(s.index.uid)
        check_block(s.body, body_vis)
        if not isinstance(s.keep.type, ScalarType):
          _fail(fn, phase, f"IndexFilter predicate has non-scalar type {s.keep.type}")
        check_expr(s.keep, body_vis, "IndexFilter predicate")
        visible.add(s.out.uid)
        visible.add(s.count.uid)
      elif isinstance(s, IndexScan):
        if not _is_int(s.n.type):
          _fail(fn, phase, f"IndexScan length has non-integer type {s.n.type}")
        if not _is_int(s.index.type):
          _fail(fn, phase, f"IndexScan index {s.index.name} has non-integer type {s.index.type}")
        check_expr(s.n, visible, "IndexScan length")
        f = s.fold
        if f.init is not None:
          check_expr(f.init, visible, f"IndexScan init for {f.result.name}")
        check_scope(s.scope)
        body_vis = set(visible)
        body_vis.add(s.index.uid)
        check_block(s.body, body_vis)
        check_expr(f.value, body_vis, "IndexScan fold value")
        comb_vis = set(body_vis)
        comb_vis.update((f.a.uid, f.b.uid))
        check_block(f.combine, comb_vis)
        check_expr(f.combined, comb_vis, "IndexScan combiner")
        emit_vis = set(body_vis)
        emit_vis.add(f.result.uid)
        check_block(s.emit, emit_vis)

  check_scope(fn.scope)
  visible = set(param_vars)
  check_block(fn.body, visible)
  for r in fn.results:
    if isinstance(r, Expr):
      check_expr(r, visible, "return value")
    elif isinstance(r, ArrayResult):
      check_expr(r.buf, visible, "returned array buffer")
      check_expr(r.offset, visible, "returned array offset")
      for d in r.shape:
        check_expr(d, visible, "returned array shape")
      for st in r.strides:
        check_expr(st, visible, "returned array stride")
  return fn


__all__ = ["verify"]
