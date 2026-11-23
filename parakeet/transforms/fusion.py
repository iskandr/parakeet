"""Fusion rules that are passes over the operator statements.

*vertical fusion* (thesis Fig. 6.1, at the operator level)::

    Map -> Map       epilogue fusion (e.g. ``np.tanh(a @ b + c)`` into a tile)
    Map -> Filter    predicate fusion (no intermediate boolean mask array)
    Scan -> Map      ``Map(g, Scan(f, +, e, X)) => Scan(f, +, g o e, X)``
    Map -> Reduce    producer map fused directly into reduction

*filter fusion* (thesis Fig. 6.1)::

    Filter -> Filter   ``Filter(q, Filter(p, X)) => Filter(p and q, X)``
    Filter -> Reduce   ``Reduce(g, +, Filter(f, p, X)) => FilterReduce``

*sibling fusion* (thesis Fig. 6.3, horizontal fusion)::

    Reduce + Reduce    adjacent reductions over the same index space share one loop
    Map + Map          adjacent independent maps over the same shape share one loop
"""

from __future__ import annotations

from ..ndtypes import Bool
from ..syntax import ir
from ..syntax.clone import Cloner, defined
from ..syntax.ir import Assign, Const, Expr, IndexFilter, IndexMap, IndexReduce, IndexScan, Var
from .lowering import outer_expr, pure_defs
from .simplify import drop_buffers, has_effects


def resolved(e: Expr, defs: dict[int, Expr], budget: int = 20) -> Expr:
  """``e`` with single-assignment integer variables replaced by their definitions."""
  if isinstance(e, Var):
    d = defs.get(e.uid)
    return e if d is None or budget <= 0 else resolved(d, defs, budget - 1)
  if isinstance(e, ir.Prim):
    return ir.Prim(e.op, tuple(resolved(a, defs, budget - 1) for a in e.args), e.type)
  if isinstance(e, ir.Cast):
    return ir.Cast(resolved(e.arg, defs, budget - 1), e.type)
  return e


def key(e: Expr) -> object:
  """Structural identity of an expression."""
  if isinstance(e, Var):
    return ("v", e.uid)
  if isinstance(e, Const):
    return ("c", e.value, e.type)
  if isinstance(e, ir.Prim):
    return ("p", e.op, tuple(key(a) for a in e.args))
  if isinstance(e, ir.Cast):
    return ("t", e.type, key(e.arg))
  return ("?", id(e))


def same_shape(a: tuple[Expr, ...], b: tuple[Expr, ...], defs: dict[int, Expr]) -> bool:
  return len(a) == len(b) and all(key(resolved(x, defs)) == key(resolved(y, defs)) for x, y in zip(a, b))


def uses(stmts: list, uids: set[int]) -> bool:
  for s in ir.walk_stmts(stmts):
    for e in ir.stmt_exprs(s):
      if any(isinstance(x, Var) and x.uid in uids for x in ir.walk_expr(e)):
        return True
    if isinstance(s, ir.Store) and s.buf.uid in uids:
      return True
  return False


def pure(e: Expr) -> bool:
  return not any(isinstance(x, ir.RtCall) and not x.pure for x in ir.walk_expr(e))


def _buffer_accesses(stmts: list) -> tuple[set[int], set[int]]:
  """Return ``(reads, writes)`` buffer uids touched anywhere in ``stmts``."""
  reads: set[int] = set()
  writes: set[int] = set()
  for s in ir.walk_stmts(stmts):
    if isinstance(s, ir.Store):
      writes.add(s.buf.uid)
      for e in (s.index, s.value):
        for x in ir.walk_expr(e):
          if isinstance(x, ir.Load):
            reads.add(x.buf.uid)
    else:
      for e in ir.stmt_exprs(s):
        for x in ir.walk_expr(e):
          if isinstance(x, ir.Load):
            reads.add(x.buf.uid)
          elif isinstance(x, Var) and isinstance(x.type, ir.BufType):
            reads.add(x.uid)
  return reads, writes


def _buffers_may_alias(first: set[int], second: set[int], local_allocs: set[int]) -> bool:
  """True if any buffer in ``first`` could alias a buffer in ``second``.

  Two distinct local ``Alloc`` buffers never alias; two non-local buffers
  (e.g. function parameters or views) conservatively may."""
  if not first or not second:
    return False
  if first & second:
    return True
  return any(u not in local_allocs for u in first) and any(u not in local_allocs for u in second)


def _function_buffer_info(fn: ir.Function) -> tuple[set[int], dict[int, int]]:
  """Return ``(local_allocs, store_counts)`` across ``fn``."""
  allocs: set[int] = set()
  stores: dict[int, int] = {}
  for s in ir.walk_stmts(fn.body):
    if isinstance(s, ir.Alloc):
      allocs.add(s.var.uid)
    elif isinstance(s, ir.Store):
      stores[s.buf.uid] = stores.get(s.buf.uid, 0) + 1
  return allocs, stores


def count_readers(fn: ir.Function) -> dict[int, int]:
  """Number of non-store reads of every variable across ``fn``."""
  out: dict[int, int] = {}

  def note(e) -> None:
    if isinstance(e, Expr):
      for x in ir.walk_expr(e):
        if isinstance(x, Var):
          out[x.uid] = out.get(x.uid, 0) + 1
    elif isinstance(e, (tuple, list)):
      for x in e:
        note(x)

  for s in ir.walk_stmts(fn.body):
    if isinstance(s, ir.Store):
      # A Store writes s.buf; only s.index and s.value read variables
      note(s.index)
      note(s.value)
    else:
      note(ir.stmt_exprs(s))
  for r in fn.results:
    if isinstance(r, Expr):
      note(r)
    else:
      for name in getattr(type(r), "__dataclass_fields__", ()):
        note(getattr(r, name))
  return out


def _count_reads_in(stmts: list, var: Var) -> int:
  """Count non-store reads of ``var`` anywhere inside ``stmts``."""
  count = 0
  for s in ir.walk_stmts(stmts):
    exprs = (s.index, s.value) if isinstance(s, ir.Store) else ir.stmt_exprs(s)
    for e in exprs:
      for x in ir.walk_expr(e):
        if x is var:
          count += 1
  return count


# ---------------------------------------------------------------------------
# vertical fusion (Map->Map, Map->Filter, Scan->Map, Map->Reduce)
# ---------------------------------------------------------------------------


def _local_int_defs(block: list, base: dict[int, Expr]) -> dict[int, Expr]:
  """Extend ``base`` with single-assignment integer definitions local to ``block``."""
  out = dict(base)
  for s in block:
    if isinstance(s, Assign) and not isinstance(s.var.type, ir.BufType) and s.var.type.kind in "iu":
      out[s.var.uid] = s.value
  return out


class _InlineBufferLoad(Cloner):
  """Clone a consumer body, replacing every ``Load(buf, idx)`` at the
  producer's store index with ``value`` (rejecting any other use of ``buf``)."""

  def __init__(self, subst: dict[int, Expr], buf: Var, expected_index_key, value: Var, defs: dict[int, Expr]):
    super().__init__(subst)
    self.buf = buf
    self.expected_index_key = expected_index_key
    self.value = value
    self.defs = defs
    self.ok = True
    self.replaced = 0

  def expr(self, e: Expr) -> Expr:
    if isinstance(e, ir.Load) and e.buf is self.buf:
      idx = self.expr(e.index)
      if key(resolved(idx, self.defs)) == self.expected_index_key:
        self.replaced += 1
        return ir.Cast(self.value, e.type) if e.type is not self.value.type else self.value
      self.ok = False
      return e
    if isinstance(e, Var) and e is self.buf:
      self.ok = False
      return e
    if isinstance(e, ir.Prim):
      return ir.Prim(e.op, tuple(self.expr(a) for a in e.args), e.type)
    if isinstance(e, ir.Cast):
      return ir.Cast(self.expr(e.arg), e.type)
    if isinstance(e, ir.Load):
      return ir.Load(e.buf, self.expr(e.index))
    if isinstance(e, ir.RtCall):
      return ir.RtCall(e.name, tuple(self.expr(a) for a in e.args), e.type, e.pure)
    return super().expr(e)


def _match_single_output_map(s: ir.Stmt, defs: dict[int, Expr], allocs: set[int], store_counts: dict[int, int]):
  """Match an ``IndexMap`` that writes a freshly allocated buffer once at the
  end of its body.

  Returns ``(prefix, buf, store_index, index_key, value)`` or ``None``."""
  if not isinstance(s, IndexMap) or s.ordered or s.scope.owned:
    return None
  body = [t for t in s.body if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
  if not body or not isinstance(body[-1], ir.Store):
    return None
  store = body[-1]
  buf = store.buf
  if buf.uid not in allocs or store_counts.get(buf.uid, 0) != 1:
    return None
  prefix = body[:-1]
  if uses(prefix, {buf.uid}):
    return None
  prefix_defs = _local_int_defs(prefix, defs)
  return prefix, buf, store.index, key(resolved(store.index, prefix_defs)), store.value


def _match_single_output_scan(s: ir.Stmt, defs: dict[int, Expr], allocs: set[int], store_counts: dict[int, int]):
  """Match an ``IndexScan`` whose ``emit`` block writes a freshly allocated
  buffer once at its end.

  Returns ``(emit_prefix, buf, index_key, value)`` or ``None``."""
  if not isinstance(s, IndexScan) or s.scope.owned:
    return None
  emit = [t for t in s.emit if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
  if not emit or not isinstance(emit[-1], ir.Store):
    return None
  store = emit[-1]
  buf = store.buf
  if buf.uid not in allocs or store_counts.get(buf.uid, 0) != 1:
    return None
  emit_prefix = emit[:-1]
  if uses(s.body + emit_prefix, {buf.uid}):
    return None
  emit_defs = _local_int_defs(s.body + emit_prefix, defs)
  return emit_prefix, buf, key(resolved(store.index, emit_defs)), store.value


def _try_fuse_vertical_pair(
  producer: ir.Stmt,
  consumer: ir.Stmt,
  readers: dict[int, int],
  defs: dict[int, Expr],
  allocs: set[int],
  store_counts: dict[int, int],
  options,
):
  """Try to vertically fuse ``producer`` into ``consumer``.

  Returns ``(fused_stmt, eliminated_buf_uid)`` or ``None``."""
  # ---- Map -> Map / Filter / Reduce -----------------------------------------
  map_match = _match_single_output_map(producer, defs, allocs, store_counts)
  if map_match is not None:
    assert isinstance(producer, IndexMap)
    prefix, buf, store_index, index_key, value = map_match
    consumer_reads = _count_reads_in([consumer], buf)
    if consumer_reads == 0 or readers.get(buf.uid, 0) != consumer_reads:
      return None
    producer_reads, _ = _buffer_accesses(prefix)
    _, consumer_writes = _buffer_accesses([consumer])
    if _buffers_may_alias(producer_reads, consumer_writes, allocs):
      return None

    if isinstance(consumer, IndexMap) and not consumer.ordered and same_shape(producer.shape, consumer.shape, defs):
      # Map -> Map (including matmul epilogues like tanh(a @ b + c))
      elem = Var("v", buf.type.elt)
      consumer_body = [t for t in consumer.body if not (isinstance(t, ir.ReleaseScope) and t.scope is consumer.scope)]
      consumer_defs = _local_int_defs(consumer_body, defs)
      subst = {v.uid: u for v, u in zip(consumer.index, producer.index)}
      rewriter = _InlineBufferLoad(subst, buf, index_key, elem, consumer_defs)
      new_consumer_body = rewriter.block(consumer_body)
      if not rewriter.ok or rewriter.replaced != consumer_reads:
        return None
      producer.body[:] = prefix + [Assign(elem, value)] + new_consumer_body + [ir.ReleaseScope(producer.scope)]
      producer.scope.owned.extend(rewriter.scope(consumer.scope).owned)
      producer.parallel = producer.parallel and consumer.parallel
      return producer, buf.uid

    if (
      isinstance(consumer, IndexFilter) and len(producer.shape) == 1 and same_shape(producer.shape, (consumer.n,), defs)
    ):
      # Map -> Filter: only inline straight-line predicates (filter runs its body twice)
      if not all(isinstance(t, (Assign, ir.Check, ir.Comment)) for t in prefix):
        return None
      elem = Var("v", buf.type.elt)
      inliner = Cloner({producer.index[0].uid: consumer.index})
      inlined_prefix = inliner.block(prefix)
      inlined_value = inliner.expr(value)
      expected_key = key(resolved(inliner.expr(store_index), _local_int_defs(inlined_prefix, defs)))
      consumer_defs = _local_int_defs(consumer.body, defs)
      rewriter = _InlineBufferLoad(
        {v.uid: v for v in defined([consumer]).values()}, buf, expected_key, elem, consumer_defs
      )
      new_consumer_body = [rewriter.stmt(t) for t in consumer.body]
      new_keep = rewriter.expr(consumer.keep)
      if not rewriter.ok or rewriter.replaced != consumer_reads:
        return None
      consumer.body[:] = inlined_prefix + [Assign(elem, inlined_value)] + new_consumer_body
      consumer.keep = new_keep
      return consumer, buf.uid

    if (
      isinstance(consumer, IndexReduce) and consumer.where is None and same_shape(producer.shape, consumer.shape, defs)
    ):
      # Map -> Reduce: keep matmul-shaped maps intact when tiling is enabled
      if options.tiling and any(isinstance(t, IndexReduce) for t in ir.walk_stmts(prefix)):
        return None
      elem = Var("v", buf.type.elt)
      inliner = Cloner({u.uid: v for u, v in zip(producer.index, consumer.index)})
      inlined_prefix = inliner.block(prefix)
      inlined_value = inliner.expr(value)
      expected_key = key(resolved(inliner.expr(store_index), _local_int_defs(inlined_prefix, defs)))
      consumer_defs = _local_int_defs(consumer.body, defs)
      rewriter = _InlineBufferLoad(
        {v.uid: v for v in defined([consumer]).values()}, buf, expected_key, elem, consumer_defs
      )
      new_consumer_body = [rewriter.stmt(t) for t in consumer.body]
      new_folds = [rewriter.fold(f) for f in consumer.folds]
      if not rewriter.ok or rewriter.replaced != consumer_reads:
        return None
      consumer.body[:] = inlined_prefix + [Assign(elem, inlined_value)] + new_consumer_body
      consumer.folds[:] = new_folds
      return consumer, buf.uid

  # ---- Scan -> Map ----------------------------------------------------------
  scan_match = _match_single_output_scan(producer, defs, allocs, store_counts)
  if scan_match is not None and isinstance(consumer, IndexMap) and not consumer.ordered and len(consumer.shape) == 1:
    assert isinstance(producer, IndexScan)
    if not same_shape((producer.n,), consumer.shape, defs):
      return None
    emit_prefix, buf, index_key, value = scan_match
    consumer_reads = _count_reads_in([consumer], buf)
    if consumer_reads == 0 or readers.get(buf.uid, 0) != consumer_reads:
      return None
    producer_reads, _ = _buffer_accesses(producer.body + emit_prefix)
    _, consumer_writes = _buffer_accesses([consumer])
    if _buffers_may_alias(producer_reads, consumer_writes, allocs):
      return None
    elem = Var("v", buf.type.elt)
    consumer_body = [t for t in consumer.body if not (isinstance(t, ir.ReleaseScope) and t.scope is consumer.scope)]
    consumer_defs = _local_int_defs(consumer_body, defs)
    rewriter = _InlineBufferLoad({consumer.index[0].uid: producer.index}, buf, index_key, elem, consumer_defs)
    new_consumer_body = rewriter.block(consumer_body)
    if not rewriter.ok or rewriter.replaced != consumer_reads:
      return None
    producer.emit[:] = emit_prefix + [Assign(elem, value)] + new_consumer_body
    producer.scope.owned.extend(rewriter.scope(consumer.scope).owned)
    return producer, buf.uid

  return None


def _flatten_nested_map(outer: IndexMap, defs: dict[int, Expr]) -> IndexMap | None:
  """``CombineNestedMaps`` (thesis Fig. 6.2): flatten ``IndexMap(S1, i -> IndexMap(S2, j -> body))``
  into ``IndexMap(S1 ++ S2, (i, j) -> body)`` when ``S2`` does not depend on ``i``."""
  if outer.scope.owned:
    return None
  body = [t for t in outer.body if not (isinstance(t, ir.ReleaseScope) and t.scope is outer.scope)]
  if not body or not isinstance(body[-1], IndexMap):
    return None
  inner = body[-1]
  prefix = body[:-1]
  if outer.ordered != inner.ordered:
    return None
  if not all(
    isinstance(t, (Assign, ir.Comment))
    and (not isinstance(t, Assign) or (not isinstance(t.var.type, ir.BufType) and pure(t.value)))
    for t in prefix
  ):
    return None
  outer_locals = set(defined(prefix)) | {v.uid for v in outer.index}
  local_defs = _local_int_defs(prefix, defs)
  lifted_shape = [outer_expr(local_defs, d, outer_locals) for d in inner.shape]
  if any(d is None for d in lifted_shape):
    return None
  inner_body = [t for t in inner.body if not (isinstance(t, ir.ReleaseScope) and t.scope is inner.scope)]
  outer.scope.owned.extend(inner.scope.owned)
  return IndexMap(
    outer.shape + tuple(lifted_shape),  # type: ignore[arg-type]
    outer.index + inner.index,
    prefix + inner_body + [ir.ReleaseScope(outer.scope)],
    outer.scope,
    outer.ordered,
    outer.parallel or inner.parallel,
    outer.loc,
  )


def _vertical_block(
  block: list,
  readers: dict[int, int],
  defs: dict[int, Expr],
  allocs: set[int],
  store_counts: dict[int, int],
  gone: set[int],
  options,
) -> tuple[list, bool]:
  """Walk ``block`` flattening nested maps and fusing adjacent producer-consumer operators."""
  changed = False
  for i, s in enumerate(block):
    for b in ir.sub_blocks(s):
      b[:], inner = _vertical_block(b, readers, defs, allocs, store_counts, gone, options)
      changed |= inner
    if isinstance(s, IndexMap):
      flat = _flatten_nested_map(s, defs)
      if flat is not None:
        block[i] = flat
        changed = True
  out: list = []
  k = 0
  while k < len(block):
    s = block[k]
    k += 1
    if not isinstance(s, (IndexMap, IndexScan)):
      out.append(s)
      continue
    # Step over hoistable statements (allocs, pure scalar assigns, comments,
    # and pure checks when s cannot raise and only writes fresh buffers)
    j = k
    between: list = []
    producer_defs = set(defined([s]))
    _, producer_writes = _buffer_accesses([s])
    producer_cannot_raise = producer_writes <= allocs and not any(
      isinstance(x, ir.Check) or (isinstance(x, IndexReduce) and any(f.empty is not None for f in x.folds))
      for x in ir.walk_stmts([s])
    )
    while j < len(block):
      t = block[j]
      if isinstance(t, ir.Comment):
        between.append(t)
      elif isinstance(t, ir.Alloc) and not uses([t], producer_defs | producer_writes):
        between.append(t)
      elif (
        producer_cannot_raise
        and isinstance(t, ir.Check)
        and pure(t.cond)
        and all(pure(a) for a in t.args)
        and not uses([t], producer_defs | producer_writes)
      ):
        between.append(t)
      elif (
        isinstance(t, Assign)
        and not isinstance(t.var.type, ir.BufType)
        and pure(t.value)
        and not uses([t], producer_defs | producer_writes)
      ):
        between.append(t)
      else:
        break
      j += 1
    nxt = block[j] if j < len(block) else None
    if nxt is not None and isinstance(nxt, (IndexMap, IndexFilter, IndexReduce)):
      res = _try_fuse_vertical_pair(s, nxt, readers, defs, allocs, store_counts, options)
      if res is not None:
        fused, dead_buf = res
        out.extend(between)
        out.append(fused)
        gone.add(dead_buf)
        k = j + 1
        changed = True
        continue
    out.append(s)
  return out, changed


def fuse_vertical(fn: ir.Function, options) -> None:
  """Vertical fusion across operator statements: Map->Map (including epilogues),
  Map->Filter, Scan->Map and Map->Reduce."""
  if not options.fusion:
    return
  for _ in range(4):
    gone: set[int] = set()
    allocs, store_counts = _function_buffer_info(fn)
    fn.body[:], changed = _vertical_block(
      fn.body, count_readers(fn), pure_defs(fn.body), allocs, store_counts, gone, options
    )
    if gone:
      drop_buffers(fn, gone, drop_allocs=True)
    if not changed:
      break


# ---------------------------------------------------------------------------
# Filter-Filter and FilterReduce
# ---------------------------------------------------------------------------


def strip(e: Expr) -> Expr:
  while isinstance(e, ir.Cast):
    e = e.arg
  return e


class Survivors(Cloner):
  """Rewrites a consumer of a filter's output into terms of the filter's own
  index: ``idx[k]`` is ``i``, and ``i`` is known to be in ``[0, n)``."""

  def __init__(self, f: IndexFilter, consumer_stmts: list, k_var: Var, defs: dict[int, Expr]):
    super().__init__({v.uid: v for v in defined(consumer_stmts).values()})
    self.f, self.k, self.defs = f, k_var, defs
    self.inrange: set[int] = {f.index.uid}  # variables equal to the filter's index
    self.ok = True

  def expr(self, e: Expr) -> Expr:
    if isinstance(e, ir.Load) and e.buf is self.f.out:
      if e.index is not self.k:
        self.ok = False
      return ir.Cast(self.f.index, e.type) if e.type is not self.f.index.type else self.f.index
    if isinstance(e, Var):
      if e is self.k or e is self.f.out:
        self.ok = False  # the position among the survivors is not known any more
      return e
    if isinstance(e, ir.Prim):
      args = tuple(self.expr(a) for a in e.args)
      if e.op == "select" and self.is_negative_test(args[0], args[2]):
        return args[2]  # wrapping a negative index: this one is not
      return ir.Prim(e.op, args, e.type)
    if isinstance(e, ir.Cast):
      return ir.Cast(self.expr(e.arg), e.type)
    if isinstance(e, ir.Load):
      return ir.Load(e.buf, self.expr(e.index))
    if isinstance(e, ir.RtCall):
      return ir.RtCall(e.name, tuple(self.expr(a) for a in e.args), e.type, e.pure)
    return e

  def known(self, e: Expr) -> bool:
    e = strip(e)
    return isinstance(e, Var) and e.uid in self.inrange

  def is_negative_test(self, c: Expr, otherwise: Expr) -> bool:
    return (
      isinstance(c, ir.Prim)
      and c.op == "lt"
      and self.known(c.args[0])
      and isinstance(c.args[1], Const)
      and c.args[1].value == 0
      and self.known(otherwise)
    )

  def proven(self, c: Expr) -> bool:
    """Is ``c`` the bounds check ``0 <= i < n`` of the filter's own index?"""
    if not (isinstance(c, ir.Prim) and c.op == "and"):
      return False
    lo, hi = c.args
    return (
      isinstance(lo, ir.Prim) and lo.op == "ge" and self.known(lo.args[0])
      and isinstance(hi, ir.Prim) and hi.op == "lt" and self.known(hi.args[0])
      and key(resolved(hi.args[1], self.defs)) == key(resolved(self.f.n, self.defs))
    )  # fmt: skip

  def body(self, block: list) -> list:
    out: list = []
    for s in block:
      t = self.stmt(s)
      if isinstance(t, Assign) and self.known(t.value):
        self.inrange.add(t.var.uid)
      if isinstance(t, ir.Check) and self.proven(t.cond):
        continue
      out.append(t)
    return out


class _CollapseFilterChain(Cloner):
  """After fusing ``first`` into ``second`` so ``second.out`` holds original
  indices in ``[0, first.n)`` directly, rewrite remaining consumers so
  ``first.out[norm(second.out[m])]`` collapses to ``second.out[m]``."""

  def __init__(self, first: IndexFilter, second: IndexFilter, rest: list, defs: dict[int, Expr]):
    super().__init__({v.uid: v for v in defined(rest).values()})
    self.first, self.second, self.defs = first, second, defs
    self.from_second: set[int] = set()  # variables holding a value loaded from second.out
    self.ok = True

  def _is_second_index(self, e: Expr) -> bool:
    e = strip(e)
    return isinstance(e, Var) and e.uid in self.from_second

  def _is_second_load(self, e: Expr) -> bool:
    e = strip(e)
    return isinstance(e, ir.Load) and e.buf is self.second.out

  def _is_negative_wrap(self, c: Expr, otherwise: Expr) -> bool:
    return (
      isinstance(c, ir.Prim)
      and c.op == "lt"
      and self._is_second_index(c.args[0])
      and isinstance(c.args[1], Const)
      and c.args[1].value == 0
      and self._is_second_index(otherwise)
    )

  def _is_first_count_check(self, c: Expr) -> bool:
    if not (isinstance(c, ir.Prim) and c.op == "and"):
      return False
    lo, hi = c.args
    return (
      isinstance(lo, ir.Prim) and lo.op == "ge" and self._is_second_index(lo.args[0])
      and isinstance(hi, ir.Prim) and hi.op == "lt" and self._is_second_index(hi.args[0])
      and key(resolved(hi.args[1], self.defs)) == key(resolved(self.first.count, self.defs))
    )  # fmt: skip

  def expr(self, e: Expr) -> Expr:
    if isinstance(e, ir.Load) and e.buf is self.first.out:
      idx = self.expr(e.index)
      if self._is_second_index(idx):
        return ir.Cast(idx, e.type) if e.type is not idx.type else idx
      self.ok = False
      return e
    if isinstance(e, Var) and (e is self.first.out or e is self.first.count):
      self.ok = False
      return e
    if isinstance(e, ir.Prim):
      if e.op == "select" and self._is_negative_wrap(e.args[0], e.args[2]):
        return self.expr(e.args[2])
      return ir.Prim(e.op, tuple(self.expr(a) for a in e.args), e.type)
    if isinstance(e, ir.Cast):
      return ir.Cast(self.expr(e.arg), e.type)
    if isinstance(e, ir.Load):
      return ir.Load(e.buf, self.expr(e.index))
    if isinstance(e, ir.RtCall):
      return ir.RtCall(e.name, tuple(self.expr(a) for a in e.args), e.type, e.pure)
    return super().expr(e)

  def rewrite_block(self, block: list) -> list:
    """Rewrite ``block`` in place of the removed intermediate filter buffer."""
    out: list = []
    for s in block:
      if isinstance(s, ir.Check) and self._is_first_count_check(s.cond):
        continue
      if isinstance(s, Assign):
        rhs = self.expr(s.value)
        if self._is_second_load(s.value) or self._is_second_index(rhs):
          self.from_second.add(s.var.uid)
        out.append(Assign(self.var(s.var), rhs))
        continue
      if isinstance(s, ir.If):
        out.append(ir.If(self.expr(s.cond), self.rewrite_block(s.then), self.rewrite_block(s.orelse)))
        continue
      if isinstance(s, IndexMap):
        out.append(
          IndexMap(
            tuple(self.expr(d) for d in s.shape),
            tuple(self.var(v) for v in s.index),
            self.rewrite_block(s.body),
            self.scope(s.scope),
            s.ordered,
            s.parallel,
            s.loc,
          )
        )
        continue
      if isinstance(s, IndexReduce):
        out.append(
          IndexReduce(
            tuple(self.expr(d) for d in s.shape),
            tuple(self.var(v) for v in s.index),
            self.rewrite_block(s.body),
            [self.fold(f) for f in s.folds],
            self.scope(s.scope),
            self.home or s.home,
            s.parallel,
            s.loc,
            self.rewrite_block(s.pre),
            None if s.where is None else self.expr(s.where),
          )
        )
        continue
      if isinstance(s, IndexFilter):
        out.append(
          IndexFilter(
            self.expr(s.n),
            self.var(s.index),
            self.rewrite_block(s.body),
            self.expr(s.keep),
            self.var(s.out),
            self.var(s.count),
            self.scope(s.scope),
            s.parallel,
            s.loc,
          )
        )
        continue
      if isinstance(s, IndexScan):
        out.append(
          IndexScan(
            self.expr(s.n),
            self.var(s.index),
            self.rewrite_block(s.body),
            self.fold(s.fold),
            self.rewrite_block(s.emit),
            self.scope(s.scope),
            s.parallel,
            s.loc,
          )
        )
        continue
      out.append(self.stmt(s))
    return out


def filter_filter(first: IndexFilter, second: IndexFilter, rest: list, readers: dict[int, int], defs: dict[int, Expr]):
  """``Filter(q, Filter(p, X)) => Filter(p and q, X)`` with short-circuiting."""
  if second.n is not first.count:
    return None
  # All reads of first.out and first.count in the whole function must be inside second + rest
  out_reads_here = _count_reads_in([second] + rest, first.out)
  count_reads_here = _count_reads_in([second] + rest, first.count)
  if readers.get(first.out.uid, 0) != out_reads_here or readers.get(first.count.uid, 0) != count_reads_here:
    return None
  # second's own body cannot depend on first.count
  if uses(second.body, {first.count.uid}) or uses([Assign(second.count, second.keep)], {first.count.uid}):
    return None
  survivors = Survivors(first, [second], second.index, defs)
  second_body = survivors.body(second.body)
  second_keep = survivors.expr(second.keep)
  if not survivors.ok:
    return None
  collapser = _CollapseFilterChain(first, second, rest, defs)
  new_rest = collapser.rewrite_block(rest)
  if not collapser.ok:
    return None
  keep_both = Var("keep", Bool)
  fused_body = list(first.body) + [
    Assign(keep_both, Const(False, Bool)),
    ir.If(first.keep, second_body + [Assign(keep_both, second_keep)], []),
  ]
  scope = survivors.scope(second.scope)
  scope.owned.extend(first.scope.owned)
  fused = IndexFilter(
    first.n,
    first.index,
    fused_body,
    keep_both,
    second.out,
    second.count,
    scope,
    first.parallel and second.parallel,
    second.loc,
  )
  return fused, new_rest


def filter_reduce(f: IndexFilter, r: IndexReduce, readers: dict[int, int], defs: dict[int, Expr]):
  """Fuse an ``IndexFilter`` followed by an ``IndexReduce`` over its survivors
  into one guarded ``IndexReduce`` with no intermediate index array."""
  from .lowering import identity

  if len(r.shape) != 1 or r.where is not None or any(g.init is None for g in r.folds):
    return None
  if r.shape[0] is not f.count:
    return None
  # f.count is only computed at the end of the fused loop, so r's body/inits cannot read it
  if uses(r.body, {f.count.uid}) or any(
    uses([Assign(g.result, g.init)], {f.count.uid}) for g in r.folds if g.init is not None
  ):
    return None
  rewriter = Survivors(f, [r], r.index[0], defs)
  body = rewriter.body(r.body)
  folds = [rewriter.fold(g) for g in r.folds]
  if not rewriter.ok:
    return None
  # nothing but this reduction may have looked at the kept indices
  inside = _count_reads_in([r], f.out)
  if readers.get(f.out.uid, 0) != inside:
    return None
  if readers.get(f.count.uid, 0) > 1:  # someone else wants to know how many survived
    a, b = Var("acc", f.count.type), Var("x", f.count.type)
    folds.append(
      ir.Fold(
        f.count, Const(1, f.count.type), a, b, [], ir.Prim("add", (a, b), f.count.type), Const(0, f.count.type), "add"
      )
    )
  parallel = f.parallel and all(
    g.op is not None and not g.combine and identity(g.op, g.a.type) is not None for g in folds
  )
  scope = rewriter.scope(r.scope)
  scope.owned.extend(f.scope.owned)
  return IndexReduce((f.n,), (f.index,), body, folds, scope, r.home, parallel, r.loc, list(f.body), f.keep)


def filter_block(block: list, readers: dict[int, int], defs: dict[int, Expr], gone: set[int]) -> tuple[list, bool]:
  """Walk ``block`` collapsing ``Filter->Filter`` and ``Filter->Reduce`` pairs."""
  changed = False
  for s in block:
    for b in ir.sub_blocks(s):
      b[:], inner = filter_block(b, readers, defs, gone)
      changed |= inner
  out: list = []
  k = 0
  while k < len(block):
    s = block[k]
    k += 1
    nxt = block[k] if k < len(block) else None
    if isinstance(s, IndexFilter) and isinstance(nxt, IndexFilter):
      res = filter_filter(s, nxt, block[k + 1 :], readers, defs)
      if res is not None:
        fused, new_rest = res
        out.append(fused)
        out.extend(new_rest)
        gone.add(s.out.uid)
        return out, True
    if isinstance(s, IndexFilter) and isinstance(nxt, IndexReduce):
      fused = filter_reduce(s, nxt, readers, defs)
      if fused is not None:
        out.append(fused)
        gone.add(s.out.uid)
        k += 1
        changed = True
        continue
    out.append(s)
  return out, changed


def fuse_filters(fn: ir.Function, options) -> None:
  """Filter-Filter and FilterReduce fusion."""
  if not options.fusion:
    return
  for _ in range(4):
    gone: set[int] = set()
    fn.body[:], changed = filter_block(fn.body, count_readers(fn), pure_defs(fn.body), gone)
    if gone:
      drop_buffers(fn, gone, drop_allocs=True)
    if not changed:
      break


# ---------------------------------------------------------------------------
# sibling fusion (horizontal fusion of reductions and maps)
# ---------------------------------------------------------------------------


def simd_op(r: IndexReduce) -> str | None:
  """The operation a backend may vectorise the fold by, if there is one."""
  op = r.folds[0].op
  return op if op in ("add", "mul") and all(f.op == op and not f.combine for f in r.folds) else None


def compatible(a: IndexReduce, b: IndexReduce, defs: dict[int, Expr]) -> bool:
  """True if two ``IndexReduce`` statements can safely share one loop."""
  if len(a.shape) != len(b.shape) or a.parallel != b.parallel or a.where is not None or b.where is not None:
    return False
  if (a.folds[0].init is None) != (b.folds[0].init is None):
    return False
  if has_effects(a.body) or has_effects(b.body):
    return False
  # do not trade a vectorised loop for a fused one that is not
  if simd_op(a) != simd_op(b):
    return False
  return same_shape(a.shape, b.shape, defs)


def absorb(a: IndexReduce, b: IndexReduce) -> None:
  """Move the folds of ``b`` into ``a``."""
  cloner = Cloner({v.uid: v for v in defined([b]).values()})  # same names...
  cloner.m.update({v.uid: i for v, i in zip(b.index, a.index)})  # ...but the index of ``a``
  a.body.extend(cloner.stmt(t) for t in b.body)
  a.folds.extend(cloner.fold(f) for f in b.folds)
  a.scope.owned.extend(cloner.scope(b.scope).owned)


def _contains_reduce(s: IndexMap) -> bool:
  """True if ``s`` wraps an ``IndexReduce`` (e.g. a matmul or row-reduction)."""
  return any(isinstance(t, IndexReduce) for t in ir.walk_stmts(s.body))


def _compatible_maps(a: IndexMap, b: IndexMap, defs: dict[int, Expr], allocs: set[int]) -> bool:
  """True if two ``IndexMap`` statements over the same shape are independent
  and safe to merge into one loop."""
  if a.ordered != b.ordered or a.parallel != b.parallel or not same_shape(a.shape, b.shape, defs):
    return False
  # Leave matmul-shaped maps untouched so tiling / interchange still recognise them
  if _contains_reduce(a) or _contains_reduce(b):
    return False
  a_reads, a_writes = _buffer_accesses(a.body)
  b_reads, b_writes = _buffer_accesses(b.body)
  if _buffers_may_alias(a_writes, b_reads | b_writes, allocs) or _buffers_may_alias(
    b_writes, a_reads | a_writes, allocs
  ):
    return False
  # Preserve exact state on exception: do not hoist a check past a write to an input array
  if any(isinstance(t, ir.Check) for t in ir.walk_stmts(b.body)) and any(u not in allocs for u in a_writes):
    return False
  return True


def _absorb_map(a: IndexMap, b: IndexMap) -> None:
  """Merge ``b``'s body into ``a`` over ``a.index``."""
  a_body = [t for t in a.body if not (isinstance(t, ir.ReleaseScope) and t.scope is a.scope)]
  b_body = [t for t in b.body if not (isinstance(t, ir.ReleaseScope) and t.scope is b.scope)]
  cloner = Cloner({v.uid: v for v in defined([b]).values()})
  cloner.m.update({v.uid: i for v, i in zip(b.index, a.index)})
  a.body[:] = a_body + [cloner.stmt(t) for t in b_body] + [ir.ReleaseScope(a.scope)]
  a.scope.owned.extend(cloner.scope(b.scope).owned)


def fuse_block(block: list, defs: dict[int, Expr], allocs: set[int]) -> list:
  """Walk ``block`` merging adjacent compatible reductions and maps."""
  for s in block:
    for b in ir.sub_blocks(s):
      b[:] = fuse_block(b, defs, allocs)
  out: list = []
  k = 0
  while k < len(block):
    s = block[k]
    k += 1
    if isinstance(s, IndexReduce):
      # ``s`` may sink past scalar definitions that do not need its result,
      # to meet the next reduction
      while True:
        results = {v.uid for f in s.folds for v in ((f.result, f.payload[0]) if f.payload is not None else (f.result,))}
        between: list = []
        j = k
        while j < len(block) and isinstance(block[j], (Assign, ir.Comment)):
          t = block[j]
          if isinstance(t, Assign) and (isinstance(t.var.type, ir.BufType) or not pure(t.value) or uses([t], results)):
            break
          between.append(t)
          j += 1
        nxt = block[j] if j < len(block) else None
        if not isinstance(nxt, IndexReduce) or uses([nxt], results) or not compatible(s, nxt, defs):
          break
        out.extend(between)
        absorb(s, nxt)
        k = j + 1
      out.append(s)
      continue
    if isinstance(s, IndexMap):
      sunk: list = []
      while True:
        _, a_writes = _buffer_accesses(s.body)
        map_outputs = set(defined([s])) | a_writes
        hoist: list = []
        sink_more: list = []
        j = k
        while j < len(block):
          t = block[j]
          if isinstance(t, ir.Comment):
            hoist.append(t)
          elif isinstance(t, ir.Incref):
            sink_more.append(t)
          elif isinstance(t, ir.Alloc) and not uses([t], map_outputs):
            hoist.append(t)
          elif (
            isinstance(t, Assign)
            and not isinstance(t.var.type, ir.BufType)
            and pure(t.value)
            and not uses([t], map_outputs)
          ):
            hoist.append(t)
          else:
            break
          j += 1
        nxt = block[j] if j < len(block) else None
        if not isinstance(nxt, IndexMap) or not _compatible_maps(s, nxt, defs, allocs):
          break
        out.extend(hoist)
        sunk.extend(sink_more)
        _absorb_map(s, nxt)
        k = j + 1
      out.append(s)
      out.extend(sunk)
      continue
    out.append(s)
  return out


def fuse_siblings(fn: ir.Function, options) -> None:
  """Sibling fusion of adjacent reductions and independent maps."""
  if not options.fusion:
    return
  allocs, _ = _function_buffer_info(fn)
  fn.body[:] = fuse_block(fn.body, pure_defs(fn.body), allocs)


__all__ = ["fuse_filters", "fuse_siblings", "fuse_vertical"]
