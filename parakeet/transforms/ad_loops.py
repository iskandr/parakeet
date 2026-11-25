"""Wave 1 AD upgrades: stop_gradient, custom_vjp, checkpoint, and IR reverse loops.

Removes Parakeet's straight-line / ``MAX_UNROLL = 4096`` AD cliff by emitting
native IR-level checkpointed forward and reverse loops directly into the
operator IR:

* :func:`stop_gradient_val` — zero-copy gradient barrier on scalars, arrays,
  tuples, ``NamedTuple`` / frozen ``dataclass`` records, and ``RaggedArray``.
* :func:`custom_vjp_call` — user-defined ``(fwd, bwd)`` reverse-mode rules with
  full pytree/record residual & cotangent support and higher-order AD support.
* :func:`checkpoint_call` — activation rematerialization (``par.checkpoint``)
  that drops forward intermediates and composes with Law 2 algebraic VJPs.
* :func:`differentiate_for_loop` — 2-level ``O(sqrt(T))`` checkpointed IR
  reverse ``ForRange`` for runtime-bound and large static ``for`` loops.
* :func:`differentiate_while_loop` — fixed-slot stride-doubling ``O(log T)``
  checkpointed IR reverse ``While`` for data-dependent ``while`` loops.
* :func:`checkpointed_fold` — ``O(sqrt(N))`` checkpointed reverse mode for
  general ``par.reduce(f, x)`` folds (replacing ``O(N)`` full scan storage).
* :func:`differentiate_scan` — reverse-mode AD for ``par.scan`` (parallel
  reverse scan for additive scans, checkpointed reverse scan for general
  combiners).
"""

from __future__ import annotations

import ast
from typing import Any

from ..errors import TypeInferenceError, UnsupportedError
from ..ndtypes import Float64
from ..ragged import RaggedArray
from ..syntax import ops
from ..syntax.ir import (
  Assign,
  Break,
  Const,
  Expr,
  ForRange,
  If,
  Incref,
  Load,
  ReleaseScope,
  Store,
  Var,
  While,
)
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, idiv, imax, imin, imod, imul, isub
from ..syntax.records import record_type
from ..syntax.values import ArrayVal, RangeVal, describe
from .derivatives import is_float

# Static for-loops up to this length keep fast straight-line unrolling (zero
# checkpoint overhead, full cross-iteration fusion). Larger static loops and
# all runtime-bound loops use IR-level O(sqrt(T)) checkpointed reverse loops.
UNROLL_THRESHOLD = 32

# Fixed checkpoint slot budget for data-dependent while loops (stride doubles
# on overflow -> O(1) slots during forward pass, zero dynamic reallocations).
WHILE_CHECKPOINT_SLOTS = 64


# ---------------------------------------------------------------------------
# Tree utilities (scalars, arrays, tuples, records, RaggedArray)
# ---------------------------------------------------------------------------


def flatten_tree(v) -> tuple[list[Any], Any]:
  """Flatten a nested value into ``(leaves, spec)``."""
  if isinstance(v, RaggedArray):
    v_leaves, v_spec = flatten_tree(v.values)
    o_leaves, o_spec = flatten_tree(v.offsets)
    return v_leaves + o_leaves, ("ragged", len(v_leaves), v_spec, o_spec)
  if record := record_type(v):
    leaves, spec = flatten_tree(record.values(v))
    return leaves, ("rec", record, spec)
  if isinstance(v, tuple):
    all_leaves: list[Any] = []
    sub_specs = []
    counts = []
    for x in v:
      ls, sp = flatten_tree(x)
      all_leaves.extend(ls)
      sub_specs.append(sp)
      counts.append(len(ls))
    return all_leaves, ("tuple", tuple(counts), tuple(sub_specs))
  return [v], ("leaf",)


def unflatten_tree(leaves: list[Any], spec: Any):
  """Reconstruct a nested value from ``(leaves, spec)``."""
  tag = spec[0]
  if tag == "leaf":
    return leaves[0]
  if tag == "tuple":
    _, counts, sub_specs = spec
    out = []
    pos = 0
    for c, sp in zip(counts, sub_specs):
      out.append(unflatten_tree(leaves[pos : pos + c], sp))
      pos += c
    return tuple(out)
  if tag == "rec":
    _, record, sub_spec = spec
    return record.build(unflatten_tree(leaves, sub_spec))
  if tag == "ragged":
    _, n_val, v_spec, o_spec = spec
    return RaggedArray(unflatten_tree(leaves[:n_val], v_spec), unflatten_tree(leaves[n_val:], o_spec))
  raise AssertionError(f"unknown tree spec {tag!r}")


def _is_float_leaf(v) -> bool:
  if isinstance(v, Expr):
    return is_float(v)
  if isinstance(v, ArrayVal):
    return v.elt.kind == "f"
  return False


def zero_like_leaf(el, v):
  if isinstance(v, Expr):
    if is_float(v):
      return Const(0.0, v.type.strong)
    return v
  if isinstance(v, ArrayVal):
    if v.elt.kind == "f":
      return el.alloc_array(v.elt, v.shape, zero=True, label="zero")
    return v
  return v


def zero_like_tree(el, v):
  leaves, spec = flatten_tree(v)
  return unflatten_tree([zero_like_leaf(el, x) for x in leaves], spec)


def _register_output_tree(el, tape, node, out) -> tuple[list[Any], Any]:
  """Register every float leaf of ``out`` on ``tape`` and return ``(holders, spec)``."""
  leaves, spec = flatten_tree(out)
  holders: list[Any] = []
  new_leaves: list[Any] = []
  for leaf in leaves:
    if isinstance(leaf, Expr) and is_float(leaf):
      if not isinstance(leaf, Var):
        v = Var("out", leaf.type.strong)
        el.block.append(Assign(v, cast(leaf, v.type)))
        leaf = v
      h = tape.scalars[leaf.uid] = _scalar_adj(tape, leaf, node)
      el.ad_vars.add(leaf.uid)
      el.tainted_vars.pop(leaf.uid, None)
      holders.append(h)
      new_leaves.append(leaf)
    elif isinstance(leaf, ArrayVal) and leaf.elt.kind == "f":
      if leaf.buf is None or not leaf.contig or tape.find_root(leaf.buf.uid) is not None:
        tape.muted += 1
        try:
          leaf = el.materialize(leaf, "out")
        finally:
          tape.muted -= 1
      h = tape.root(el, leaf, node)
      holders.append(h)
      new_leaves.append(leaf)
    else:
      holders.append(None)
      new_leaves.append(leaf)
  return holders, spec, unflatten_tree(new_leaves, spec)


def _scalar_adj(tape, var: Var, node):
  from .autodiff import ScalarAdj

  return ScalarAdj(tape, var, node)


def _collect_cotangent_tree(el, holders: list[Any], out_leaves: list[Any], spec: Any):
  """Collect cotangents for a multi-output node; returns ``(any_live, g_tree)``."""
  g_leaves: list[Any] = []
  any_live = False
  for h, leaf in zip(holders, out_leaves):
    if h is None:
      g_leaves.append(leaf)
      continue
    tot = h.total(el)
    if tot is not None:
      any_live = True
      if isinstance(leaf, ArrayVal):
        tot = el.as_concrete(el.cast_array(tot, leaf.elt))
      else:
        tot = cast(tot, leaf.type.strong)
      g_leaves.append(tot)
    else:
      g_leaves.append(zero_like_leaf(el, leaf))
  return any_live, unflatten_tree(g_leaves, spec)


def _add_cotangent_tree(tape, el, target, g) -> None:
  """Recursively add cotangent tree ``g`` into primal tree ``target`` on ``tape``."""
  t_leaves, _ = flatten_tree(target)
  g_leaves, _ = flatten_tree(g)
  if len(t_leaves) != len(g_leaves):
    raise el.error(
      TypeInferenceError,
      f"custom VJP cotangent structure mismatch: expected {len(t_leaves)} leaves, got {len(g_leaves)}",
    )
  for t_leaf, g_leaf in zip(t_leaves, g_leaves):
    if g_leaf is None or not _is_float_leaf(t_leaf):
      continue
    tape.add(el, t_leaf, g_leaf)


# ---------------------------------------------------------------------------
# 1. stop_gradient
# ---------------------------------------------------------------------------


def stop_gradient_val(el, v):
  """Zero-copy gradient barrier: detach ``v`` from active reverse-mode tape and forward-mode JVP."""
  tape = el.tape
  jvp = el.jvp
  if tape is None and (jvp is None or jvp.muted):
    return v
  if isinstance(v, RaggedArray):
    return RaggedArray(stop_gradient_val(el, v.values), v.offsets)
  if record := record_type(v):
    return record.build(stop_gradient_val(el, record.values(v)))
  if isinstance(v, tuple):
    return tuple(stop_gradient_val(el, x) for x in v)
  if isinstance(v, Expr):
    if not is_float(v):
      return v
    sg = Var("sg", v.type.strong)
    # Emit directly to block without note_def / observe / jvp.scalar_t so derivative is zero.
    el.block.append(Assign(sg, cast(v, sg.type)))
    return sg
  if isinstance(v, ArrayVal):
    if v.elt.kind != "f":
      return v
    if tape is not None:
      tape.muted += 1
    if jvp is not None:
      jvp.muted += 1
    try:
      c = el.as_concrete(v)
    finally:
      if tape is not None:
        tape.muted -= 1
      if jvp is not None:
        jvp.muted -= 1
    sg_buf = Var("sg", c.buf.type)
    el.block.append(Incref(c.buf))
    el.block.append(Assign(sg_buf, c.buf))
    el.scope.owned.append(sg_buf)
    return ArrayVal(
      c.elt,
      c.shape,
      buf=sg_buf,
      offset=c.offset,
      strides=c.strides,
      roots=el.get_roots(c),
      contig=c.contig,
    )
  return v


# ---------------------------------------------------------------------------
# 2. @par.custom_vjp
# ---------------------------------------------------------------------------


def custom_vjp_call(el, fn, fwd, bwd, args: list, kwargs: dict):
  """Elaborate a ``@par.custom_vjp`` call, recording its custom VJP when active."""
  from .autodiff import Node

  tape = el.tape
  if tape is None or tape.muted:
    return el.call(fn, args, kwargs)
  if kwargs:
    raise el.error(UnsupportedError, "custom_vjp functions under grad do not yet accept keyword arguments")
  if not any(tape.is_active(el, a) for a in args):
    return el.call(fn, args, kwargs)
  if fwd is None or bwd is None:
    raise el.error(UnsupportedError, f"custom_vjp rule for {getattr(fn, '__name__', 'function')} was not registered")

  with tape.computing(el):
    pair = el.call(fwd, args, {})
  if not isinstance(pair, tuple) or len(pair) != 2:
    raise el.error(TypeInferenceError, "custom_vjp forward rule must return a 2-tuple (output, residuals)")
  raw_out, res = pair

  node = Node("custom_vjp", tuple(args), None)
  tape.record(el, node)
  holders, out_spec, out = _register_output_tree(el, tape, node, raw_out)
  out_leaves, _ = flatten_tree(out)
  node.info = (bwd, res, holders, out_leaves, out_spec, list(args), el.capture())
  return out


def backward_custom_vjp(tape, el, node, _g) -> None:
  bwd, res, holders, out_leaves, out_spec, args, context = node.info
  any_live, g_out = _collect_cotangent_tree(el, holders, out_leaves, out_spec)
  if not any_live:
    return
  with el.restored(context):
    d_args = el.call(bwd, [res, g_out], {})
  if len(args) == 1:
    # Accept either a single cotangent or a 1-tuple (d_x,)
    if isinstance(d_args, tuple) and len(d_args) == 1 and not isinstance(args[0], tuple):
      d_args = d_args[0]
    _add_cotangent_tree(tape, el, args[0], d_args)
  else:
    if not isinstance(d_args, tuple) or len(d_args) != len(args):
      raise el.error(
        TypeInferenceError,
        f"custom_vjp backward rule must return a {len(args)}-tuple of cotangents",
      )
    for arg, d_arg in zip(args, d_args):
      if d_arg is not None:
        _add_cotangent_tree(tape, el, arg, d_arg)


# ---------------------------------------------------------------------------
# 3. par.checkpoint
# ---------------------------------------------------------------------------


def checkpoint_call(el, fn, args: list, kwargs: dict):
  """Rematerialize ``fn(*args, **kwargs)`` during reverse-mode AD."""
  from .autodiff import Node, Tape

  tape = el.tape
  if tape is None or tape.muted:
    return el.call(fn, args, kwargs)

  context = el.capture()

  def run_fn():
    with el.restored(context):
      return el.call(fn, args, kwargs)

  # Dry-run probe to detect activity and mark any outer captured adjoints.
  with el.trial():
    child = Tape(el, tape, dry=True, key=(*tape.key, id(el.node_stack[-1]) if el.node_stack else 0, "ckpt_probe"))
    el.tape = child
    try:
      probe_out = run_fn()
    finally:
      el.tape = tape
    active = child.is_active(el, probe_out)
    if active:
      el.tape = None
      try:
        child.backward(el, probe_out, zero_like_tree(el, probe_out))
      finally:
        el.tape = tape

  if not active:
    return run_fn()

  if tape.dry:
    for t in child.touched:
      if t.tape is not tape:
        tape.touched.add(t)

  tape.first_order(el, "checkpoint")
  tape.muted += 1
  try:
    raw_out = run_fn()
  finally:
    tape.muted -= 1

  node = Node("checkpoint", tuple(args), None)
  tape.record(el, node)
  holders, out_spec, out = _register_output_tree(el, tape, node, raw_out)
  out_leaves, _ = flatten_tree(out)
  node.info = (run_fn, holders, out_leaves, out_spec)
  return out


def backward_checkpoint(tape, el, node, _g) -> None:
  from .autodiff import Tape

  run_fn, holders, out_leaves, out_spec = node.info
  any_live, g_out = _collect_cotangent_tree(el, holders, out_leaves, out_spec)
  if not any_live:
    return
  tape.stale(el, node)
  child = Tape(el, tape, dry=tape.dry, key=(*tape.key, id(node.origin), "ckpt"))
  el.tape = child
  try:
    r = run_fn()
  finally:
    el.tape = None
  child.backward(el, r, g_out)
  if tape.dry:
    for t in child.touched:
      if t.tape is not tape:
        tape.touched.add(t)


# ---------------------------------------------------------------------------
# 4. Shared checkpointed loop state helpers (for / while)
# ---------------------------------------------------------------------------


def _store_leaf_slot(el, buf_arr: ArrayVal, slot: Expr, val) -> None:
  if isinstance(val, Expr):
    el.emit(Store(buf_arr.buf, slot, cast(val, buf_arr.elt)))
    jvp = el.jvp
    if jvp is not None and not jvp.muted and buf_arr.buf.uid in jvp.buf_t:
      dv = jvp.tangent_expr(el, val)
      dbuf = jvp.buf_t[buf_arr.buf.uid]
      with jvp.computing(el):
        el.emit(Store(dbuf, slot, cast(dv, buf_arr.elt) if dv is not None else Const(0.0, buf_arr.elt)))
  else:
    el.store_array(el.axis_view(buf_arr, 0, slot), val)


def _load_leaf_slot(el, buf_arr: ArrayVal, slot: Expr, is_scalar: bool):
  if is_scalar:
    return Load(buf_arr.buf, slot)
  return el.axis_view(buf_arr, 0, slot)


def _copy_state_leaves(el, targets: list[Any], vals: list[Any]) -> None:
  """Copy updated leaf values into fixed-shape loop targets (ping-pong safe)."""
  jvp = el.jvp
  snaps = []
  for tgt, v in zip(targets, vals):
    if isinstance(tgt, Var):
      sv = el.tmp(cast(el.as_scalar(v, "loop state"), tgt.type), "st")
      dsv = None
      if jvp is not None and not jvp.muted and tgt.uid in jvp.targets:
        dt = jvp.tangent_expr(el, sv)
        with jvp.computing(el):
          dsv = el.tmp(cast(dt, tgt.type) if dt is not None else Const(0.0, tgt.type), "dst")
      snaps.append((tgt, sv, dsv))
    else:
      c = el.materialize(el.cast_array(v, tgt.elt), "st")
      snaps.append((tgt, c, None))
  for tgt, s, ds in snaps:
    if isinstance(tgt, Var):
      el.emit(Assign(tgt, s))
      if ds is not None and jvp is not None and tgt.uid in jvp.targets:
        with jvp.computing(el):
          el.emit(Assign(jvp.targets[tgt.uid], ds))
    else:
      el.store_array(tgt, s)


def _prepare_carried_state(el, node: ast.AST, names: frozenset[str]):
  """Extract loop-carried variables into ``(carried_names, init_vals, leaf_specs)``.

  Returns ``(carried_names, static_vars, init_leaves, state_spec, vtypes)``.
  """
  fr = el.frame
  carried_names: list[str] = []
  static_vars: dict[str, Any] = {}
  init_vals: list[Any] = []
  vtypes: dict[str, Any] = {}
  for name in sorted(names):
    v = fr.env.get(name, None)
    if v is None or getattr(type(v), "__name__", "") == "Undefined":
      continue
    if name not in fr.env:
      continue
    hint_key = (fr.path, id(node), name)
    vt = el.join_vtypes(el.vtype(v), el.hint_get(hint_key))
    if vt is None or vt[0] not in ("s", "a", "t", "rec"):
      static_vars[name] = v
      continue
    carried_names.append(name)
    vtypes[name] = vt
    init_vals.append(_coerce_to_vtype(el, v, vt))
  init_leaves, state_spec = flatten_tree(tuple(init_vals))
  return carried_names, static_vars, init_leaves, state_spec, vtypes


def _coerce_to_vtype(el, v, vt):
  tag = vt[0]
  if tag == "s":
    return cast(el.as_scalar(v, "loop variable"), vt[1])
  if tag == "a":
    return el.cast_array(v, vt[1])
  if tag == "rec":
    return vt[1].build(_coerce_to_vtype(el, vt[1].values(v), vt[2]))
  if tag == "t":
    return tuple(_coerce_to_vtype(el, x, t) for x, t in zip(v, vt[1:]))
  return v


def _alloc_state_buffers(el, leaves: list[Any], count: Expr, label: str) -> list[ArrayVal]:
  bufs: list[ArrayVal] = []
  jvp = el.jvp
  for leaf in leaves:
    if isinstance(leaf, Expr):
      b = el.alloc_array(leaf.type.strong, (count,), zero=True, label=label)
    elif isinstance(leaf, ArrayVal):
      b = el.alloc_array(leaf.elt, (count, *leaf.shape), zero=True, label=label)
    else:
      raise el.error(UnsupportedError, f"cannot checkpoint loop variable of type {describe(leaf)}")
    if jvp is not None and not jvp.muted and b.elt.kind == "f":
      jvp.ensure_zero_buf_tangent(el, b, label=f"d_{label}")
    bufs.append(b)
  return bufs


def _make_fixed_targets(el, leaves: list[Any], label: str) -> list[Any]:
  """Allocate fixed mutable scalar vars / fixed-shape contiguous buffers for loop state."""
  targets: list[Any] = []
  jvp = el.jvp
  for i, leaf in enumerate(leaves):
    if isinstance(leaf, Expr):
      v = Var(f"{label}_{i}", leaf.type.strong)
      el.mutables.add(v.uid)
      el.emit(Assign(v, cast(leaf, v.type)))
      if jvp is not None and not jvp.muted and v.type.kind == "f":
        dv = Var(f"d_{label}_{i}", v.type)
        el.mutables.add(dv.uid)
        dl = jvp.tangent_expr(el, leaf)
        with jvp.computing(el):
          el.emit(Assign(dv, cast(dl, v.type) if dl is not None else Const(0.0, v.type)))
        jvp.targets[v.uid] = dv
      targets.append(v)
    elif isinstance(leaf, ArrayVal):
      buf = el.alloc_array(leaf.elt, leaf.shape, label=f"{label}_{i}")
      if jvp is not None and not jvp.muted and buf.elt.kind == "f":
        jvp.ensure_zero_buf_tangent(el, buf, label=f"d_{label}_{i}")
      el.store_array(buf, leaf)
      targets.append(buf)
    else:
      raise el.error(UnsupportedError, f"cannot differentiate loop variable of type {describe(leaf)}")
  return targets


def _bind_step_env(el, carried_names: list[str], state_tuple: tuple, target_node=None, target_val=None) -> dict:
  fr = el.frame
  saved = dict(fr.env)
  for name, val in zip(carried_names, state_tuple):
    fr.env[name] = val
  if target_node is not None:
    el.assign(target_node, target_val)
  return saved


def _read_step_outputs(
  el, node: ast.AST, carried_names: list[str], vtypes: dict[str, Any], static_vars: dict[str, Any]
):
  fr = el.frame
  out_vals: list[Any] = []
  for name in carried_names:
    v = fr.env[name]
    vt = vtypes[name]
    new_vt = el.join_vtypes(vt, el.vtype(v))
    if new_vt is None:
      raise el.error(
        TypeInferenceError,
        f"variable '{name}' changes type inside a differentiated loop: "
        f"{el.describe_vtype(vt)} becomes {el.describe_vtype(el.vtype(v))}",
      )
    if new_vt != vt:
      el.state.hints[(fr.path, id(node), name)] = new_vt
      el.restart("loop variable type")
    out_vals.append(_coerce_to_vtype(el, v, vt))
  for name, before in static_vars.items():
    v = fr.env.get(name)
    if v is not None and not el.static_equal(v, before):
      raise el.error(
        TypeInferenceError,
        f"variable '{name}' changes from {describe(before)} to {describe(v)} inside a loop",
      )
  return tuple(out_vals)


def _check_shape_invariance(el, init_leaves: list[Any], step_leaves: list[Any], node: ast.AST) -> None:
  for init_l, step_l in zip(init_leaves, step_leaves):
    if isinstance(init_l, ArrayVal) and isinstance(step_l, ArrayVal):
      if init_l.rank != step_l.rank:
        raise el.error(UnsupportedError, "differentiated loop variable changed array rank", node)
      for d0, d1 in zip(init_l.shape, step_l.shape):
        el.check(cmp("eq", d0, d1), "ValueError", "array shape changed inside a differentiated loop")


def _probe_and_backprop_step(
  el,
  parent_tape,
  step_fn,
  in_leaves: list[Any],
  adj_bufs: list[ArrayVal | None],
  state_spec: Any,
  idx_expr: Expr,
  dry: bool,
  key: Any,
) -> tuple[bool, set]:
  """Run one step of ``step_fn`` on a child tape and backprop ``adj_bufs`` in place."""
  from .autodiff import Tape

  child = Tape(el, parent_tape, dry=dry, key=(*parent_tape.key, key))
  child_leaves: list[Any] = []
  child_holders: list[Any] = []
  saved_tape = el.tape
  try:
    el.tape = None
    for leaf in in_leaves:
      if isinstance(leaf, Expr) and is_float(leaf):
        v = Var("ls", leaf.type.strong)
        el.block.append(Assign(v, cast(leaf, v.type)))
        child_leaves.append(v)
        child_holders.append(child.leaf_scalar(el, v))
      elif isinstance(leaf, ArrayVal) and leaf.elt.kind == "f":
        arr = el.materialize(leaf, "ls")
        child_leaves.append(arr)
        child_holders.append(child.root(el, arr))
      else:
        child_leaves.append(leaf)
        child_holders.append(None)

    in_state = unflatten_tree(child_leaves, state_spec)
    el.tape = child
    try:
      out_state = step_fn(idx_expr, in_state)
    finally:
      el.tape = None

    out_leaves, _ = flatten_tree(out_state)
    active = any(child.is_active(el, x) for x in out_leaves)

    seed_leaves: list[Any] = []
    for leaf, abuf in zip(out_leaves, adj_bufs):
      if abuf is None:
        seed_leaves.append(zero_like_leaf(el, leaf))
      elif isinstance(leaf, Expr):
        seed_leaves.append(el.tmp(Load(abuf.buf, ZERO), "g"))
      else:
        seed_leaves.append(el.materialize(abuf, "g"))
    seed_tree = unflatten_tree(seed_leaves, state_spec)

    child.backward(el, out_state, seed_tree)

    if not dry:
      for h, abuf, leaf in zip(child_holders, adj_bufs, in_leaves):
        if abuf is None:
          continue
        tot = h.total(el) if h is not None else None
        if isinstance(leaf, Expr):
          el.emit(Store(abuf.buf, ZERO, cast(tot, abuf.elt) if tot is not None else Const(0.0, abuf.elt)))
        else:
          if tot is None:
            el.store_array(abuf, Const(0.0, abuf.elt))
          else:
            el.store_array(abuf, el.cast_array(tot, abuf.elt))

    return active, set(child.touched)
  finally:
    el.tape = saved_tape


def _isqrt_ir(el, n: Expr) -> Expr:
  """Compute ``max(1, int64(sqrt(float64(max(1, n)))))`` in IR."""
  pos = imax(n, ONE)
  r = cast(ops.prim("sqrt", (cast(pos, Float64),), Float64), I)
  return el.tmp(imax(r, ONE), "stride")


def _emit_segment_backward(
  tape,
  el,
  *,
  step_fn,
  ckpt_bufs: list[ArrayVal],
  init_leaves: list[Any],
  state_spec: Any,
  total_steps: Expr,
  stride: Expr,
  n_segs: Expr,
  adj_bufs: list[ArrayVal | None],
  key: Any,
) -> None:
  """Emit reverse segment loop: restore ckpt -> replay <= stride steps -> backprop backwards."""
  win_bufs = _alloc_state_buffers(el, init_leaves, stride, "win")
  cur_targets = _make_fixed_targets(el, init_leaves, "rep")
  is_scalar = [isinstance(l, Expr) for l in init_leaves]

  def seg_body(seg_rev: Expr) -> None:
    seg = el.tmp(isub(isub(n_segs, ONE), seg_rev), "seg")
    seg_start = el.tmp(imul(seg, stride), "sstart")
    seg_len = el.tmp(imin(stride, isub(total_steps, seg_start)), "slen")

    # Restore segment start state from checkpoint
    restored = [_load_leaf_slot(el, cb, seg, sc) for cb, sc in zip(ckpt_bufs, is_scalar)]
    _copy_state_leaves(el, cur_targets, restored)

    # Short forward replay into window[0..seg_len-1]
    def replay_step(u: Expr) -> None:
      for wb, ct in zip(win_bufs, cur_targets):
        _store_leaf_slot(el, wb, u, ct)
      cur_state = unflatten_tree(list(cur_targets), state_spec)
      nxt_state = step_fn(iadd(seg_start, u), cur_state)
      nxt_leaves, _ = flatten_tree(nxt_state)
      _copy_state_leaves(el, cur_targets, nxt_leaves)

    el.for_range(ZERO, seg_len, replay_step)

    # Reverse backprop u = seg_len - 1 down to 0
    def back_step(u_rev: Expr) -> None:
      u = el.tmp(isub(isub(seg_len, ONE), u_rev), "u")
      in_leaves = [_load_leaf_slot(el, wb, u, sc) for wb, sc in zip(win_bufs, is_scalar)]
      _probe_and_backprop_step(
        el,
        tape,
        step_fn,
        in_leaves,
        adj_bufs,
        state_spec,
        iadd(seg_start, u),
        dry=tape.dry,
        key=key,
      )

    el.for_range(ZERO, seg_len, back_step)

  el.for_range(ZERO, n_segs, seg_body)


# ---------------------------------------------------------------------------
# 5. IR-level checkpointed reverse `for` loop
# ---------------------------------------------------------------------------


def differentiate_for_loop(
  el,
  node: ast.For,
  start: Expr,
  stop: Expr,
  step: Expr,
  getter,
  names: frozenset[str],
  end_checks: list,
) -> bool:
  """Differentiate a runtime-bound or large static ``for`` loop with O(sqrt(T)) checkpointing."""
  from .autodiff import Node

  tape = el.tape
  assert tape is not None
  tape.first_order(el, "a checkpointed for loop")

  if not isinstance(step, Const):
    el.check(cmp("ne", step, 0), "ValueError", "range() arg 3 must not be zero")
  total = el.tmp(el.range_length(RangeVal(start, stop, step)), "ntrip")
  carried_names, static_vars, init_leaves, state_spec, vtypes = _prepare_carried_state(el, node, names)
  context = el.capture()

  def step_fn(t_idx: Expr, state_tuple: tuple) -> tuple:
    with el.restored(context):
      loop_val = iadd(start, imul(t_idx, step))
      item = loop_val if getter is None else getter(loop_val)
      saved = _bind_step_env(el, carried_names, state_tuple, node.target, item)
      try:
        el.exec_block(node.body)
        return _read_step_outputs(el, node, carried_names, vtypes, static_vars)
      finally:
        el.frame.env.clear()
        el.frame.env.update(saved)

  # Dry-run 1 step to validate vtypes, shape invariance, and mark outer touched adjoints
  el.loop_depth += 1
  try:
    with el.trial():
      probe_adj = [
        el.alloc_array(l.type.strong, (ONE,), zero=True, label="padj")
        if isinstance(l, Expr) and is_float(l)
        else (
          el.alloc_array(l.elt, l.shape, zero=True, label="padj")
          if isinstance(l, ArrayVal) and l.elt.kind == "f"
          else None
        )
        for l in init_leaves
      ]
      probe_t = Var("t", I)
      el.nonneg.add(probe_t.uid)
      el.upper[probe_t.uid] = total
      # Validate shape invariance on a clean forward step too
      saved_tape = el.tape
      el.tape = None
      try:
        probe_out = step_fn(probe_t, unflatten_tree(init_leaves, state_spec))
        probe_out_leaves, _ = flatten_tree(probe_out)
        _check_shape_invariance(el, init_leaves, probe_out_leaves, node)
      finally:
        el.tape = saved_tape
      _, touched = _probe_and_backprop_step(
        el,
        tape,
        step_fn,
        init_leaves,
        probe_adj,
        state_spec,
        probe_t,
        dry=True,
        key=(id(node), "for_probe"),
      )
  finally:
    el.loop_depth -= 1

  if tape.dry:
    for t in touched:
      if t.tape is not tape:
        tape.touched.add(t)

  # Forward pass with O(sqrt(T)) checkpoints (muted on tape)
  stride = _isqrt_ir(el, total)
  n_segs = el.tmp(idiv(iadd(total, isub(stride, ONE)), stride), "nsegs")
  ckpt_bufs = _alloc_state_buffers(el, init_leaves, imax(n_segs, ONE), "ckpt")

  tape.muted += 1
  try:
    cur_targets = _make_fixed_targets(el, init_leaves, "st")
    t_var = Var("t", I)
    el.nonneg.add(t_var.uid)
    el.upper[t_var.uid] = total
    fwd_blk: list = []
    fwd_scope = el.new_scope("loop")
    with el.in_block(fwd_blk, fwd_scope, loop=True):
      save_blk: list = []
      save_scope = el.new_scope("if")
      with el.in_block(save_blk, save_scope):
        slot = el.tmp(idiv(t_var, stride), "slot")
        for cb, ct in zip(ckpt_bufs, cur_targets):
          _store_leaf_slot(el, cb, slot, ct)
        el.emit(ReleaseScope(save_scope))
      el.emit(If(cmp("eq", imod(t_var, stride), ZERO), save_blk, []))
      nxt_state = step_fn(t_var, unflatten_tree(list(cur_targets), state_spec))
      nxt_leaves, _ = flatten_tree(nxt_state)
      _copy_state_leaves(el, cur_targets, nxt_leaves)
      el.emit(ReleaseScope(fwd_scope))
    el.emit(ForRange(t_var, ZERO, total, ONE, fwd_blk, fwd_scope))
    # Snapshot final state leaves into clean SSA outputs while muted, then register on tape
    raw_out_leaves = []
    jvp = el.jvp
    for ct in cur_targets:
      if isinstance(ct, Var):
        v = Var("fout", ct.type)
        el.block.append(Assign(v, ct))
        if jvp is not None and not jvp.muted and ct.uid in jvp.targets:
          dv = Var("dfout", ct.type)
          with jvp.computing(el):
            el.block.append(Assign(dv, jvp.targets[ct.uid]))
          jvp.scalar_t[v.uid] = dv
        raw_out_leaves.append(v)
      else:
        raw_out_leaves.append(el.materialize(ct, "fout"))
  finally:
    tape.muted -= 1
  raw_out = unflatten_tree(raw_out_leaves, state_spec)

  node_rec = Node("rev_for", tuple(init_leaves), None)
  tape.record(el, node_rec)
  holders, out_spec, out_tree = _register_output_tree(el, tape, node_rec, raw_out)
  out_leaves, _ = flatten_tree(out_tree)
  node_rec.info = (step_fn, ckpt_bufs, init_leaves, state_spec, total, stride, n_segs, holders, out_leaves, out_spec)

  # Bind updated carried variables into current frame environment
  fr = el.frame
  for name, val in zip(carried_names, out_tree):
    fr.env[name] = val
  el.block.extend(end_checks)
  return False


def backward_rev_for(tape, el, node, _g) -> None:
  step_fn, ckpt_bufs, init_leaves, state_spec, total, stride, n_segs, holders, out_leaves, out_spec = node.info
  any_live, g_out = _collect_cotangent_tree(el, holders, out_leaves, out_spec)
  if not any_live:
    return
  tape.stale(el, node)

  g_leaves, _ = flatten_tree(g_out)
  adj_bufs: list[ArrayVal | None] = []
  jvp = el.jvp
  for leaf, gl in zip(init_leaves, g_leaves):
    if isinstance(leaf, Expr) and is_float(leaf):
      ab = el.alloc_array(leaf.type.strong, (ONE,), zero=True, label="ladj")
      if jvp is not None and not jvp.muted:
        jvp.ensure_zero_buf_tangent(el, ab, label="d_ladj")
      el.emit(Store(ab.buf, ZERO, cast(gl, ab.elt)))
      if jvp is not None and not jvp.muted and ab.buf.uid in jvp.buf_t:
        dgl = jvp.tangent_expr(el, gl)
        with jvp.computing(el):
          el.emit(Store(jvp.buf_t[ab.buf.uid], ZERO, cast(dgl, ab.elt) if dgl is not None else Const(0.0, ab.elt)))
      adj_bufs.append(ab)
    elif isinstance(leaf, ArrayVal) and leaf.elt.kind == "f":
      ab = el.alloc_array(leaf.elt, leaf.shape, label="ladj")
      if jvp is not None and not jvp.muted:
        jvp.ensure_zero_buf_tangent(el, ab, label="d_ladj")
      el.store_array(ab, el.cast_array(gl, ab.elt))
      adj_bufs.append(ab)
    else:
      adj_bufs.append(None)

  _emit_segment_backward(
    tape,
    el,
    step_fn=step_fn,
    ckpt_bufs=ckpt_bufs,
    init_leaves=init_leaves,
    state_spec=state_spec,
    total_steps=total,
    stride=stride,
    n_segs=n_segs,
    adj_bufs=adj_bufs,
    key=id(node.origin),
  )

  # Propagate surviving adjoints at t=0 back to pre-loop initial values
  for init_l, ab in zip(init_leaves, adj_bufs):
    if ab is None:
      continue
    if isinstance(init_l, Expr):
      tape.add(el, init_l, el.tmp(Load(ab.buf, ZERO), "g0"))
    else:
      tape.add(el, init_l, el.materialize(ab, "g0"))


# ---------------------------------------------------------------------------
# 6. IR-level checkpointed reverse `while` loop (stride-doubling online O(log T))
# ---------------------------------------------------------------------------


def differentiate_while_loop(el, node: ast.While, names: frozenset[str]) -> bool:
  """Differentiate a data-dependent ``while`` loop with fixed-slot stride-doubling checkpointing."""
  from .autodiff import Node

  tape = el.tape
  assert tape is not None
  tape.first_order(el, "a checkpointed while loop")

  carried_names, static_vars, init_leaves, state_spec, vtypes = _prepare_carried_state(el, node, names)
  context = el.capture()

  def cond_fn(state_tuple: tuple) -> Expr:
    with el.restored(context):
      saved = _bind_step_env(el, carried_names, state_tuple)
      try:
        return el.truth(el.eval(node.test), node.test)
      finally:
        el.frame.env.clear()
        el.frame.env.update(saved)

  def step_fn(_t_idx: Expr, state_tuple: tuple) -> tuple:
    with el.restored(context):
      saved = _bind_step_env(el, carried_names, state_tuple)
      try:
        el.exec_block(node.body)
        return _read_step_outputs(el, node, carried_names, vtypes, static_vars)
      finally:
        el.frame.env.clear()
        el.frame.env.update(saved)

  # Dry-run 1 step to validate vtypes, shape invariance, and mark outer touched adjoints
  el.loop_depth += 1
  try:
    with el.trial():
      probe_adj = [
        el.alloc_array(l.type.strong, (ONE,), zero=True, label="padj")
        if isinstance(l, Expr) and is_float(l)
        else (
          el.alloc_array(l.elt, l.shape, zero=True, label="padj")
          if isinstance(l, ArrayVal) and l.elt.kind == "f"
          else None
        )
        for l in init_leaves
      ]
      saved_tape = el.tape
      el.tape = None
      try:
        probe_out = step_fn(ZERO, unflatten_tree(init_leaves, state_spec))
        probe_out_leaves, _ = flatten_tree(probe_out)
        _check_shape_invariance(el, init_leaves, probe_out_leaves, node)
      finally:
        el.tape = saved_tape
      _, touched = _probe_and_backprop_step(
        el,
        tape,
        step_fn,
        init_leaves,
        probe_adj,
        state_spec,
        ZERO,
        dry=True,
        key=(id(node), "while_probe"),
      )
  finally:
    el.loop_depth -= 1

  if tape.dry:
    for t in touched:
      if t.tape is not tape:
        tape.touched.add(t)

  slots = Const(WHILE_CHECKPOINT_SLOTS, I)
  half_slots = Const(WHILE_CHECKPOINT_SLOTS // 2, I)
  ckpt_bufs = _alloc_state_buffers(el, init_leaves, slots, "wckpt")
  is_scalar = [isinstance(l, Expr) for l in init_leaves]

  tape.muted += 1
  try:
    cur_targets = _make_fixed_targets(el, init_leaves, "wst")
    total_var = Var("wtrip", I)
    stride_var = Var("wstride", I)
    nsaved_var = Var("wnsaved", I)
    for v in (total_var, stride_var, nsaved_var):
      el.mutables.add(v.uid)
    el.emit(Assign(total_var, ZERO))
    el.emit(Assign(stride_var, ONE))
    el.emit(Assign(nsaved_var, ZERO))

    wblk: list = []
    wscope = el.new_scope("loop")
    with el.in_block(wblk, wscope, loop=True):
      cond = cond_fn(unflatten_tree(list(cur_targets), state_spec))
      if not (isinstance(cond, Const) and cond.value):
        el.emit(If(ops.lnot(cond), [ReleaseScope(wscope), Break()], []))

      # Save checkpoint when total_var % stride_var == 0, thinning & doubling stride when full
      save_blk: list = []
      save_scope = el.new_scope("if")
      with el.in_block(save_blk, save_scope):
        thin_blk: list = []
        thin_scope = el.new_scope("if")
        with el.in_block(thin_blk, thin_scope):

          def copy_even(k: Expr) -> None:
            src_idx = el.tmp(imul(k, Const(2, I)), "src")
            for cb, sc in zip(ckpt_bufs, is_scalar):
              _store_leaf_slot(el, cb, k, _load_leaf_slot(el, cb, src_idx, sc))

          el.for_range(ZERO, half_slots, copy_even)
          el.emit(Assign(nsaved_var, half_slots))
          el.emit(Assign(stride_var, imul(stride_var, Const(2, I))))
          el.emit(ReleaseScope(thin_scope))
        el.emit(If(cmp("eq", nsaved_var, slots), thin_blk, []))

        # After thinning, total_var == half_slots * old_stride * 2 == half_slots * new_stride,
        # so total_var % stride_var == 0 still holds!
        for cb, ct in zip(ckpt_bufs, cur_targets):
          _store_leaf_slot(el, cb, nsaved_var, ct)
        el.emit(Assign(nsaved_var, iadd(nsaved_var, ONE)))
        el.emit(ReleaseScope(save_scope))

      el.emit(If(cmp("eq", imod(total_var, stride_var), ZERO), save_blk, []))

      nxt_state = step_fn(total_var, unflatten_tree(list(cur_targets), state_spec))
      nxt_leaves, _ = flatten_tree(nxt_state)
      _copy_state_leaves(el, cur_targets, nxt_leaves)
      el.emit(Assign(total_var, iadd(total_var, ONE)))
      el.emit(ReleaseScope(wscope))
    el.emit(While(wblk, wscope))
    raw_out_leaves = []
    for ct in cur_targets:
      if isinstance(ct, Var):
        v = Var("wout", ct.type)
        el.block.append(Assign(v, ct))
        raw_out_leaves.append(v)
      else:
        raw_out_leaves.append(el.materialize(ct, "wout"))
  finally:
    tape.muted -= 1

  total_val = el.tmp(total_var, "ntrip")
  stride_val = el.tmp(stride_var, "stride")
  nsaved_val = el.tmp(nsaved_var, "nsegs")
  raw_out = unflatten_tree(raw_out_leaves, state_spec)

  node_rec = Node("rev_while", tuple(init_leaves), None)
  tape.record(el, node_rec)
  holders, out_spec, out_tree = _register_output_tree(el, tape, node_rec, raw_out)
  out_leaves, _ = flatten_tree(out_tree)
  node_rec.info = (
    step_fn,
    ckpt_bufs,
    init_leaves,
    state_spec,
    total_val,
    stride_val,
    nsaved_val,
    holders,
    out_leaves,
    out_spec,
  )

  fr = el.frame
  for name, val in zip(carried_names, out_tree):
    fr.env[name] = val
  return False


# ---------------------------------------------------------------------------
# 7. Checkpointed O(sqrt(N)) general fold & scan AD
# ---------------------------------------------------------------------------


def checkpointed_fold(tape, el, combine, x: ArrayVal, init, what: str):
  """Checkpointed O(sqrt(N)) reverse-mode AD for ``par.reduce(combine, x, init=...)``."""
  from .autodiff import Node, Tape

  tape.first_order(el, "a fold with a function of your own")
  if x.rank != 1:
    x = el.flat_view(x)
  if x.buf is None and x.cost > 24:
    x = el.as_concrete(x)
  n = el.tmp(x.shape[0], "n")
  context = el.capture()

  def step(acc, xi):
    with el.restored(context):
      return el.as_scalar(combine(acc, xi), "reduction result")

  tape.muted += 1
  try:
    if init is None:
      el.check(cmp("gt", n, 0), "ValueError", f"zero-size array to {what}, which has no identity")
    # Probe accumulator type on one step
    with el.trial():
      seed_t = init.type.strong if init is not None else x.elt
      probe_acc = Var("acc", seed_t)
      probe_r = step(probe_acc, el.tmp(el.elem(x, (ZERO,)), "x"))
      acc_t = probe_r.type.strong

    start_idx = ZERO if init is not None else ONE
    steps = el.tmp(imax(isub(n, start_idx), ZERO), "fsteps")
    stride = _isqrt_ir(el, steps)
    n_segs = el.tmp(idiv(iadd(steps, isub(stride, ONE)), stride), "fnsegs")
    ckpt = el.alloc_array(acc_t, (imax(n_segs, ONE),), zero=True, label="fckpt")

    acc_var = Var("acc", acc_t)
    el.mutables.add(acc_var.uid)
    if init is not None:
      el.emit(Assign(acc_var, cast(init, acc_t)))
    else:
      el.emit(Assign(acc_var, cast(el.elem(x, (ZERO,)), acc_t)))

    def fwd_step(k: Expr) -> None:
      save_blk = [Store(ckpt.buf, idiv(k, stride), acc_var)]
      el.emit(If(cmp("eq", imod(k, stride), ZERO), save_blk, []))
      xi = el.tmp(el.elem(x, (iadd(start_idx, k),)), "x")
      el.emit(Assign(acc_var, cast(step(acc_var, xi), acc_t)))

    el.for_range(ZERO, steps, fwd_step)
    out = Var("red", acc_t)
    el.block.append(Assign(out, acc_var))
  finally:
    tape.muted -= 1

  # Dry-run probe to mark any outer touched adjoints
  probe = Var("i", I)
  el.nonneg.add(probe.uid)
  el.upper[probe.uid] = n
  el.loop_depth += 1
  try:
    with el.trial():
      child = Tape(el, tape, dry=True, key=(*tape.key, "fold_ckpt_probe"))
      a_var = Var("acc", acc_t)
      el.block.append(Assign(a_var, Const(0.0, acc_t)))
      child.leaf_scalar(el, a_var)
      el.tape = child
      try:
        r = step(a_var, el.tmp(el.elem(x, (probe,)), "x"))
      finally:
        el.tape = tape
      el.tape = None
      try:
        child.backward(el, r, Const(1.0, acc_t))
      finally:
        el.tape = tape
  finally:
    el.loop_depth -= 1

  node = Node("fold_ckpt", (x,), (ckpt, init, step, n, start_idx, steps, stride, n_segs, acc_t))
  tape.record(el, node)
  tape.register(el, node, out)
  return out


def backward_fold_ckpt(tape, el, node, g: Expr) -> None:
  from .autodiff import Tape

  (x,) = node.inputs
  ckpt, init, step, n, start_idx, steps, stride, n_segs, acc_t = node.info
  tape.stale(el, node)

  cell = el.alloc_array(acc_t, (ONE,), zero=True, label="adj")
  el.emit(Store(cell.buf, ZERO, cast(g, acc_t)))
  win = el.alloc_array(acc_t, (stride,), zero=True, label="fwin")
  cur_var = Var("fcur", acc_t)
  el.mutables.add(cur_var.uid)

  def seg_body(seg_rev: Expr) -> None:
    seg = el.tmp(isub(isub(n_segs, ONE), seg_rev), "seg")
    seg_start = el.tmp(imul(seg, stride), "sstart")
    seg_len = el.tmp(imin(stride, isub(steps, seg_start)), "slen")
    el.emit(Assign(cur_var, Load(ckpt.buf, seg)))

    def replay(u: Expr) -> None:
      el.emit(Store(win.buf, u, cur_var))
      idx = iadd(start_idx, iadd(seg_start, u))
      xi = el.tmp(el.elem(x, (idx,)), "x")
      el.emit(Assign(cur_var, cast(step(cur_var, xi), acc_t)))

    el.for_range(ZERO, seg_len, replay)

    def back(u_rev: Expr) -> None:
      u = el.tmp(isub(isub(seg_len, ONE), u_rev), "u")
      idx = iadd(start_idx, iadd(seg_start, u))
      acc = Var("acc", acc_t)
      el.block.append(Assign(acc, Load(win.buf, u)))
      child = Tape(el, tape, dry=tape.dry, key=(*tape.key, id(node.origin)))
      leaf = child.leaf_scalar(el, acc)
      el.tape = child
      try:
        r = step(acc, el.tmp(el.elem(x, (idx,)), "x"))
      finally:
        el.tape = None
      child.backward(el, r, el.tmp(Load(cell.buf, ZERO), "g"))
      tot = leaf.total(el)
      el.emit(Store(cell.buf, ZERO, cast(tot, acc_t) if tot is not None else Const(0.0, acc_t)))

    el.for_range(ZERO, seg_len, back)

  el.for_range(ZERO, n_segs, seg_body)

  rest = el.tmp(Load(cell.buf, ZERO), "g")
  if init is None:
    tape.add(el, el.tmp(el.elem(x, (ZERO,)), "x"), rest)
  else:
    tape.add(el, init, rest)


def differentiate_scan(tape, el, combine, x: ArrayVal, init, what: str) -> ArrayVal:
  """Checkpointed O(sqrt(N)) reverse-mode AD for 1-D ``par.scan(combine, x, init=...)``."""
  from .autodiff import Node, Tape

  tape.first_order(el, "a scan with a custom combiner")
  if x.rank != 1:
    x = el.flat_view(x)
  if x.buf is None and x.cost > 24:
    x = el.as_concrete(x)
  n = el.tmp(x.shape[0], "n")
  context = el.capture()

  def step(acc, xi):
    with el.restored(context):
      return el.as_scalar(combine(acc, xi), "scan result")

  tape.muted += 1
  try:
    with el.trial():
      seed_t = init.type.strong if init is not None else x.elt
      probe_acc = Var("acc", seed_t)
      probe_r = step(probe_acc, el.tmp(el.elem(x, (ZERO,)), "x"))
      acc_t = probe_r.type.strong

    out = el.alloc_array(acc_t, (n,), label="scan_out")
    acc_var = Var("sacc", acc_t)
    el.mutables.add(acc_var.uid)
    if init is not None:
      el.emit(Assign(acc_var, cast(init, acc_t)))

      def fwd_init(i: Expr) -> None:
        xi = el.tmp(el.elem(x, (i,)), "x")
        el.emit(Assign(acc_var, cast(step(acc_var, xi), acc_t)))
        el.emit(Store(out.buf, i, acc_var))

      el.for_range(ZERO, n, fwd_init)
    else:
      body_pos: list = []
      sc_pos = el.new_scope("if")
      with el.in_block(body_pos, sc_pos):
        el.emit(Assign(acc_var, cast(el.elem(x, (ZERO,)), acc_t)))
        el.emit(Store(out.buf, ZERO, acc_var))

        def fwd_noinit(i: Expr) -> None:
          xi = el.tmp(el.elem(x, (i,)), "x")
          el.emit(Assign(acc_var, cast(step(acc_var, xi), acc_t)))
          el.emit(Store(out.buf, i, acc_var))

        el.for_range(ONE, n, fwd_noinit)
        el.emit(ReleaseScope(sc_pos))
      el.emit(If(cmp("gt", n, ZERO), body_pos, []))
  finally:
    tape.muted -= 1

  # Dry-run probe
  probe = Var("i", I)
  el.nonneg.add(probe.uid)
  el.upper[probe.uid] = n
  el.loop_depth += 1
  try:
    with el.trial():
      child = Tape(el, tape, dry=True, key=(*tape.key, "scan_probe"))
      a_var = Var("acc", acc_t)
      el.block.append(Assign(a_var, Const(0.0, acc_t)))
      child.leaf_scalar(el, a_var)
      el.tape = child
      try:
        r = step(a_var, el.tmp(el.elem(x, (probe,)), "x"))
      finally:
        el.tape = tape
      el.tape = None
      try:
        child.backward(el, r, Const(1.0, acc_t))
      finally:
        el.tape = tape
  finally:
    el.loop_depth -= 1

  node = Node("scan_ad", (x,), (out, init, step, n, acc_t))
  tape.record(el, node)
  tape.register(el, node, out)
  return out


def backward_scan_ad(tape, el, node, g: ArrayVal) -> None:
  from .autodiff import Tape

  (x,) = node.inputs
  out, init, step, n, acc_t = node.info
  tape.stale(el, node)
  g = el.as_concrete(g)

  # Each output out[i] IS the state after step i, so prev state is out[i-1] (or init at i=0).
  carry = el.alloc_array(acc_t, (ONE,), zero=True, label="sadj")
  lower = ZERO if init is not None else ONE

  def bstep(i: Expr) -> None:
    prev = Load(out.buf, imax(isub(i, ONE), ZERO))
    if init is not None:
      prev = ops.select(cmp("gt", i, ZERO), prev, cast(init, acc_t))
    acc = Var("acc", acc_t)
    el.block.append(Assign(acc, prev))
    gi = el.tmp(ops.prim("add", (cast(el.elem(g, (i,)), acc_t), Load(carry.buf, ZERO)), acc_t), "g")
    child = Tape(el, tape, dry=tape.dry, key=(*tape.key, id(node.origin)))
    leaf = child.leaf_scalar(el, acc)
    el.tape = child
    try:
      r = step(acc, el.tmp(el.elem(x, (i,)), "x"))
    finally:
      el.tape = None
    child.backward(el, r, gi)
    tot = leaf.total(el)
    el.emit(Store(carry.buf, ZERO, cast(tot, acc_t) if tot is not None else Const(0.0, acc_t)))

  el.for_range(isub(n, ONE), isub(lower, ONE), bstep, step=Const(-1, I))
  if init is None:
    # At i = 0, out[0] = x[0], so d_x[0] += g[0] + carry
    body_pos: list = []
    sc_pos = el.new_scope("if")
    with el.in_block(body_pos, sc_pos):
      g0 = el.tmp(ops.prim("add", (cast(el.elem(g, (ZERO,)), acc_t), Load(carry.buf, ZERO)), acc_t), "g0")
      tape.add(el, el.tmp(el.elem(x, (ZERO,)), "x0"), g0)
      el.emit(ReleaseScope(sc_pos))
    el.emit(If(cmp("gt", n, ZERO), body_pos, []))
  else:
    tape.add(el, init, el.tmp(Load(carry.buf, ZERO), "gi"))
