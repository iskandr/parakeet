"""The host backend: turn IR into a C module, compile it and load it.

Every other backend builds on this one - GPU backends run the host program
here and divert eligible parallel loops to a device.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field

from ..config import config
from ..ndtypes import BufType, ScalarType
from ..syntax import ir
from . import toolchain
from .cfamily import CFamilyEmitter, defined_vars, owned_vars
from .runtime_c import HEADER_SLOTS, PRELUDE


@dataclass
class KernelInfo:
  """Layout of the context struct handed to a device for one parallel loop."""

  index: int
  loop: ir.ParLoop
  free: list[ir.Var]
  fields: list[tuple[str, str, object]] = field(default_factory=list)  # (name, kind, type)
  device: object = None  # backend-specific compiled kernel


class HostEmitter(CFamilyEmitter):
  def __init__(self, fn: ir.Function, device=None):
    super().__init__()
    self.fn = fn
    self.device = device
    self.functions: list[str] = []
    self.kernels: list[KernelInfo] = []
    self.bodies: dict[tuple, int] = {}  # structural IR key of an outlined loop body -> the kernel that owns it
    self.nesting = 0  # > 0 while printing the body of an outlined loop
    self._host_sync_suppressed = False
    self.loop_info: dict[int, tuple[bool, bool]] = {}
    self.n_labels = 0
    self.label_used = False
    self.stack_bufs: dict[int, int] = {
      t.var.uid: int(t.count.value)
      for t in ir.walk_stmts(fn.body)
      if isinstance(t, ir.Alloc) and t.stack and isinstance(t.count, ir.Const)
    }

  # ---- statements ----------------------------------------------------------

  def check(self, s: ir.Check, out: list[str], pad: str) -> None:
    eid = self.error_id(s.err)
    args = [self.expr(a) for a in s.args[:4]]
    args += ["0"] * (4 - len(args))
    args = [f"(int64_t)({a})" for a in args]
    self.label_used = True
    out.append(
      f"{pad}if (PK_UNLIKELY(!({self.expr(s.cond)}))) "
      f"{{ pk_fail(rt, {eid}, {', '.join(args)}); goto {self.fail_label}; }}"
    )

  def assign(self, s: ir.Assign, out: list[str], pad: str) -> None:
    v = s.value
    if isinstance(v, ir.RtCall) and v.name == "num_chunks":
      heavy, on_device = self.loop_info.get(s.var.uid, (False, False))
      out.append(
        f"{pad}{s.var.name} = pk_num_chunks(rt, {self.expr(v.args[0])}, "
        f"{int(bool(v.args[1].value))}, {int(heavy)}, {int(on_device)});"
      )
      return
    super().assign(s, out, pad)

  def assign_buf(self, s: ir.Assign, out: list[str], pad: str) -> None:
    name = s.var.name
    if isinstance(s.value, ir.Null):
      out.append(f"{pad}{name} = NULL; {name}_d = NULL;")
    else:
      src = s.value.name  # type: ignore[attr-defined]
      out.append(f"{pad}{name} = {src}; {name}_d = {src}_d;")

  def alloc(self, s: ir.Alloc, out: list[str], pad: str) -> None:
    name = s.var.name
    ct = self.ctype(s.var.type.elt)
    if s.stack and isinstance(s.count, ir.Const):
      if s.zero:
        out.append(f"{pad}memset({name}_d, 0, sizeof({name}_d));")
      return
    self.label_used = True
    out.append(f"{pad}{name} = pk_alloc(rt, {self.expr(s.count)}, sizeof({ct}), {int(s.zero)});")
    out.append(f"{pad}if (PK_UNLIKELY(!{name})) goto {self.fail_label};")
    out.append(f"{pad}{name}_d = ({ct}*){name}->data;")

  def atomic_store(self, s: ir.Store, ct: str, value: str, out: list[str], pad: str) -> None:
    if s.atomic == "add" and s.buf.type.elt.name in ("float32", "float64", "int32", "uint32"):
      out.append(f"{pad}pk_atomic_add_{s.buf.type.elt.name}(&{self.data(s.buf)}[{self.expr(s.index)}], {value});")
      return
    super().atomic_store(s, ct, value, out, pad)

  def incref(self, s: ir.Incref, out: list[str], pad: str) -> None:
    out.append(f"{pad}pk_incref({self.expr(s.buf)});")

  def release(self, v: ir.Var, out: list[str], pad: str) -> None:
    if self.device is not None and not self.nesting:
      out.append(f"{pad}if (PK_UNLIKELY(rt->dev_live)) pk_dev_forget(rt, {v.name});")
    out.append(f"{pad}pk_release(&{v.name});")

  def _emit_host_sync(self, stmts: list[ir.Stmt], out: list[str], pad: str, local_defs: set[int]) -> bool:
    """Emit per-buffer ``PK_DEV_SYNC_READ`` / ``PK_DEV_SYNC_WRITE`` for outer
    buffers touched by ``stmts``.  Returns True if handled (or no sync needed)."""
    has_gemm = False
    rt_bufs: dict[int, ir.Var] = {}
    writes: dict[int, ir.Var] = {}
    reads: dict[int, ir.Var] = {}
    for st in stmts:
      if isinstance(st, ir.Store) and st.buf.uid not in local_defs:
        writes[st.buf.uid] = st.buf
      for e in ir.stmt_exprs(st):
        for x in ir.walk_expr(e):
          if isinstance(x, ir.Load) and x.buf.uid not in local_defs:
            reads[x.buf.uid] = x.buf
          elif isinstance(x, ir.RtCall) and not x.pure and x.args and x.name in ("gemm", "argsort"):
            has_gemm = True
            for a in x.args:
              if isinstance(a, ir.Var) and isinstance(a.type, BufType) and a.uid not in local_defs:
                rt_bufs[a.uid] = a
    if has_gemm:
      out.append(f"{pad}PK_DEV_SYNC(rt);")
      for b in {**writes, **rt_bufs}.values():
        out.append(f"{pad}{b.name}->flags &= ~PK_UNTOUCHED;")
      return True
    for uid, b in reads.items():
      if uid not in writes:
        out.append(f"{pad}PK_DEV_SYNC_READ(rt, {b.name});")
    for b in writes.values():
      out.append(f"{pad}PK_DEV_SYNC_WRITE(rt, {b.name});")
      out.append(f"{pad}{b.name}->flags &= ~PK_UNTOUCHED;")
    return True

  def stmt(self, s: ir.Stmt, out: list[str], indent: int) -> None:
    # Arrays may be on the device: host code that touches array memory itself
    # syncs only the specific buffers it reads/writes (keeping all other live
    # tensors resident on the GPU).
    if (
      self.device is not None and not self.nesting and not self._host_sync_suppressed and not isinstance(s, ir.ParLoop)
    ):
      pad = "  " * indent
      if isinstance(s, (ir.ForRange, ir.While)):
        inner = list(ir.walk_stmts([s]))
        if not any(isinstance(t, ir.ParLoop) for t in inner) and not any(
          isinstance(t, ir.Assign) and isinstance(t.var.type, BufType) for t in inner
        ):
          local_bufs = {t.var.uid for t in inner if isinstance(t, ir.Alloc)}
          self._emit_host_sync(inner, out, pad, local_bufs)
          self._host_sync_suppressed = True
          try:
            super().stmt(s, out, indent)
          finally:
            self._host_sync_suppressed = False
          return
      else:
        self._emit_host_sync([s], out, pad, set())
    super().stmt(s, out, indent)

  def loop_pragma(self, s: ir.ForRange) -> str | None:
    # a floating point fold may only be vectorised if the compiler is told
    # that reordering it is fine (integers need no permission)
    red = s.reduction
    if red is None or not toolchain.supports_simd_pragmas():
      return None
    op, acc = red
    accs = acc if isinstance(acc, (list, tuple)) else [acc]
    simple = all(
      isinstance(b, (ir.Assign, ir.Comment)) or (isinstance(b, ir.ReleaseScope) and not b.scope.owned) for b in s.body
    )
    if any(a.type.kind != "f" for a in accs) or not simple:
      return None
    return f"#pragma omp simd reduction({'+' if op == 'add' else '*'}:{', '.join(a.name for a in accs)})"

  def rtcall(self, e: ir.RtCall) -> str:
    args = ", ".join(self.expr(a) for a in e.args)
    if e.name == "rng_key":
      return "pk_rng_key(rt)"
    if e.pure:
      return f"pk_{e.name}({args})"
    return f"pk_{e.name}(rt{', ' if args else ''}{args})"

  # ---- declarations --------------------------------------------------------

  def declare(self, v: ir.Var, out: list[str], pad: str = "  ") -> None:
    t = v.type
    if isinstance(t, BufType):
      if v.uid in self.stack_bufs:
        out.append(f"{pad}{self.ctype(t.elt)} {v.name}_d[{self.stack_bufs[v.uid]}];")
      else:
        out.append(f"{pad}pk_buf* {v.name} = NULL; {self.ctype(t.elt)}* {v.name}_d = NULL;")
    elif t.kind == "b":
      out.append(f"{pad}bool {v.name} = false;")
    else:
      out.append(f"{pad}{self.ctype(t)} {v.name} = 0;")

  # ---- parallel loops ------------------------------------------------------

  def loop_signature(self, s: ir.ParLoop, defined: dict[int, ir.Var]) -> tuple[tuple, list[ir.Var]]:
    """Return ``(signature_key, free_vars)`` for structural body deduplication.

    ``signature_key`` is a hashable tuple representing ``s.body`` up to
    alpha-renaming of local and free variables (preserving scalar/buffer types,
    constants, ops, checks, and owned buffers).  ``free_vars`` lists the
    captured outer variables in their order of first appearance in ``s.body``,
    so any two ``ParLoop``s with equal ``signature_key`` share one compiled C
    context struct and ``pk_body_k`` function without re-emitting C text."""
    free: list[ir.Var] = []
    free_index: dict[int, int] = {}
    local_index: dict[int, int] = {s.chunk.uid: -1, s.start.uid: -2, s.stop.uid: -3}

    def var_key(v: ir.Var) -> tuple:
      if v.uid in defined:
        slot = local_index.get(v.uid)
        if slot is None:
          slot = local_index[v.uid] = len(local_index)
        return ("local", slot, v.type)
      slot = free_index.get(v.uid)
      if slot is None:
        slot = free_index[v.uid] = len(free)
        free.append(v)
      return ("free", slot, v.type)

    def expr_key(e: ir.Expr) -> tuple:
      if isinstance(e, ir.Var):
        return var_key(e)
      if isinstance(e, ir.Const):
        val = e.value
        if isinstance(val, float):
          # Distinguish NaN and signed zero (-0.0 vs +0.0) in float constants
          tag = ("nan",) if math.isnan(val) else (val, math.copysign(1.0, val))
          return ("const", e.type, tag)
        return ("const", e.type, type(val), val)
      if isinstance(e, ir.Prim):
        return ("prim", e.op, e.type, tuple(expr_key(a) for a in e.args))
      if isinstance(e, ir.Cast):
        return ("cast", e.type, expr_key(e.arg))
      if isinstance(e, ir.Load):
        return ("load", var_key(e.buf), expr_key(e.index))
      if isinstance(e, ir.RtCall):
        return ("rtcall", e.name, e.pure, e.type, tuple(expr_key(a) for a in e.args))
      if isinstance(e, ir.Null):
        return ("null", e.type)
      return ("other", id(e))

    def block_key(stmts: list[ir.Stmt]) -> tuple:
      items = []
      for t in stmts:
        if isinstance(t, ir.Comment):
          continue
        if isinstance(t, ir.Assign):
          chunk_hint = (
            self.loop_info.get(t.var.uid, (False, False))
            if isinstance(t.value, ir.RtCall) and t.value.name == "num_chunks"
            else None
          )
          items.append(("assign", var_key(t.var), expr_key(t.value), chunk_hint))
        elif isinstance(t, ir.Store):
          items.append(("store", var_key(t.buf), expr_key(t.index), expr_key(t.value), t.atomic))
        elif isinstance(t, ir.If):
          items.append(("if", expr_key(t.cond), block_key(t.then), block_key(t.orelse)))
        elif isinstance(t, ir.ForRange):
          reduction = t.reduction
          if reduction is not None:
            accs = reduction[1] if isinstance(reduction[1], (list, tuple)) else [reduction[1]]
            reduction = (reduction[0], tuple(var_key(a) for a in accs))
          items.append(
            (
              "for",
              var_key(t.var),
              expr_key(t.start),
              expr_key(t.stop),
              expr_key(t.step),
              t.simd,
              reduction,
              block_key(t.body),
            )
          )
        elif isinstance(t, ir.While):
          items.append(("while", block_key(t.body)))
        elif isinstance(t, ir.Break):
          items.append(("break",))
        elif isinstance(t, ir.Continue):
          items.append(("continue",))
        elif isinstance(t, ir.Alloc):
          items.append(("alloc", var_key(t.var), expr_key(t.count), t.zero, t.stack))
        elif isinstance(t, ir.Incref):
          items.append(("incref", expr_key(t.buf)))
        elif isinstance(t, ir.Release):
          items.append(("release", var_key(t.var)))
        elif isinstance(t, ir.ReleaseScope):
          if t.scope.owned:
            items.append(("release_scope", tuple(var_key(v) for v in t.scope.owned)))
        elif isinstance(t, ir.Check):
          items.append(
            ("check", (t.err.exc, t.err.fmt, t.err.loc), expr_key(t.cond), tuple(expr_key(a) for a in t.args[:4]))
          )
        elif isinstance(t, ir.ParLoop):
          items.append(
            (
              "parloop",
              t.kind,
              expr_key(t.total),
              expr_key(t.nchunks),
              var_key(t.chunk),
              var_key(t.start),
              var_key(t.stop),
              block_key(t.body),
            )
          )
        else:
          items.append((type(t).__name__, id(t)))
      return tuple(items)

    body_key = block_key(s.body)
    owned_key = tuple(var_key(v) for v in owned_vars(s.body, s.scope))
    shared_key = None
    if s.shared_tile is not None:
      shared_key = (
        tuple(expr_key(d) for d in s.shared_tile.shape),
        tuple(expr_key(d) for d in s.shared_tile.reduce.shape),
        tuple(
          (chk.err.exc, chk.err.fmt, chk.err.loc, expr_key(chk.cond), tuple(expr_key(a) for a in chk.args[:4]))
          for chk in s.shared_tile.pre_checks
        ),
      )
    monoid_key = None
    if s.monoidal_tile is not None:
      mp = s.monoidal_tile
      saved_def = set(defined)
      for v in mp.batch_index:
        defined[v.uid] = v
      if mp.mask_index_vars is not None:
        for v in mp.mask_index_vars:
          defined[v.uid] = v
      if mp.tile_origin_vars is not None:
        for v in mp.tile_origin_vars:
          defined[v.uid] = v

      def _stage_key(st: ir.TileStage) -> tuple:
        st_defs = defined_vars(st.body, into_parloops=True)
        for v in st.index:
          st_defs[v.uid] = v
        added = [u for u in st_defs if u not in defined]
        defined.update(st_defs)
        try:
          return (block_key(st.body), expr_key(st.value))
        finally:
          for u in added:
            defined.pop(u, None)

      try:
        monoid_key = (
          mp.kind,
          mp.elt,
          mp.score_op,
          mp.split_k,
          tuple(expr_key(d) for d in mp.batch_shape),
          expr_key(mp.m_dim),
          expr_key(mp.n_dim),
          expr_key(mp.k_dim),
          None if mp.d_dim is None else expr_key(mp.d_dim),
          None if mp.score_scale is None else expr_key(mp.score_scale),
          None if mp.mask_cond is None else expr_key(mp.mask_cond),
          None if mp.mask_buf is None else var_key(mp.mask_buf),
          None if mp.mask_index is None else expr_key(mp.mask_index),
          None if mp.mask_fill is None else expr_key(mp.mask_fill),
          None if mp.tile_false_guard is None else expr_key(mp.tile_false_guard),
          None if mp.tile_true_guard is None else expr_key(mp.tile_true_guard),
          tuple((k, var_key(b)) for k, b in sorted(mp.out_bufs.items())),
          tuple((k, expr_key(e)) for k, e in sorted(mp.aux_exprs.items())),
          tuple((k, _stage_key(st)) for k, st in sorted(mp.stages.items())),
        )
      finally:
        for u in list(defined):
          if u not in saved_def:
            defined.pop(u, None)
    warp_key = None
    if s.warp_reduce is not None:
      warp_key = (
        tuple(expr_key(d) for d in s.warp_reduce.shape),
        tuple(expr_key(d) for d in s.warp_reduce.reduce.shape),
      )
    free_types = tuple(v.type for v in free)
    return (s.kind, free_types, body_key, owned_key, shared_key, monoid_key, warp_key), free

  def par_loop(self, s: ir.ParLoop, out: list[str], indent: int) -> None:
    pad = "  " * indent
    k = len(self.kernels)
    defined = defined_vars(s.body, into_parloops=True)
    for v in (s.chunk, s.start, s.stop):
      defined[v.uid] = v
    ir_key, free = self.loop_signature(s, defined)
    info = KernelInfo(k, s, free)
    self.kernels.append(info)
    info.fields = [("rt", "ptr", None), ("total", "scalar", None), ("csize", "scalar", None)]
    for v in free:
      if isinstance(v.type, BufType):
        info.fields.append((v.name, "buf", v.type.elt))
      else:
        info.fields.append((v.name, "scalar", v.type))

    # Loops with the same IR structure share one compiled body (no C-text
    # canonicalization or duplicate codegen)
    shared = self.bodies.setdefault(ir_key, k)
    if shared == k:
      lines = ["typedef struct {", "  pk_call* rt;", "  int64_t total;", "  int64_t csize;"]
      for v in free:
        if isinstance(v.type, BufType):
          lines.append(f"  pk_buf* {v.name}; {self.ctype(v.type.elt)}* {v.name}_d;")
        else:
          lines.append(f"  {self.ctype(v.type)} {v.name};")
      lines.append(f"}} pk_ctx_{k};")
      lines.append("")

      lines.append(f"static void pk_body_{k}(void* ctx_, int64_t {s.chunk.name}) {{")
      lines.append(f"  pk_ctx_{k}* ctx = (pk_ctx_{k}*)ctx_;")
      lines.append("  pk_call* rt = ctx->rt;")
      for v in free:
        if isinstance(v.type, BufType):
          ct = self.ctype(v.type.elt)
          lines.append(f"  pk_buf* const {v.name} = ctx->{v.name}; {ct}* const {v.name}_d = ctx->{v.name}_d;")
        else:
          lines.append(f"  const {self.ctype(v.type)} {v.name} = ctx->{v.name};")
      for _uid, v in sorted(defined.items()):
        if v is not s.chunk:
          self.declare(v, lines)
      saved = (self.fail_label, self.label_used)
      self.n_labels += 1
      self.fail_label, self.label_used = f"pk_fail_{self.n_labels}", False
      lines.append("  if (rt->err_code) return;")
      lines.append(f"  {s.start.name} = {s.chunk.name} * ctx->csize;")
      lines.append(f"  {s.stop.name} = {s.start.name} + ctx->csize;")
      lines.append(f"  if ({s.stop.name} > ctx->total) {s.stop.name} = ctx->total;")
      self.nesting += 1
      try:
        self.block(s.body, lines, 1)
      finally:
        self.nesting -= 1
      lines.append("  return;")
      if self.label_used:
        lines.append(f"{self.fail_label}:")
        for v in owned_vars(s.body, s.scope):
          lines.append(f"  pk_release(&{v.name});")
        lines.append("  return;")
      lines.append("}")
      lines.append("")
      self.fail_label, self.label_used = saved
      self.functions.append("\n".join(lines))
    fields = [v.name for v in self.kernels[shared].free]

    # device kernel (shared across structurally identical loops so repeated
    # transformer layers compile one GPU kernel instead of L copies!)
    on_device = False
    if self.device is not None and not self.nesting:
      if shared == k:
        on_device = self.device.add_kernel(info, self)
      else:
        info.device = self.kernels[shared].device
        on_device = info.device is not None

    # launch site
    total, nchunks = self.expr(s.total), self.expr(s.nchunks)
    out.append(f"{pad}{{")
    out.append(f"{pad}  pk_ctx_{shared} c;")
    out.append(f"{pad}  c.rt = rt; c.total = {total};")
    out.append(f"{pad}  c.csize = ({nchunks}) > 0 ? (c.total + ({nchunks}) - 1) / ({nchunks}) : 0;")
    for name, v in zip(fields, free):
      if isinstance(v.type, BufType):
        out.append(f"{pad}  c.{name} = {v.name}; c.{name}_d = {v.name}_d;")
      else:
        out.append(f"{pad}  c.{name} = {v.name};")
    launch = f"pk_parallel_for(rt, pk_body_{shared}, &c, {nchunks});"
    sync = ""
    if self.device is not None and not self.nesting:
      # run on the CPU: sync only the specific buffers this loop touches
      inner_stmts = list(ir.walk_stmts(s.body))
      stored = {x.buf.uid for x in inner_stmts if isinstance(x, ir.Store)}
      loaded = {
        x.buf.uid for st in inner_stmts for e in ir.stmt_exprs(st) for x in ir.walk_expr(e) if isinstance(x, ir.Load)
      }
      rt_buf_uids: set[int] = set()
      has_gemm = False
      for st in inner_stmts:
        for e in ir.stmt_exprs(st):
          for x in ir.walk_expr(e):
            if isinstance(x, ir.RtCall) and not x.pure and x.args and x.name in ("gemm", "argsort"):
              has_gemm = True
              for a in x.args:
                if isinstance(a, ir.Var) and isinstance(a.type, BufType):
                  rt_buf_uids.add(a.uid)
      if has_gemm:
        sync = "PK_DEV_SYNC(rt); " + "".join(
          f"{v.name}->flags &= ~PK_UNTOUCHED; " for v in free if v.uid in stored or v.uid in rt_buf_uids
        )
      else:
        parts = []
        for v in free:
          if not isinstance(v.type, BufType):
            continue
          if v.uid in stored:
            parts.append(f"PK_DEV_SYNC_WRITE(rt, {v.name}); {v.name}->flags &= ~PK_UNTOUCHED; ")
          elif v.uid in loaded:
            parts.append(f"PK_DEV_SYNC_READ(rt, {v.name}); ")
        sync = "".join(parts)
    if on_device:
      out.append(
        f"{pad}  if (rt->device && c.total >= pk_device_min(rt, {self.expr(s.work) if s.work is not None else 1}) && "
        f"((pk_device_fn)rt->device)({shared}, &c, c.total, {nchunks}) == 0) rt->dev_live = 1;"
      )
      out.append(f"{pad}  else {{ {sync}{launch} }}")
    else:
      out.append(f"{pad}  {sync}{launch}")
    out.append(f"{pad}}}")
    self.label_used = True
    out.append(f"{pad}if (PK_UNLIKELY(rt->err_code)) goto {self.fail_label};")

  # ---- whole module --------------------------------------------------------

  def module(self) -> str:
    fn = self.fn
    # which nchunks variables belong to which loop (for heavy / device hints)
    for s in ir.walk_stmts(fn.body):
      if isinstance(s, ir.ParLoop) and isinstance(s.nchunks, ir.Var):
        on_device = self.device is not None and self.device.eligible(s)
        self.loop_info[s.nchunks.uid] = (s.heavy, on_device)

    body: list[str] = []
    self.fail_label = "pk_fail_0"
    self.block(fn.body, body, 1)

    head: list[str] = []
    head.append(
      "PK_EXPORT int32_t pk_entry(const int64_t* ia, const double* fa, int64_t* io, double* fo, int64_t* eo) {"
    )
    head.append("  pk_call rt_; pk_call* rt = &rt_;")
    head.append("  memset(rt, 0, sizeof(pk_call));")
    head.append("  rt->nthreads = ia[0]; rt->grain = ia[1];")
    head.append("  rt->parfor = (void*)(intptr_t)ia[2]; rt->device = (void*)(intptr_t)ia[3];")
    head.append("  rt->gpu_min = ia[4]; rt->rng_key = (uint64_t)ia[5]; rt->device_chunks = ia[6];")
    head.append("  rt->blas = ia[7]; rt->gemm[0] = (void*)(intptr_t)ia[8]; rt->gemm[1] = (void*)(intptr_t)ia[9];")
    i_pos, f_pos = HEADER_SLOTS, 0
    ext: list[str] = []
    param_uids: set[int] = set()
    self.params_plan: list[tuple] = []
    for p in fn.params:
      if isinstance(p, ir.ArrayParam):
        elt = p.buf.type.elt
        ct = self.ctype(elt)
        name = p.buf.name
        head.append(f"  pk_buf {name}_ext = {{ (void*)(intptr_t)ia[{i_pos}], ia[{i_pos + 1}], 1, PK_EXTERNAL }};")
        head.append(f"  pk_buf* {name} = &{name}_ext; {ct}* {name}_d = ({ct}*){name}_ext.data;")
        head.append(f"  int64_t {p.offset.name} = ia[{i_pos + 2}];")
        i_pos += 3
        for v in p.shape:
          head.append(f"  int64_t {v.name} = ia[{i_pos}];")
          i_pos += 1
        if p.strides is not None:
          for v in p.strides:
            head.append(f"  int64_t {v.name} = ia[{i_pos}];")
            i_pos += 1
        ext.append(f"&{name}_ext")
        for v in (p.buf, p.offset, *p.shape, *(p.strides or ())):
          param_uids.add(v.uid)
        self.params_plan.append(("a", elt, len(p.shape), p.strides is not None, p))
      else:
        t = p.type
        ct = self.ctype(t)
        if t.kind == "f":
          head.append(f"  {ct} {p.name} = ({ct})fa[{f_pos}];")
          f_pos += 1
        elif t.kind == "b":
          head.append(f"  bool {p.name} = ia[{i_pos}] != 0;")
          i_pos += 1
        else:
          head.append(f"  {ct} {p.name} = ({ct})ia[{i_pos}];")
          i_pos += 1
        param_uids.add(p.uid)
        self.params_plan.append(("s", t))
    self.n_ia, self.n_fa = i_pos, f_pos
    head.append(f"  pk_buf* pk_exts[{max(len(ext), 1)}] = {{ {', '.join(ext) if ext else 'NULL'} }};")
    for uid, v in sorted(defined_vars(fn.body).items()):
      if uid not in param_uids:
        self.declare(v, head)

    tail: list[str] = []
    o_pos, fo_pos = 0, 0
    self.results_plan: list[tuple] = []
    for r in fn.results:
      if isinstance(r, ir.ArrayResult):
        rank = len(r.shape)
        tail.append(f"  pk_export(rt, {self.expr(r.buf)}, pk_exts, {len(ext)}, io + {o_pos});")
        tail.append(f"  io[{o_pos + 3}] = {self.expr(r.offset)};")
        for j, d in enumerate(r.shape):
          tail.append(f"  io[{o_pos + 4 + j}] = {self.expr(d)};")
        for j, d in enumerate(r.strides):
          tail.append(f"  io[{o_pos + 4 + rank + j}] = {self.expr(d)};")
        self.results_plan.append(("a", r.elt, rank, o_pos))
        o_pos += 4 + 2 * rank
      else:
        t: ScalarType = r.type  # type: ignore[assignment]
        if t.kind == "f":
          tail.append(f"  fo[{fo_pos}] = (double)({self.expr(r)});")
          self.results_plan.append(("s", t, fo_pos))
          fo_pos += 1
        else:
          tail.append(f"  io[{o_pos}] = (int64_t)({self.expr(r)});")
          self.results_plan.append(("s", t, o_pos))
          o_pos += 1
    self.n_io, self.n_fo = o_pos, fo_pos
    tail.append("  if (PK_UNLIKELY(rt->err_code)) goto pk_fail_0;")
    for v in fn.scope.owned:  # results hold their own references by now
      self.release(v, tail, "  ")
    tail.append("  return 0;")
    tail.append("pk_fail_0:")
    for v in owned_vars(fn.body, fn.scope):
      self.release(v, tail, "  ")
    tail.append("  eo[0] = rt->err_id;")
    tail.append("  eo[1] = rt->err_args[0]; eo[2] = rt->err_args[1]; eo[3] = rt->err_args[2]; eo[4] = rt->err_args[3];")
    tail.append("  return 1;")
    tail.append("}")
    loc = fn.loc.short() if fn.loc else fn.name
    parts = [f"/* Generated by Parakeet from {loc} */", PRELUDE, *self.functions, *head, *body, *tail, ""]
    return "\n".join(parts)


class CModule:
  """A compiled specialization: native entry point plus calling metadata."""

  def __init__(self, fn: ir.Function, device=None):
    emitter = HostEmitter(fn, device)
    self.source = emitter.module()
    if config.print_code:
      print(self.source)
    gpu_opt = (0 if len(self.source) > 256 * 1024 else 1) if (device is not None and config.opt_level > 0) else None
    self.path = toolchain.compile_shared(self.source, opt_level=gpu_opt)
    self.lib = toolchain.load(self.path)
    self.entry = self.lib.pk_entry
    self.entry.restype = ctypes.c_int32
    self.entry.argtypes = [ctypes.c_void_p] * 5
    self.decref = self.lib.pk_decref
    self.decref.restype = None
    self.decref.argtypes = [ctypes.c_void_p]
    self.errors = emitter.errors
    self.params_plan = emitter.params_plan
    self.results_plan = emitter.results_plan
    self.n_ia, self.n_fa = emitter.n_ia, emitter.n_fa
    self.n_io, self.n_fo = max(emitter.n_io, 1), max(emitter.n_fo, 1)
    self.kernels = emitter.kernels
    self.device = device
