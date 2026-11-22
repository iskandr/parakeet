"""The host backend: turn IR into a C module, compile it and load it.

Every other backend builds on this one - GPU backends run the host program
here and divert eligible parallel loops to a device.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

from ..config import config
from ..ndtypes import BufType, ScalarType
from ..syntax import ir
from . import toolchain
from .cfamily import CFamilyEmitter, defined_vars, owned_vars, used_vars
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
    self.loop_info: dict[int, tuple[bool, bool]] = {}
    self.n_labels = 0
    self.label_used = False

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
    self.label_used = True
    out.append(f"{pad}{name} = pk_alloc(rt, {self.expr(s.count)}, sizeof({ct}), {int(s.zero)});")
    out.append(f"{pad}if (PK_UNLIKELY(!{name})) goto {self.fail_label};")
    out.append(f"{pad}{name}_d = ({ct}*){name}->data;")

  def incref(self, s: ir.Incref, out: list[str], pad: str) -> None:
    out.append(f"{pad}pk_incref({self.expr(s.buf)});")

  def release(self, v: ir.Var, out: list[str], pad: str) -> None:
    out.append(f"{pad}pk_release(&{v.name});")

  def loop_pragma(self, s: ir.ForRange) -> str | None:
    # a floating point fold may only be vectorised if the compiler is told
    # that reordering it is fine (integers need no permission)
    red = s.reduction
    if red is None or not toolchain.supports_simd_pragmas():
      return None
    op, acc = red
    simple = all(
      isinstance(b, (ir.Assign, ir.Comment)) or (isinstance(b, ir.ReleaseScope) and not b.scope.owned) for b in s.body
    )
    if acc.type.kind != "f" or not simple:
      return None
    return f"#pragma omp simd reduction({'+' if op == 'add' else '*'}:{acc.name})"

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
      out.append(f"{pad}pk_buf* {v.name} = NULL; {self.ctype(t.elt)}* {v.name}_d = NULL;")
    elif t.kind == "b":
      out.append(f"{pad}bool {v.name} = false;")
    else:
      out.append(f"{pad}{self.ctype(t)} {v.name} = 0;")

  # ---- parallel loops ------------------------------------------------------

  def par_loop(self, s: ir.ParLoop, out: list[str], indent: int) -> None:
    pad = "  " * indent
    k = len(self.kernels)
    defined = defined_vars(s.body, into_parloops=True)
    for v in (s.chunk, s.start, s.stop):
      defined[v.uid] = v
    used = used_vars(s.body)
    free = [v for uid, v in sorted(used.items()) if uid not in defined]
    info = KernelInfo(k, s, free)
    self.kernels.append(info)

    # context struct
    lines = ["typedef struct {", "  pk_call* rt;", "  int64_t total;", "  int64_t csize;"]
    info.fields = [("rt", "ptr", None), ("total", "scalar", None), ("csize", "scalar", None)]
    for v in free:
      if isinstance(v.type, BufType):
        lines.append(f"  pk_buf* {v.name}; {self.ctype(v.type.elt)}* {v.name}_d;")
        info.fields.append((v.name, "buf", v.type.elt))
      else:
        lines.append(f"  {self.ctype(v.type)} {v.name};")
        info.fields.append((v.name, "scalar", v.type))
    lines.append(f"}} pk_ctx_{k};")
    lines.append("")

    # outlined body
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
    self.block(s.body, lines, 1)
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

    # device kernel (if this loop is eligible)
    on_device = False
    if self.device is not None:
      on_device = self.device.add_kernel(info, self)

    # launch site
    total, nchunks = self.expr(s.total), self.expr(s.nchunks)
    out.append(f"{pad}{{")
    out.append(f"{pad}  pk_ctx_{k} c;")
    out.append(f"{pad}  c.rt = rt; c.total = {total};")
    out.append(f"{pad}  c.csize = ({nchunks}) > 0 ? (c.total + ({nchunks}) - 1) / ({nchunks}) : 0;")
    for v in free:
      if isinstance(v.type, BufType):
        out.append(f"{pad}  c.{v.name} = {v.name}; c.{v.name}_d = {v.name}_d;")
      else:
        out.append(f"{pad}  c.{v.name} = {v.name};")
    launch = f"pk_parallel_for(rt, pk_body_{k}, &c, {nchunks});"
    if on_device:
      out.append(
        f"{pad}  if (!(rt->device && c.total >= rt->gpu_min && "
        f"((pk_device_fn)rt->device)({k}, &c, c.total, {nchunks}) == 0)) {launch}"
      )
    else:
      out.append(f"{pad}  {launch}")
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
      tail.append(f"  pk_release(&{v.name});")
    tail.append("  return 0;")
    tail.append("pk_fail_0:")
    for v in owned_vars(fn.body, fn.scope):
      tail.append(f"  pk_release(&{v.name});")
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
    self.path = toolchain.compile_shared(self.source)
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
