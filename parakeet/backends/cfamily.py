"""Shared code generation for C-like targets (C, Metal Shading Language, CUDA C).

All three backends print the same IR; they differ in type names, math function
spellings, how failures are reported and which statements are allowed.  Those
differences are hooks on :class:`CFamilyEmitter`.
"""

from __future__ import annotations

import math

import numpy as np

from ..errors import InternalError
from ..ndtypes import BufType, ScalarType
from ..syntax import ir

_CMP = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_C_MATH = {
    "sqrt": "sqrt", "exp": "exp", "exp2": "exp2", "expm1": "expm1", "log": "log", "log10": "log10",
    "log2": "log2", "log1p": "log1p", "sin": "sin", "cos": "cos", "tan": "tan", "arcsin": "asin",
    "arccos": "acos", "arctan": "atan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "arcsinh": "asinh", "arccosh": "acosh", "arctanh": "atanh", "floor": "floor", "ceil": "ceil",
    "trunc": "trunc", "rint": "rint", "cbrt": "cbrt", "erf": "erf", "erfc": "erfc",
    "lgamma": "lgamma", "tgamma": "tgamma", "arctan2": "atan2", "hypot": "hypot",
    "copysign": "copysign",
}  # fmt: skip


class Unsupported(Exception):
  """Raised by an emitter when a construct cannot be expressed on its target."""


class CFamilyEmitter:
  """Prints IR expressions and statements as C-like source."""

  #: suffix appended to libm names for float32 (``sinf``); Metal overloads instead
  float32_suffix = "f"
  int64_name = "int64_t"

  def __init__(self) -> None:
    self.errors: list[ir.ErrorInfo] = []
    self.fail_label = "pk_fail"

  # ---- types & constants ---------------------------------------------------

  def ctype(self, t: ScalarType) -> str:
    return t.ctype

  def const(self, c: ir.Const) -> str:
    t, v = c.type, c.value
    if t.kind == "b":
      return "true" if v else "false"
    if t.kind == "f":
      return self.float_literal(float(v), t)
    return self.int_literal(int(v), t)

  def int_literal(self, v: int, t: ScalarType) -> str:
    ct = self.ctype(t)
    if t.name == "int64":
      if v == -(2**63):
        return "(-INT64_C(9223372036854775807) - 1)"
      return f"INT64_C({v})"
    if t.name == "uint64":
      return f"UINT64_C({v % 2**64})"
    return f"(({ct}){v})"

  def float_literal(self, v: float, t: ScalarType) -> str:
    ct = self.ctype(t)
    if math.isnan(v):
      return f"(({ct})NAN)"
    if math.isinf(v):
      return f"(({ct}){'-' if v < 0 else ''}INFINITY)"
    if t.name == "float32":
      text = f"{float(np.float32(v)):.9g}"
      if "." not in text and "e" not in text and "n" not in text:
        text += ".0"
      return text + "f"
    text = repr(float(v))
    if "." not in text and "e" not in text:
      text += ".0"
    return text

  # ---- expressions ---------------------------------------------------------

  def var(self, v: ir.Var) -> str:
    return v.name

  def data(self, buf: ir.Var) -> str:
    """Expression for the element pointer of a buffer variable."""
    return f"{buf.name}_d"

  def math_fn(self, op: str, t: ScalarType) -> str:
    name = _C_MATH[op]
    return name + (self.float32_suffix if t.name == "float32" else "")

  def expr(self, e: ir.Expr) -> str:
    if isinstance(e, ir.Const):
      return self.const(e)
    if isinstance(e, ir.Var):
      return self.var(e)
    if isinstance(e, ir.Load):
      return f"{self.data(e.buf)}[{self.expr(e.index)}]"
    if isinstance(e, ir.Cast):
      return self.cast(e)
    if isinstance(e, ir.Prim):
      return self.prim(e)
    if isinstance(e, ir.RtCall):
      return self.rtcall(e)
    if isinstance(e, ir.Null):
      return "NULL"
    raise InternalError(f"cannot emit expression {e!r}")

  def cast(self, e: ir.Cast) -> str:
    src, dst = e.arg.type, e.type
    x = self.expr(e.arg)
    if src.name == dst.name:
      return x
    ct = self.ctype(dst)
    if dst.kind == "b":
      return f"({x} != 0)"
    if src.kind == "f" and dst.kind in "iu":
      return self.float_to_int(x, src, dst)
    return f"(({ct}){x})"

  def float_to_int(self, x: str, src: ScalarType, dst: ScalarType) -> str:
    ct = self.ctype(dst)
    if dst.name == "uint64":
      return f"pk_f2u64((double){x})"
    if dst.name == "int64":
      return f"pk_f2i64((double){x})"
    return f"(({ct})pk_f2i64((double){x}))"

  def prim(self, e: ir.Prim) -> str:
    op, t = e.op, e.type
    args = [self.expr(a) for a in e.args]
    at = e.args[0].type
    ct = self.ctype(t)
    if op in _CMP:
      return f"({args[0]} {_CMP[op]} {args[1]})"
    if op == "and":
      return f"({args[0]} && {args[1]})"
    if op == "or":
      return f"({args[0]} || {args[1]})"
    if op == "not":
      return f"(!{args[0]})"
    if op == "select":
      return f"({args[0]} ? {args[1]} : {args[2]})"
    if op in ("isnan", "isinf", "isfinite", "signbit"):
      return self.predicate(op, args[0], at)
    if op in ("popcount", "clz", "ctz"):
      return f"(({ct})pk_{op}_{at.name}({args[0]}))"
    if op in ("rotl", "rotr"):
      return f"pk_{op}_{t.name}({args[0]}, {args[1]})"
    if op == "bswap":
      return f"pk_bswap_{t.name}({args[0]})"
    if op == "bf16_to_f32":
      return f"pk_bf16_to_f32({args[0]})"
    if op == "f32_to_bf16":
      return f"pk_f32_to_bf16({args[0]})"
    if t.kind == "f":
      return self.float_prim(op, args, t)
    if t.kind == "b":
      raise InternalError(f"primitive {op} on booleans")
    return self.int_prim(op, args, t, ct)

  def predicate(self, op: str, x: str, t: ScalarType) -> str:
    if op == "isnan":
      return f"({x} != {x})"
    if op == "signbit":
      return f"(signbit({x}) != 0)"
    return f"({op}({x}) != 0)"

  def float_prim(self, op: str, a: list[str], t: ScalarType) -> str:
    n = t.name
    if op == "add":
      return f"({a[0]} + {a[1]})"
    if op == "sub":
      return f"({a[0]} - {a[1]})"
    if op == "mul":
      return f"({a[0]} * {a[1]})"
    if op == "div":
      return f"({a[0]} / {a[1]})"
    if op == "neg":
      return f"(-{a[0]})"
    if op == "abs":
      return f"{'fabsf' if n == 'float32' and self.float32_suffix else 'fabs'}({a[0]})"
    if op in ("min", "max", "floordiv", "mod"):
      return f"pk_{op}_{n}({a[0]}, {a[1]})"
    if op == "fmod":
      return f"{'fmodf' if n == 'float32' and self.float32_suffix else 'fmod'}({a[0]}, {a[1]})"
    if op == "pow":
      return f"{'powf' if n == 'float32' and self.float32_suffix else 'pow'}({a[0]}, {a[1]})"
    if op in _C_MATH:
      return f"{self.math_fn(op, t)}({', '.join(a)})"
    raise InternalError(f"unknown float primitive {op}")

  def int_prim(self, op: str, a: list[str], t: ScalarType, ct: str) -> str:
    n = t.name
    if op == "add":
      return f"(({ct})({a[0]} + {a[1]}))"
    if op == "sub":
      return f"(({ct})({a[0]} - {a[1]}))"
    if op == "mul":
      return f"(({ct})({a[0]} * {a[1]}))"
    if op == "neg":
      return f"(({ct})(0 - {a[0]}))"
    if op == "abs":
      return f"(({ct})({a[0]} < 0 ? (0 - {a[0]}) : {a[0]}))"
    if op == "min":
      return f"({a[0]} < {a[1]} ? {a[0]} : {a[1]})"
    if op == "max":
      return f"({a[0]} > {a[1]} ? {a[0]} : {a[1]})"
    if op == "bitand":
      return f"(({ct})({a[0]} & {a[1]}))"
    if op == "bitor":
      return f"(({ct})({a[0]} | {a[1]}))"
    if op == "bitxor":
      return f"(({ct})({a[0]} ^ {a[1]}))"
    if op == "invert":
      return f"(({ct})(~{a[0]}))"
    if op in ("floordiv", "mod", "fmod", "pow", "lshift", "rshift"):
      fn = {"fmod": "cmod", "pow": "ipow", "lshift": "shl", "rshift": "shr"}.get(op, op)
      return f"pk_{fn}_{n}({a[0]}, {a[1]})"
    raise InternalError(f"unknown integer primitive {op}")

  def rtcall(self, e: ir.RtCall) -> str:
    raise InternalError(f"runtime call {e.name} is not available here")

  # ---- statements ----------------------------------------------------------

  def block(self, stmts: list[ir.Stmt], out: list[str], indent: int) -> None:
    for s in stmts:
      self.stmt(s, out, indent)

  def stmt(self, s: ir.Stmt, out: list[str], indent: int) -> None:
    pad = "  " * indent
    if isinstance(s, ir.Assign):
      self.assign(s, out, pad)
    elif isinstance(s, ir.Store):
      self.store(s, out, pad)
    elif isinstance(s, ir.If):
      out.append(f"{pad}if ({self.expr(s.cond)}) {{")
      self.block(s.then, out, indent + 1)
      if s.orelse:
        out.append(f"{pad}}} else {{")
        self.block(s.orelse, out, indent + 1)
      out.append(f"{pad}}}")
    elif isinstance(s, ir.ForRange):
      self.for_range(s, out, indent)
    elif isinstance(s, ir.While):
      out.append(f"{pad}for (;;) {{")
      self.block(s.body, out, indent + 1)
      out.append(f"{pad}}}")
    elif isinstance(s, ir.Break):
      out.append(f"{pad}break;")
    elif isinstance(s, ir.Continue):
      out.append(f"{pad}continue;")
    elif isinstance(s, ir.Check):
      self.check(s, out, pad)
    elif isinstance(s, ir.Comment):
      out.append(f"{pad}/* {s.text.replace('*/', '* /')} */")
    elif isinstance(s, ir.Alloc):
      self.alloc(s, out, pad)
    elif isinstance(s, ir.Incref):
      self.incref(s, out, pad)
    elif isinstance(s, ir.Release):
      self.release(s.var, out, pad)
    elif isinstance(s, ir.ReleaseScope):
      for v in s.scope.owned:
        self.release(v, out, pad)
    elif isinstance(s, ir.ParLoop):
      self.par_loop(s, out, indent)
    else:
      raise InternalError(f"cannot emit statement {s!r}")

  def assign(self, s: ir.Assign, out: list[str], pad: str) -> None:
    if isinstance(s.var.type, BufType):
      self.assign_buf(s, out, pad)
    else:
      out.append(f"{pad}{s.var.name} = {self.expr(s.value)};")

  def store(self, s: ir.Store, out: list[str], pad: str) -> None:
    ct = self.ctype(s.buf.type.elt)
    value = self.expr(s.value)
    if s.value.type is not s.buf.type.elt and s.value.type.name != s.buf.type.elt.name:
      value = f"(({ct}){value})"
    if s.atomic is not None:
      self.atomic_store(s, ct, value, out, pad)
      return
    out.append(f"{pad}{self.data(s.buf)}[{self.expr(s.index)}] = {value};")

  def atomic_store(self, s: ir.Store, ct: str, value: str, out: list[str], pad: str) -> None:
    raise Unsupported(f"atomic {s.atomic} store")

  def for_range(self, s: ir.ForRange, out: list[str], indent: int) -> None:
    pad = "  " * indent
    i = s.var.name
    start, stop, step = self.expr(s.start), self.expr(s.stop), self.expr(s.step)
    if isinstance(s.step, ir.Const):
      if s.step.value > 0:
        cond = f"{i} < {stop}"
        inc = f"++{i}" if s.step.value == 1 else f"{i} += {step}"
      else:
        cond = f"{i} > {stop}"
        inc = f"--{i}" if s.step.value == -1 else f"{i} += {step}"
    else:
      cond = f"({step} > 0) ? ({i} < {stop}) : ({i} > {stop})"
      inc = f"{i} += {step}"
    pragma = self.loop_pragma(s)
    if pragma:
      out.append(f"{pad}{pragma}")
    out.append(f"{pad}for ({i} = {start}; {cond}; {inc}) {{")
    self.block(s.body, out, indent + 1)
    out.append(f"{pad}}}")

  def loop_pragma(self, s: ir.ForRange) -> str | None:
    return None

  def error_id(self, err: ir.ErrorInfo) -> int:
    self.errors.append(err)
    return len(self.errors) - 1

  # hooks with no default
  def check(self, s: ir.Check, out: list[str], pad: str) -> None:
    raise NotImplementedError

  def assign_buf(self, s: ir.Assign, out: list[str], pad: str) -> None:
    raise Unsupported("buffer variables")

  def alloc(self, s: ir.Alloc, out: list[str], pad: str) -> None:
    raise Unsupported("array allocation")

  def incref(self, s: ir.Incref, out: list[str], pad: str) -> None:
    raise Unsupported("reference counting")

  def release(self, v: ir.Var, out: list[str], pad: str) -> None:
    raise Unsupported("reference counting")

  def par_loop(self, s: ir.ParLoop, out: list[str], indent: int) -> None:
    # a loop of exactly one chunk is a region the host outlines into its own
    # function; anywhere else it is just its body
    if not (isinstance(s.nchunks, ir.Const) and s.nchunks.value == 1):
      raise Unsupported("nested parallel loop")
    pad = "  " * indent
    out.append(f"{pad}{s.chunk.name} = 0; {s.start.name} = 0; {s.stop.name} = {self.expr(s.total)};")
    self.block(s.body, out, indent)


# ---------------------------------------------------------------------------
# analysis helpers shared by backends
# ---------------------------------------------------------------------------


def defined_vars(block: list[ir.Stmt], into_parloops: bool = False) -> dict[int, ir.Var]:
  """Variables assigned in ``block`` (not descending into ParLoop bodies)."""
  out: dict[int, ir.Var] = {}

  def walk(stmts):
    for s in stmts:
      for v in ir.stmt_defs(s):
        out[v.uid] = v
      if isinstance(s, ir.ParLoop):
        if into_parloops:
          walk(s.body)
        else:
          # chunk/start/stop belong to the outlined body
          for v in (s.chunk, s.start, s.stop):
            out.pop(v.uid, None)
        continue
      for b in ir.sub_blocks(s):
        walk(b)

  walk(block)
  return out


def used_vars(block: list[ir.Stmt]) -> dict[int, ir.Var]:
  """Variables read anywhere in ``block`` (including nested blocks)."""
  out: dict[int, ir.Var] = {}
  for s in ir.walk_stmts(block):
    for e in ir.stmt_exprs(s):
      for x in ir.walk_expr(e):
        if isinstance(x, ir.Var):
          out[x.uid] = x
    if isinstance(s, ir.Store):
      out[s.buf.uid] = s.buf
  return out


def owned_vars(block: list[ir.Stmt], scope: ir.Scope) -> list[ir.Var]:
  """Buffer variables owning a reference inside one C function."""
  out = list(scope.owned)

  def walk(stmts):
    for s in stmts:
      if isinstance(s, ir.ParLoop):
        continue
      if isinstance(s, (ir.ForRange, ir.While)):
        out.extend(s.scope.owned)
      for b in ir.sub_blocks(s):
        walk(b)

  walk(block)
  seen: set[int] = set()
  uniq = []
  for v in out:
    if v.uid not in seen:
      seen.add(v.uid)
      uniq.append(v)
  return uniq


def loop_depth(block: list[ir.Stmt]) -> int:
  best = 0
  for s in block:
    inner = max((loop_depth(b) for b in ir.sub_blocks(s)), default=0)
    if isinstance(s, (ir.ForRange, ir.While)):
      inner += 1
    best = max(best, inner)
  return best


def expr_types(block: list[ir.Stmt]) -> set[ScalarType]:
  """Every scalar type appearing in ``block`` (for device capability checks)."""
  out: set[ScalarType] = set()
  for s in ir.walk_stmts(block):
    for e in ir.stmt_exprs(s):
      for x in ir.walk_expr(e):
        t = x.type
        out.add(t.elt if isinstance(t, BufType) else t.strong)
    for v in ir.stmt_defs(s):
      t = v.type
      out.add(t.elt if isinstance(t, BufType) else t.strong)
  return out


def prim_ops(block: list[ir.Stmt]) -> set[tuple[str, str]]:
  out: set[tuple[str, str]] = set()
  for s in ir.walk_stmts(block):
    for e in ir.stmt_exprs(s):
      for x in ir.walk_expr(e):
        if isinstance(x, ir.Prim):
          out.add((x.op, x.type.kind))
        elif isinstance(x, ir.RtCall):
          out.add(("@" + x.name, ""))
  return out
