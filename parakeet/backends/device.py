"""Machinery shared by the GPU backends (Metal and CUDA).

A *planner* decides, while a module is being generated, which parallel loops
can run on the device and produces kernel source for them.  A *device* runs
those kernels when the compiled host code calls back into Python.
"""

from __future__ import annotations

import builtins
import ctypes
import threading
from dataclasses import dataclass, field
from typing import Any

from ..errors import SourceLocation
from ..ndtypes import BufType, ScalarType
from ..syntax import ir
from .cfamily import Unsupported, defined_vars, used_vars

DEVICE_FN = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64)

_CTYPES = {
    "bool": ctypes.c_bool, "int8": ctypes.c_int8, "int16": ctypes.c_int16, "int32": ctypes.c_int32,
    "int64": ctypes.c_int64, "uint8": ctypes.c_uint8, "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32, "uint64": ctypes.c_uint64, "float32": ctypes.c_float,
    "float64": ctypes.c_double,
}  # fmt: skip


def ctype_of(t: ScalarType):
  return _CTYPES[t.name]


def merge_regions(spans: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
  """Group host byte ranges ``(address, nbytes)`` that overlap.

  Arguments of one kernel may alias (``a`` and ``a[0]``, two views of one
  array, ...).  Each connected group of overlapping ranges must live in a
  single device buffer, otherwise a store through one name is invisible
  through the other and copies back to the host would clobber each other.

  Returns ``(regions, where)``: the merged ``(address, nbytes)`` regions and,
  for every input span, ``(region index, byte offset inside the region)``.
  Empty spans each get a region of their own.
  """
  regions: list[tuple[int, int]] = []
  where: list[tuple[int, int]] = [(0, 0)] * len(spans)
  order = sorted((i for i, (_, n) in enumerate(spans) if n > 0), key=lambda i: spans[i])
  lo = hi = -1
  for i in order:
    addr, n = spans[i]
    if regions and addr < hi:
      hi = max(hi, addr + n)
      regions[-1] = (lo, hi - lo)
    else:
      lo, hi = addr, addr + n
      regions.append((lo, n))
    where[i] = (len(regions) - 1, addr - lo)
  for i, (_, n) in enumerate(spans):
    if n <= 0:
      regions.append((0, 0))
      where[i] = (len(regions) - 1, 0)
  return regions, where


def make_runtime_error(info: ir.ErrorInfo, args: list[int]) -> BaseException:
  """Build the Python exception for a failure reported by generated code."""
  cls = getattr(builtins, info.exc, RuntimeError)
  if not (isinstance(cls, type) and issubclass(cls, BaseException)):
    cls = RuntimeError
  try:
    msg = info.fmt.format(*args)
  except (IndexError, KeyError, ValueError):
    msg = info.fmt
  exc = cls(msg)
  if info.loc is not None:
    exc.add_note(f"raised by compiled code at {info.loc.short()}")
    line = info.loc.line()
    if line:
      exc.add_note("    " + line.strip())
  return exc


@dataclass(eq=False)
class KernelPlan:
  loop: ir.ParLoop
  source: str
  scalars: list[ir.Var]
  bufs: list[ir.Var]
  errors: list[ir.ErrorInfo]
  written: set[int]
  compiled: Any = None
  host_struct: Any = None
  arg_struct: Any = None
  failed: bool = False
  extra: dict = field(default_factory=dict)


class _PkCall(ctypes.Structure):
  _fields_ = [("err_code", ctypes.c_int32), ("err_id", ctypes.c_int32), ("err_args", ctypes.c_int64 * 4)]


class DevicePlanner:
  """Per-module helper: selects loops for the device and writes their kernels."""

  name = "device"

  def __init__(self, device: DeviceBase):
    self.device = device
    self.plans: dict[int, KernelPlan | None] = {}
    self.fallbacks: list[tuple[SourceLocation | None, str]] = []

  def generate(self, loop: ir.ParLoop, free: list[ir.Var]) -> KernelPlan:  # pragma: no cover
    raise NotImplementedError

  @staticmethod
  def free_vars(loop: ir.ParLoop) -> list[ir.Var]:
    defined = defined_vars(loop.body, into_parloops=True)
    for v in (loop.chunk, loop.start, loop.stop):
      defined[v.uid] = v
    used = used_vars(loop.body)
    return [v for uid, v in sorted(used.items()) if uid not in defined]

  def plan(self, loop: ir.ParLoop) -> KernelPlan | None:
    key = id(loop)
    if key not in self.plans:
      try:
        self.plans[key] = self.generate(loop, self.free_vars(loop))
      except Unsupported as e:
        self.plans[key] = None
        self.fallbacks.append((loop.loc, str(e)))
    return self.plans[key]

  def eligible(self, loop: ir.ParLoop) -> bool:
    return self.plan(loop) is not None

  def add_kernel(self, info, host_emitter) -> bool:
    """Called by the host emitter for each parallel loop it outlines."""
    plan = self.plan(info.loop)
    if plan is None:
      return False
    # mirror of the C context struct the host passes to the callback
    fields: list = []
    for name, kind, t in info.fields:
      if kind == "ptr":
        fields.append((name, ctypes.c_void_p))
      elif kind == "buf":
        fields.append((name, ctypes.c_void_p))
        fields.append((name + "_d", ctypes.c_void_p))
      elif t is None:
        fields.append((name, ctypes.c_int64))
      else:
        fields.append((name, ctype_of(t)))
    plan.host_struct = type("HostCtx", (ctypes.Structure,), {"_fields_": fields})
    arg_fields = [("total", ctypes.c_int64), ("csize", ctypes.c_int64)]
    arg_fields += [(v.name, ctype_of(v.type)) for v in plan.scalars]
    plan.arg_struct = type("KernelArgs", (ctypes.Structure,), {"_fields_": arg_fields})
    info.device = plan
    return True

  def kernel_sources(self) -> list[str]:
    return [p.source for p in self.plans.values() if p is not None]

  def report_fallbacks(self, function_name: str) -> None:
    """Tell the user (once, at compile time) which loops stay on the CPU."""
    from ..api import warn_fallback

    seen = set()
    for loc, reason in self.fallbacks:
      where = f" at {loc.short()}" if loc is not None else ""
      msg = f"{function_name}: a parallel loop{where} will run on the CPU instead of {self.name} because {reason}"
      if msg not in seen:
        seen.add(msg)
        warn_fallback(msg)


class _CallState(threading.local):
  def __init__(self) -> None:
    self.stack: list = []
    self.cache: dict[int, _Resident] = {}


class _Resident:
  """Host memory that currently also lives on the device."""

  __slots__ = ("addr", "nbytes", "dev", "dirty")

  def __init__(self, addr: int, nbytes: int, dev) -> None:
    self.addr, self.nbytes, self.dev = addr, nbytes, dev
    self.dirty = False  # the device copy is newer than the host's


class _Token:
  __slots__ = ("spec", "pending")

  def __init__(self, spec) -> None:
    self.spec = spec
    self.pending: BaseException | None = None


class DeviceBase:
  """Receives ``run this loop`` callbacks from compiled host code."""

  name = "device"

  def __init__(self) -> None:
    self._state = _CallState()
    #: number of loops that actually ran on the device (diagnostics)
    self.launches = 0
    self._callback = DEVICE_FN(self._launch)  # keep alive: C holds its address
    self.callback_address = ctypes.cast(self._callback, ctypes.c_void_p).value or 0

  # -- call bracketing (see api.Specialization.run)
  def begin_call(self, spec, arrays):
    token = _Token(spec)
    self._state.stack.append(token)
    return token

  def end_call(self, token) -> BaseException | None:
    stack = self._state.stack
    if stack and stack[-1] is token:
      stack.pop()
    if not stack:
      try:
        self.flush()  # whatever is still only on the device belongs to the caller now
      except BaseException as e:  # noqa: BLE001
        self._state.cache.clear()
        token.pending = token.pending or e
    return token.pending

  def run(self, plan: KernelPlan, ctx_addr: int, total: int, nchunks: int) -> None:  # pragma: no cover
    raise NotImplementedError

  # -- arrays that stay on the device between the loops of one call.  Host
  # code asks for a flush before it touches array memory itself (see the
  # PK_DEV_SYNC calls the C backend emits), so between two flushes the device
  # copy of anything a kernel wrote is the only current one.

  def upload(self, addr: int, nbytes: int, external: bool):  # pragma: no cover
    raise NotImplementedError

  def download(self, dev, addr: int, nbytes: int) -> None:  # pragma: no cover
    raise NotImplementedError

  def acquire(self, regions: list[tuple[int, int]], written: set[int], external: list[bool]) -> list:
    """Device buffers for host memory ``regions``, uploaded only if they are not there already."""
    cache = self._state.cache
    out = []
    for r, (addr, nbytes) in enumerate(regions):
      if nbytes <= 0:
        out.append(self.upload(addr, 0, external[r]))
        continue
      hit = cache.get(addr)
      if hit is None or hit.nbytes != nbytes:
        for other in [e for e in cache.values() if e.addr < addr + nbytes and addr < e.addr + e.nbytes]:
          self._retire(other, True)
        hit = cache[addr] = _Resident(addr, nbytes, self.upload(addr, nbytes, external[r]))
      hit.dirty = hit.dirty or r in written
      out.append(hit.dev)
    return out

  def _retire(self, entry: _Resident, download: bool) -> None:
    self._state.cache.pop(entry.addr, None)
    if download and entry.dirty:
      self.download(entry.dev, entry.addr, entry.nbytes)

  def flush(self) -> None:
    """Bring the host up to date and forget what is on the device."""
    for entry in list(self._state.cache.values()):
      self._retire(entry, True)

  def forget(self, addr: int) -> None:
    """The host is about to free the memory at ``addr``."""
    entry = self._state.cache.get(addr)
    if entry is not None:
      self._retire(entry, False)

  def _launch(self, kernel: int, ctx_addr: int, total: int, nchunks: int) -> int:
    """Returns 0 if the loop ran (or failed for good); 1 to run it on the CPU.
    Kernel -1 is a request to flush, -2 to forget the buffer at ``ctx_addr``."""
    stack = self._state.stack
    if kernel < 0:
      try:
        if kernel == -1:
          self.flush()
        else:
          self.forget(ctx_addr or 0)
      except BaseException as e:  # noqa: BLE001 - must not propagate through C frames
        self._state.cache.clear()
        if stack:
          stack[-1].pending = stack[-1].pending or e
      return 0
    if not stack:
      return 1
    token = stack[-1]
    try:
      plan = token.spec.module.kernels[kernel].device
      if plan is None or plan.failed:
        return 1
      self.run(plan, ctx_addr, total, nchunks)
      self.launches += 1
      return 0
    except BaseException as e:  # noqa: BLE001 - must not propagate through C frames
      token.pending = e
      try:
        rt = ctypes.c_void_p.from_address(ctx_addr).value
        call = _PkCall.from_address(rt)
        call.err_id = -2
        call.err_code = 1
      except Exception:  # noqa: BLE001
        pass
      return 0


_ = BufType
