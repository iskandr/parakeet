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

import numpy as np

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


def untouched_regions(pks: list, where: list[tuple[int, int]], n_regions: int) -> list[int]:
  """Per region: 2 if the host never wrote any buffer in it, 4 if it only
  zeroed them, else 0.  The buffers' flags are cleared: from here on their
  contents are whatever the device makes of them."""
  kinds = [6] * n_regions
  for pk, (r, _) in zip(pks, where):
    kinds[r] &= pk.flags
  for pk in pks:
    pk.flags &= ~14
  return [4 if k & 4 else (2 if k & 2 else 0) for k in kinds]


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

  def __init__(self, device: DeviceBase, profile=None):
    self.device = device
    self.profile = profile if profile is not None else getattr(device, "profile", None)
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
    if loop.shared_tile is not None:
      for e in (*loop.shared_tile.shape, *loop.shared_tile.reduce.shape):
        for x in ir.walk_expr(e):
          if isinstance(x, ir.Var):
            used[x.uid] = x
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
    seen: set[int] = set()
    out: list[str] = []
    for p in self.plans.values():
      if p is not None and id(p) not in seen:
        seen.add(id(p))
        out.append(p.source)
    return out

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


class DeviceArray:
  """An array resident on a GPU device (Metal or CUDA) across ``@jit`` calls.

  Created with :func:`parakeet.to_device` or returned automatically by a
  ``@jit`` function called with ``DeviceArray`` inputs.  Passing a
  ``DeviceArray`` into subsequent ``@jit`` calls on the same device avoids
  redundant host-to-device uploads, and GPU-computed outputs are promoted
  directly to ``DeviceArray`` without device-to-host downloads — keeping
  weights, gradients, and optimizer state resident on the GPU throughout a
  training loop.

  Call :meth:`numpy` or :func:`parakeet.to_host` (or ``np.asarray(da)``) to
  bring the contents back to host memory.
  """

  __parakeet_device_array__ = True
  __slots__ = ("_addr", "_dev_buf", "_device", "_dirty", "_host", "_nbytes")

  def __init__(self, host: np.ndarray, device: DeviceBase, dev_buf: Any = None, *, dirty: bool = False) -> None:
    if not host.flags.c_contiguous:
      host = np.ascontiguousarray(host)
    self._host = host
    self._addr = host.ctypes.data
    self._nbytes = host.nbytes
    self._device = device
    self._dev_buf = dev_buf
    self._dirty = dirty

  @property
  def shape(self) -> tuple[int, ...]:
    return self._host.shape

  @property
  def dtype(self) -> np.dtype:
    return self._host.dtype

  @property
  def ndim(self) -> int:
    return self._host.ndim

  @property
  def size(self) -> int:
    return self._host.size

  @property
  def nbytes(self) -> int:
    return self._nbytes

  @property
  def strides(self) -> tuple[int, ...]:
    return self._host.strides

  @property
  def device(self) -> str:
    return self._device.name

  @property
  def on_device(self) -> bool:
    return self._dev_buf is not None

  @property
  def _host_addr(self) -> int:
    return self._addr

  def sync_host(self) -> np.ndarray:
    """Bring the host backing array up to date if the device copy is newer."""
    if self._dirty and self._dev_buf is not None and self._host.nbytes > 0:
      self._device.download(self._dev_buf, self._host_addr, self._host.nbytes)
      self._device.downloads += 1
      self._dirty = False
    return self._host

  def ensure_device(self) -> Any:
    """Ensure a device buffer exists for this array, uploading once if needed."""
    if self._dev_buf is None:
      nbytes = self._host.nbytes
      self._dev_buf = self._device.upload(self._host_addr, nbytes, external=True, untouched=0)
      if nbytes > 0:
        self._device.uploads += 1
      self._dirty = False
    return self._dev_buf

  def numpy(self) -> np.ndarray:
    """Synchronize to host and return a NumPy ``ndarray`` copy."""
    return self.sync_host().copy()

  def copy(self) -> DeviceArray:
    """Return an independent copy resident on the same device."""
    arr = self.sync_host().copy()
    da = DeviceArray(arr, self._device)
    if self._dev_buf is not None:
      da.ensure_device()
    return da

  def __array__(self, dtype=None, copy=None):
    arr = self.sync_host()
    if dtype is not None:
      return arr.astype(dtype, copy=True)
    return arr.copy() if copy else arr

  def __repr__(self) -> str:
    state = "resident" if self._dev_buf is not None else "host-staged"
    return f"DeviceArray(shape={self.shape}, dtype={self.dtype}, device={self.device!r}, {state})"


class _CallState(threading.local):
  def __init__(self) -> None:
    self.stack: list = []
    self.cache: dict[int, _Resident] = {}
    self.pinned: dict[int, DeviceArray] = {}
    self.writable_pinned: set[int] = set()
    self.pending_checks: list[tuple[int, KernelPlan]] = []


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
    #: number of host->device copies performed (diagnostics)
    self.uploads = 0
    #: number of device->host copies performed (diagnostics)
    self.downloads = 0
    self._callback = DEVICE_FN(self._launch)  # keep alive: C holds its address
    self.callback_address = ctypes.cast(self._callback, ctypes.c_void_p).value or 0

  # -- call bracketing (see api.Specialization.run)
  def begin_call(
    self,
    spec,
    arrays,
    pinned: list[tuple[DeviceArray, bool]] | None = None,
  ):
    token = _Token(spec)
    self._state.stack.append(token)
    if pinned:
      for da, writes in pinned:
        addr = da._host_addr
        self._state.pinned[addr] = da
        if writes:
          self._state.writable_pinned.add(addr)
    return token

  def end_call(self, token) -> BaseException | None:
    stack = self._state.stack
    if stack and stack[-1] is token:
      stack.pop()
    if not stack:
      try:
        self.flush(mid_call=False)  # non-promoted GPU outputs belong to the caller now
      except BaseException as e:  # noqa: BLE001
        self._state.cache.clear()
        token.pending = token.pending or e
      finally:
        self._state.pinned.clear()
        self._state.writable_pinned.clear()
    return token.pending

  def run(self, plan: KernelPlan, ctx_addr: int, total: int, nchunks: int) -> None:  # pragma: no cover
    raise NotImplementedError

  # -- arrays that stay on the device between the loops of one call (and across
  # calls when wrapped in DeviceArray).  Host code asks for a flush before it
  # touches array memory itself (PK_DEV_SYNC), so between two flushes the device
  # copy of anything a kernel wrote is the only current one.

  def upload(self, addr: int, nbytes: int, external: bool, untouched: int = 0):  # pragma: no cover
    """A device buffer for host memory.  ``untouched`` says the host never
    wrote there (2) or only zeroed it (4): nothing has to be copied."""
    raise NotImplementedError

  def download(self, dev, addr: int, nbytes: int) -> None:  # pragma: no cover
    raise NotImplementedError

  def acquire(self, regions, written: set[int], external: list[bool], untouched: list[int] | None = None) -> list:
    """Device buffers for host memory ``regions``, uploaded only if they are not there already."""
    cache = self._state.cache
    pinned = self._state.pinned
    out = []
    untouched = untouched or [0] * len(regions)
    for r, (addr, nbytes) in enumerate(regions):
      if nbytes <= 0:
        out.append(self.upload(addr, 0, external[r]))
        continue
      da = pinned.get(addr)
      if da is not None and da._nbytes == nbytes:
        dev_buf = da.ensure_device()
        if r in written:
          da._dirty = True
        out.append(dev_buf)
        continue
      hit = cache.get(addr)
      if hit is None or hit.nbytes != nbytes:
        end = addr + nbytes
        for other in [e for e in cache.values() if e.addr < end and addr < e.addr + e.nbytes]:
          self._retire(other, True)
        # Fresh pk_alloc buffers (untouched != 0) cannot overlap live pinned host arrays
        if untouched[r] == 0:
          for pda in list(pinned.values()):
            paddr = pda._addr
            if paddr < end and addr < paddr + pda._nbytes:
              pda.sync_host()
              if r in written:
                pda._dev_buf = None
        dev_buf = self.upload(addr, nbytes, external[r], untouched[r])
        if untouched[r] == 0:
          self.uploads += 1
        hit = cache[addr] = _Resident(addr, nbytes, dev_buf)
        if untouched[r] == 4:
          # Host skipped memset (PK_LAZY_ZERO): device holds the only zeroed copy
          hit.dirty = True
      hit.dirty = hit.dirty or r in written
      out.append(hit.dev)
    return out

  def promote(self, arr: np.ndarray) -> DeviceArray:
    """Promote a freshly computed output ``arr`` into a persistent :class:`DeviceArray`
    without downloading it if its buffer is already resident in the call cache."""
    addr = arr.ctypes.data
    nbytes = arr.nbytes
    entry = self._state.cache.get(addr)
    if entry is not None and entry.nbytes >= nbytes and arr.flags.c_contiguous:
      self._state.cache.pop(addr, None)
      dev_buf = entry.dev
      if entry.nbytes > nbytes and hasattr(dev_buf, "ndim"):
        dev_buf = dev_buf[:nbytes]
      return DeviceArray(arr, self, dev_buf, dirty=entry.dirty)
    return DeviceArray(arr, self, None, dirty=False)

  def _retire(self, entry: _Resident, download: bool) -> None:
    self._state.cache.pop(entry.addr, None)
    if download and entry.dirty:
      self.download(entry.dev, entry.addr, entry.nbytes)
      self.downloads += 1

  def flush(self, mid_call: bool = True) -> None:
    """Bring the host up to date and forget non-persistent buffers on the device."""
    for entry in list(self._state.cache.values()):
      self._retire(entry, True)
    if mid_call:
      for da in self._state.pinned.values():
        if da._dirty and da._dev_buf is not None:
          da.sync_host()
        if da._host_addr in self._state.writable_pinned:
          # Host code may mutate this writable input before the next GPU kernel
          da._dev_buf = None

  def sync_buf(self, addr: int, nbytes: int, write: bool = False) -> None:
    """Bring only device buffers overlapping ``[addr, addr + nbytes)`` into host sync.

    When ``write`` is False (host read), dirty overlapping buffers are downloaded
    and kept clean on the device so later GPU reads need no re-upload.  When
    ``write`` is True (host write), overlapping device entries are also evicted
    so subsequent GPU kernels upload the host-modified data.  All non-overlapping
    live tensors remain resident on the GPU untouched.
    """
    if nbytes <= 0:
      return
    end = addr + nbytes
    cache = self._state.cache
    for entry in list(cache.values()):
      if entry.addr < end and addr < entry.addr + entry.nbytes:
        if entry.dirty:
          self.download(entry.dev, entry.addr, entry.nbytes)
          self.downloads += 1
          entry.dirty = False
        if write:
          cache.pop(entry.addr, None)
    for da in self._state.pinned.values():
      paddr = da._addr
      if paddr < end and addr < paddr + da._nbytes:
        if da._dirty and da._dev_buf is not None:
          da.sync_host()
        if write:
          da._dev_buf = None

  def forget(self, addr: int) -> None:
    """The host is about to free the memory at ``addr``."""
    entry = self._state.cache.get(addr)
    if entry is not None:
      self._retire(entry, False)

  def _launch(self, kernel: int, ctx_addr: int, total: int, nchunks: int) -> int:
    """Returns 0 if the loop ran (or failed for good); 1 to run it on the CPU.
    Kernel -1 flushes all buffers, -2 forgets ``ctx_addr``, -3 syncs a single
    buffer for host read, and -4 syncs+invalidates a single buffer for host write."""
    stack = self._state.stack
    if kernel < 0:
      try:
        if kernel == -1:
          self.flush()
        elif kernel == -2:
          self.forget(ctx_addr or 0)
        elif kernel == -3:
          self.sync_buf(ctx_addr or 0, total, write=False)
        elif kernel == -4:
          self.sync_buf(ctx_addr or 0, total, write=True)
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
