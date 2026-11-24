"""User-facing entry points: the ``@jit`` decorator and specialization cache."""

from __future__ import annotations

import builtins
import ctypes
import functools
import inspect
import linecache
import os
import threading
import types as pytypes
import warnings
import weakref
from typing import Any

import numpy as np

from .backends import blas
from .backends.device import DeviceArray
from .config import config, normalize_backend
from .errors import BackendUnavailable, CompileError, FallbackWarning, InternalError, ParakeetError
from .frontend import signature as sig
from .frontend.arrays import Restart
from .frontend.elab import Elaborated, Elaborator, ElabState, Options
from .ndtypes import ScalarType
from .syntax.ir import ArrayParam, fmt_function
from .target import TargetProfile

MAX_RESTARTS = 500
_U64 = 2**64
_I63 = 2**63

try:
  _byte_bounds = np.lib.array_utils.byte_bounds
except AttributeError:  # NumPy < 2
  _byte_bounds = np.byte_bounds  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# random seeds
# ---------------------------------------------------------------------------

_rng_lock = threading.Lock()
_rng_seed = int.from_bytes(os.urandom(8), "little")
_rng_calls = 0


def seed(value: int | None = None) -> None:
  """Seed the random number generator used by ``np.random`` in compiled code.

  Compiled code has its own counter-based generator (it must be usable from
  many threads and on GPUs), so streams differ from NumPy's, but are
  reproducible for a given seed and sequence of calls."""
  global _rng_seed, _rng_calls
  with _rng_lock:
    _rng_seed = int.from_bytes(os.urandom(8), "little") if value is None else int(value) % _U64
    _rng_calls = 0


def _next_rng_key() -> int:
  global _rng_calls
  with _rng_lock:
    _rng_calls += 1
    z = (_rng_seed + _rng_calls * 0x9E3779B97F4A7C15) % _U64
  z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) % _U64
  z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) % _U64
  z ^= z >> 31
  return z - _U64 if z >= _I63 else z


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


class _Memory:
  """Exposes native memory to NumPy and keeps its owner alive."""

  __slots__ = ("__array_interface__", "keep", "__weakref__")

  def __init__(self, interface: dict, keep: Any):
    self.__array_interface__ = interface
    self.keep = keep


def _safe_decref(decref, ptr: int) -> None:
  try:
    decref(ptr)
  except Exception:  # noqa: BLE001 - interpreter shutdown
    pass


_PY_EXC = {
  name: getattr(builtins, name)
  for name in (
    "ValueError TypeError IndexError ZeroDivisionError AssertionError RuntimeError MemoryError "
    "ArithmeticError OverflowError KeyError NotImplementedError FloatingPointError"
  ).split()
}


class Specialization:
  """One compiled version of a function for specific argument types."""

  def __init__(self, name: str, keys, elaborated: Elaborated, module, backend: Backend, type_key):
    self.name = name
    self.keys = keys
    self.type_key = type_key
    self.elaborated = elaborated
    self.module = module
    self.backend = backend
    self.result_spec = elaborated.result_spec
    self.implicits = elaborated.implicits
    self.function = elaborated.function
    if elaborated.writes_unknown:
      for p in self.function.params:
        if isinstance(p, ArrayParam):
          p.writes = True
    self.plan = module.params_plan

  # ---- introspection ---------------------------------------------------------

  @property
  def ir(self) -> str:
    """The typed loop IR of this specialization (after every pass), as text."""
    return fmt_function(self.function)

  @property
  def operators(self) -> str:
    """The function as the frontend produced it: array operators (map, reduce)
    over index spaces, before the passes of :mod:`parakeet.transforms.pipeline`."""
    return self.elaborated.operators

  @property
  def source(self) -> str:
    """The generated C source."""
    return self.module.source

  @property
  def kernels(self) -> list[str]:
    """Source of every device kernel (empty for CPU backends)."""
    dev = self.module.device
    return [] if dev is None else dev.kernel_sources()

  @property
  def signature(self) -> str:
    args = ", ".join(sig.describe(k) for k in self.keys[0])
    return f"{self.name}({args})"

  def __repr__(self) -> str:
    return f"<Specialization {self.signature} [{self.backend.name}]>"

  # ---- calling ---------------------------------------------------------------

  def run(self, leaves: list):
    mod = self.module
    backend = self.backend
    dev = backend._device
    ia: list[int] = backend.header()
    ia[5] = _next_rng_key()
    fa: list[float] = []
    arrays: list[np.ndarray] = []
    input_das: list[DeviceArray | None] = []
    pinned: list[tuple[DeviceArray, bool]] = []
    keep_on_device = False
    it = iter(leaves)
    for entry in self.plan:
      v = next(it)
      if entry[0] == "s":
        kind = entry[1].kind
        if kind == "f":
          fa.append(float(v))
        elif kind == "b":
          ia.append(1 if v else 0)
        else:
          iv = int(v)
          ia.append(iv - _U64 if iv >= _I63 else iv)
      else:
        _, elt, rank, strided, param = entry
        da: DeviceArray | None = None
        if isinstance(v, DeviceArray):
          if dev is not None and v._device is dev:
            da = v
            keep_on_device = True
            pinned.append((da, param.writes))
            arr = da._host
          else:
            arr = v.sync_host()
        else:
          arr = v
        if param.writes and not arr.flags.writeable:
          raise ValueError("assignment destination is read-only")
        arrays.append(arr)
        input_das.append(da)
        ptr = arr.ctypes.data
        if strided:
          low, high = _byte_bounds(arr)
          itemsize = elt.itemsize
          ia += [low, high - low, (ptr - low) // itemsize, *arr.shape]
          ia += [s // itemsize for s in arr.strides]
        else:
          ia += [ptr, arr.nbytes, 0, *arr.shape]
    n = len(ia)
    ia_c = (ctypes.c_int64 * n)(*ia)
    fa_c = (ctypes.c_double * (len(fa) or 1))(*fa)
    io = (ctypes.c_int64 * mod.n_io)()
    fo = (ctypes.c_double * mod.n_fo)()
    eo = (ctypes.c_int64 * 5)()
    token = backend.begin_call(self, arrays, pinned)
    decoded = None
    try:
      status = mod.entry(ia_c, fa_c, io, fo, eo)
      if status == 0 and keep_on_device and dev is not None and token is not None and token.pending is None:
        # Promote GPU-resident outputs to DeviceArray BEFORE end_call flushes the device cache
        decoded = self._decode(
          self.result_spec, iter(mod.results_plan), io, fo, arrays, promote_dev=dev, input_das=input_das
        )
    finally:
      pending = backend.end_call(token)
    if pending is not None:
      raise pending
    if status:
      raise self._error(eo)
    if decoded is not None:
      return decoded
    results = iter(mod.results_plan)
    return self._decode(self.result_spec, results, io, fo, arrays)

  def _error(self, eo) -> BaseException:
    eid = eo[0]
    if eid < 0:
      count, itemsize = eo[1], eo[2]
      return MemoryError(f"{self.name}: could not allocate an array of {count} elements of {itemsize} bytes")
    from .backends.device import make_runtime_error

    return make_runtime_error(self.module.errors[eid], [eo[i] for i in range(1, 5)])

  def _decode(self, spec, results, io, fo, arrays, promote_dev=None, input_das=None):
    tag = spec[0]
    if tag == "s":
      _, t, pos = next(results)
      return _scalar(t, fo[pos] if t.kind == "f" else io[pos])
    if tag == "a":
      _, elt, rank, pos = next(results)
      kind, ref, data, offset = io[pos], io[pos + 1], io[pos + 2], io[pos + 3]
      shape = tuple(io[pos + 4 : pos + 4 + rank])
      itemsize = elt.itemsize
      strides = tuple(s * itemsize for s in io[pos + 4 + rank : pos + 4 + 2 * rank])
      addr = data + offset * itemsize
      readonly = False
      if kind == 1:
        keep = arrays[ref]
        if keep.ctypes.data == addr and keep.shape == shape and keep.strides == strides:
          if promote_dev is not None and input_das is not None and input_das[ref] is not None:
            return input_das[ref]
          return keep
        readonly = not keep.flags.writeable
      else:
        keep = None
      mem = _Memory(
        {"data": (addr, readonly), "shape": shape, "strides": strides, "typestr": elt.dtype.str, "version": 3},
        keep,
      )
      if kind == 0:
        weakref.finalize(mem, _safe_decref, self.module.decref, ref)
      arr = np.asarray(mem)
      if promote_dev is not None:
        return promote_dev.promote(arr)
      return arr
    if tag == "t":
      return tuple(self._decode(s, results, io, fo, arrays, promote_dev, input_das) for s in spec[1:])
    if tag == "n":
      return None
    if tag == "static":
      return spec[1]
    if tag == "r":
      parts = [io[next(results)[2]] for _ in range(3)]
      return range(*parts)
    raise InternalError(f"unknown result spec {spec!r}")


def _scalar(t: ScalarType, raw):
  if t.weak:
    if t.kind == "b":
      return bool(raw)
    return float(raw) if t.kind == "f" else int(raw)
  if t.kind == "u" and raw < 0:
    raw += _U64
  if t.kind == "b":
    return np.bool_(raw != 0)
  return t.dtype.type(raw)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


class Backend:
  """How compiled code is executed: a threading runtime plus maybe a device."""

  def __init__(self, name: str, parallel: bool, threads: str | None, device_kind: str | None = None):
    self.name = name
    self.parallel = parallel
    self.thread_mode = threads
    self.device_kind = device_kind
    self._runtime = None
    self._device = None

  def runtime(self):
    if self._runtime is None:
      from .backends import toolchain

      if not self.parallel:
        self._runtime = toolchain.thread_runtime("serial")
      elif self.thread_mode is None:
        self._runtime = toolchain.best_thread_runtime()
      else:
        self._runtime = toolchain.thread_runtime(self.thread_mode)
    return self._runtime

  def device(self):
    if self.device_kind is None:
      return None
    if self._device is None:
      try:
        if self.device_kind == "metal":
          from .backends import metal

          self._device = metal.MetalDevice.get()
        else:
          from .backends import cuda

          self._device = cuda.CudaDevice.get()
      except ImportError as e:
        raise BackendUnavailable(f"the {self.device_kind} backend could not be loaded: {e}") from e
    return self._device

  def cache_key(self) -> tuple:
    return (self.parallel, self.device_kind)

  def header(self) -> list[int]:
    rt = self.runtime()
    nthreads = config.num_threads if self.parallel else 1
    dev = self._device
    prof = getattr(dev, "profile", None) if dev is not None else None
    rchunks = prof.reduce_chunks(config.device_reduce_chunks) if prof is not None else config.device_reduce_chunks
    return [
            nthreads, max(1, config.parallel_grain), rt.parfor,
            0 if dev is None else dev.callback_address, config.device_threshold, 0,
            rchunks, *blas.header(), 0, 0,
        ]  # fmt: skip

  def build(self, function, profile: TargetProfile | None = None):
    from .backends.c import CModule

    dev = self.device()
    planner = None if dev is None else dev.planner(profile)
    module = CModule(function, planner)
    if planner is not None:
      planner.report_fallbacks(function.name)
    return module

  def begin_call(self, spec: Specialization, arrays, pinned=None):
    dev = self._device
    if dev is None:
      return None
    return dev.begin_call(spec, arrays, pinned)

  def end_call(self, token):
    if token is None:
      return None
    return self._device.end_call(token)


_backends: dict[str, Backend] = {}
_backend_lock = threading.RLock()


def get_backend(name: str | None = None) -> Backend:
  name = normalize_backend(name or config.backend)
  b = _backends.get(name)
  if b is not None:
    return b
  with _backend_lock:
    if name == "c":
      b = Backend("c", False, None)
    elif name in ("auto", "multicore"):
      b = Backend(name, True, None)
    elif name in ("threads", "openmp", "gcd"):
      b = Backend(name, True, name)
    elif name in ("metal", "cuda"):
      b = Backend(name, True, None, name)
    else:
      raise ValueError(f"backend {name!r} does not compile code")
    b.runtime()  # fail fast, with an explanation, if it cannot work here
    b.device()
    _backends[name] = b
  return b


def available_backends() -> dict[str, str]:
  """Which backends work on this machine, with a reason for those that do not."""
  out: dict[str, str] = {"python": "ok"}
  for name in ("c", "threads", "gcd", "openmp", "metal", "cuda"):
    try:
      get_backend(name)
      out[name] = "ok"
    except ParakeetError as e:
      out[name] = "unavailable: " + str(e).splitlines()[0]
  return out


def target_profile(backend: str | None = None) -> TargetProfile:
  """Return the queried :class:`~parakeet.target.TargetProfile` for ``backend``
  (defaults to ``parakeet.config.backend``)."""
  bname = normalize_backend(backend or config.backend)
  if bname == "python":
    return TargetProfile.for_device(None, config.tile_bytes)
  b = get_backend(bname)
  return TargetProfile.for_device(
    b.device_kind,
    config.tile_bytes,
    device=b.device(),
    tensor_cores=config.tensor_cores or config.fast_math,
  )


def to_device(value, backend: str | None = None):
  """Move a NumPy array (or nested tuple of arrays, such as model parameters and
  optimizer state) onto a GPU backend as persistent :class:`DeviceArray`s.

  When ``backend`` is ``None``, uses ``parakeet.config.backend`` (or ``"gpu"``
  if the current backend is a CPU backend).
  """
  bname = normalize_backend(backend or (config.backend if config.backend in ("metal", "mps", "cuda") else "gpu"))
  b = get_backend(bname)
  dev = b.device()
  if dev is None:
    raise ValueError(f"to_device() requires a GPU backend ('metal' or 'cuda'), got {bname!r}")

  def _move(x):
    if isinstance(x, tuple):
      return tuple(_move(e) for e in x)
    if isinstance(x, DeviceArray):
      if x._device is dev:
        x.ensure_device()
        return x
      x = x.numpy()
    if isinstance(x, np.ndarray):
      arr = np.ascontiguousarray(x)
      da = DeviceArray(arr, dev)
      da.ensure_device()
      return da
    return x

  return _move(value)


def to_host(value):
  """Bring a :class:`DeviceArray` (or nested tuple of them) back to host NumPy arrays."""
  if isinstance(value, tuple):
    return tuple(to_host(e) for e in value)
  if isinstance(value, DeviceArray):
    return value.numpy()
  return value


# ---------------------------------------------------------------------------
# specialization tables
# ---------------------------------------------------------------------------


class _Table:
  __slots__ = ("specs", "static", "lock")

  def __init__(self) -> None:
    self.specs: dict[tuple, list[Specialization]] = {}
    self.static: dict[tuple, frozenset[int]] = {}
    self.lock = threading.RLock()


_tables: dict[pytypes.CodeType, _Table] = {}
_compile_lock = threading.RLock()


def _rewrite_keys(arg_keys, cell_keys, leaves, static: frozenset[int]):
  if not static:
    return arg_keys, cell_keys
  pos = [0]
  args2 = tuple(sig.with_static_leaves(k, leaves, static, pos) for k in arg_keys)
  cells2 = tuple((n, sig.with_static_leaves(k, leaves, static, pos)) for n, k in cell_keys)
  return args2, cells2


def _passed_leaves(key, leaves: list, pos: list[int], out: list) -> None:
  """Append the leaves that are actual runtime parameters (not constants)."""
  tag = key[0]
  if tag == "s" or tag == "a":
    out.append(leaves[pos[0]])
    pos[0] += 1
  elif tag == "c":
    pos[0] += 1
  elif tag == "r":
    out.extend(leaves[pos[0] : pos[0] + 3])
    pos[0] += 3
  elif tag in ("t", "sl"):
    for k in key[1:]:
      _passed_leaves(k, leaves, pos, out)
  elif tag == "fn" and isinstance(key[3], tuple):
    for _, k in key[3]:
      _passed_leaves(k, leaves, pos, out)


class jit:  # noqa: N801 - decorator, lower-case by convention
  """Compile a numerical Python function to native code on first call.

  Use it bare (``@jit``) or with options (``@jit(backend="metal")``).  The
  function is specialised for each distinct combination of argument types;
  every later call with the same types runs the cached native code.

  Options (all default to the corresponding ``parakeet.config`` setting):

  ``backend``
      ``"auto"``, ``"c"``, ``"multicore"``, ``"threads"``, ``"openmp"``,
      ``"gcd"``, ``"metal"``/``"mps"``, ``"cuda"`` or ``"python"``.
  ``bounds_check``
      Raise ``IndexError`` for out-of-range indices instead of crashing.
  ``fusion``
      Fuse array expressions into single loops.
  ``tiling``
      Tile matmul-shaped loops for cache locality.
  """

  __parakeet_decorator__ = True

  def __new__(cls, fn=None, **options):
    if fn is None:
      return functools.partial(cls, **options)
    if (not isinstance(fn, pytypes.FunctionType) or fn in sig.STAGED) and not hasattr(fn, "__parakeet_fn__"):
      # a NumPy function, builtin, dtype or Parakeet library function
      return StagedFunction(fn, **options)
    return super().__new__(cls)

  def __init__(
    self,
    fn=None,
    *,
    backend: str | None = None,
    bounds_check: bool | None = None,
    fusion: bool | None = None,
    tiling: bool | None = None,
  ):
    if hasattr(fn, "__parakeet_fn__"):
      fn = fn.__parakeet_fn__
    if not isinstance(fn, pytypes.FunctionType):
      raise TypeError(
        f"@jit expects a Python function, got {type(fn).__name__}"
        + (" (use it on the underlying function of a method or partial)" if callable(fn) else "")
      )
    self.fn = fn
    self.__parakeet_fn__ = fn
    self.backend = None if backend is None else normalize_backend(backend)
    self.bounds_check = bounds_check
    self.fusion = fusion
    self.tiling = tiling
    functools.update_wrapper(self, fn)
    self._signature: inspect.Signature | None = None
    self._simple: tuple | None = None
    code = fn.__code__
    simple = not (code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)) and not code.co_kwonlyargcount
    self._names = code.co_varnames[: code.co_argcount] if simple else None

  def __repr__(self) -> str:
    return f"<parakeet.jit {self.fn.__module__}.{self.fn.__qualname__}>"

  def __get__(self, obj, objtype=None):
    if obj is None:
      return self
    raise TypeError(
      f"{self.fn.__qualname__} is a @jit function used as a method; compiled functions cannot "
      "take 'self' - make it a module-level function or a @staticmethod"
    )

  # ---- argument binding ---------------------------------------------------------

  def _bind(self, args: tuple, kwargs: dict) -> tuple[tuple[str, ...], list]:
    names = self._names
    if names is not None and not kwargs and len(args) == len(names):
      return names, list(args)
    if self._signature is None:
      self._signature = inspect.signature(self.fn)
    try:
      bound = self._signature.bind(*args, **kwargs)
    except TypeError as e:
      raise TypeError(f"{self.fn.__name__}() {e}") from None
    bound.apply_defaults()
    for p in self._signature.parameters.values():
      if p.kind is p.VAR_KEYWORD:
        if bound.arguments.get(p.name):
          raise TypeError(f"{self.fn.__name__}() got unsupported **{p.name} arguments")
        bound.arguments.pop(p.name, None)
    return tuple(bound.arguments), list(bound.arguments.values())

  def _options(self) -> Options:
    return Options(
      bounds_check=config.bounds_check if self.bounds_check is None else self.bounds_check,
      parallel=True,
      fusion=config.fusion if self.fusion is None else self.fusion,
      tiling=config.tiling if self.tiling is None else self.tiling,
      tile_bytes=config.tile_bytes,
      blas=config.blas,
      skip=tuple(sorted(config.skip_transforms)),
    )

  # ---- calling -------------------------------------------------------------------

  def __call__(self, *args, **kwargs):
    backend_name = kwargs.pop("_backend", None) if kwargs else None
    backend_name = backend_name or self.backend or config.backend
    if backend_name == "python" or normalize_backend(backend_name) == "python":
      return self.fn(*args, **kwargs)
    spec, leaves, fixups = self._prepare(args, kwargs, backend_name)
    result = spec.run(leaves)
    for original, converted in fixups:
      if original.flags.writeable:
        np.copyto(original, converted, casting="unsafe")
    return result

  def _prepare(self, args: tuple, kwargs: dict, backend_name: str | None = None):
    names, values = self._bind(args, kwargs)
    leaves: list = []
    fixups: list = []
    try:
      arg_keys = tuple([sig.classify(v, leaves, fixups) for v in values])
      fn_key = sig.classify(self.fn, leaves, fixups)
    except sig.Unsupported as e:
      raise TypeError(f"{self.fn.__name__}(): {e}") from None
    cell_keys = fn_key[3] if isinstance(fn_key[3], tuple) else ()
    code = self.fn.__code__
    table = _tables.get(code)
    if table is None:
      table = _tables.setdefault(code, _Table())
    backend = get_backend(backend_name or self.backend)
    options = self._options()
    options.parallel = backend.parallel
    options.blas = options.blas and backend.device_kind is None and blas.available()
    options.target = TargetProfile.for_device(
      backend.device_kind,
      options.tile_bytes,
      device=backend.device(),
      tensor_cores=config.tensor_cores or config.fast_math,
    )
    toolchain_key = (config.cc, config.opt_level, config.fast_math, tuple(config.cflags))
    variant = (backend.cache_key(), options.key(), names, toolchain_key)
    type_key = (variant, arg_keys, cell_keys)
    static = table.static.get(type_key, frozenset())
    full_key = (variant, *_rewrite_keys(arg_keys, cell_keys, leaves, static))
    candidates = table.specs.get(full_key)
    if candidates:
      for spec in candidates:
        passed = self._match(spec, leaves, fixups)
        if passed is not None:
          return spec, passed, fixups
    with _compile_lock:
      # another thread may have compiled this specialization while we waited
      static = table.static.get(type_key, frozenset())
      full_key = (variant, *_rewrite_keys(arg_keys, cell_keys, leaves, static))
      for spec in table.specs.get(full_key, ()):
        passed = self._match(spec, leaves, fixups)
        if passed is not None:
          return spec, passed, fixups
      spec = self._compile(table, type_key, names, arg_keys, cell_keys, leaves, backend, options)
    passed = self._match(spec, leaves, fixups)
    if passed is None:
      raise InternalError(f"freshly compiled specialization of {self.fn.__name__} does not match its arguments")
    return spec, passed, fixups

  @staticmethod
  def _match(spec: Specialization, leaves: list, fixups: list) -> list | None:
    """Runtime parameter list for ``spec``, or None if a guard on a global
    or closure variable no longer holds."""
    passed: list = []
    pos = [0]
    for k in spec.keys[0]:
      _passed_leaves(k, leaves, pos, passed)
    for _, k in spec.keys[1]:
      _passed_leaves(k, leaves, pos, passed)
    for imp in spec.implicits:
      try:
        obj = imp.getter()
        lv: list = []
        k = sig.classify(obj, lv, fixups)
      except Exception:  # noqa: BLE001 - a vanished global is just a failed guard
        return None
      if imp.static:
        k = sig.with_static_leaves(k, lv, imp.static)
      if k != imp.type_key:
        return None
      _passed_leaves(k, lv, [0], passed)
    return passed

  def _compile(self, table, type_key, names, arg_keys, cell_keys, leaves, backend, options) -> Specialization:
    state = ElabState(static_leaves=set(table.static.get(type_key, ())))
    while True:
      static = frozenset(state.static_leaves)
      keys = _rewrite_keys(arg_keys, cell_keys, leaves, static)
      try:
        elaborated = Elaborator(self.fn, (names, *keys), state, options).run()
        break
      except Restart:
        state.restarts += 1
        if state.restarts > MAX_RESTARTS:
          raise InternalError(
            f"type inference for {self.fn.__name__} did not converge after {MAX_RESTARTS} attempts"
          ) from None
      except CompileError as e:
        e.add_note(f"while compiling {self.fn.__qualname__}({', '.join(sig.describe(k) for k in arg_keys)})")
        raise
    from .transforms import pipeline

    elaborated.operators = fmt_function(elaborated.function)  # before the passes: the operator level
    if config.print_operators or config.print_ir:
      print(elaborated.operators)
    trace = (lambda name, fn: print(f"--- after {name} ---\n{fmt_function(fn)}")) if config.print_ir else None
    pipeline.run(elaborated.function, options, trace)
    module = backend.build(elaborated.function, options.target)
    spec = Specialization(self.fn.__name__, keys, elaborated, module, backend, type_key)
    with table.lock:
      table.static[type_key] = static
      full_key = (type_key[0], *keys)
      table.specs.setdefault(full_key, []).insert(0, spec)
    return spec

  # ---- introspection ---------------------------------------------------------------

  def trace_passes(self, *args, _backend: str | None = None, _trace=None, **kwargs):
    """Elaborate and run the middle-end optimization passes for ``args`` without
    building or executing a kernel and without consulting or mutating the
    specialization cache.

    ``_trace(stage_name, ir_fn)`` is invoked with ``"before passes"`` prior to
    the pipeline and after every pass in :data:`parakeet.transforms.pipeline.PASSES`.
    """
    names, values = self._bind(args, kwargs)
    leaves: list = []
    fixups: list = []
    try:
      arg_keys = tuple([sig.classify(v, leaves, fixups) for v in values])
      fn_key = sig.classify(self.fn, leaves, fixups)
    except sig.Unsupported as e:
      raise TypeError(f"{self.fn.__name__}(): {e}") from None
    cell_keys = fn_key[3] if isinstance(fn_key[3], tuple) else ()
    backend = get_backend(_backend or self.backend)
    options = self._options()
    options.parallel = backend.parallel
    options.blas = options.blas and backend.device_kind is None and blas.available()
    options.target = TargetProfile.for_device(
      backend.device_kind,
      options.tile_bytes,
      device=backend.device(),
      tensor_cores=config.tensor_cores or config.fast_math,
    )
    state = ElabState()
    while True:
      static = frozenset(state.static_leaves)
      keys = _rewrite_keys(arg_keys, cell_keys, leaves, static)
      try:
        elaborated = Elaborator(self.fn, (names, *keys), state, options).run()
        break
      except Restart:
        state.restarts += 1
        if state.restarts > MAX_RESTARTS:
          raise InternalError(
            f"type inference for {self.fn.__name__} did not converge after {MAX_RESTARTS} attempts"
          ) from None
      except CompileError as e:
        e.add_note(f"while compiling {self.fn.__qualname__}({', '.join(sig.describe(k) for k in arg_keys)})")
        raise
    from .transforms import pipeline

    if _trace is not None:
      _trace("before passes", elaborated.function)
    pipeline.run(elaborated.function, options, _trace)
    return elaborated.function

  def specialize(self, *args, **kwargs) -> Specialization:
    """Compile (without running) for the given example arguments."""
    backend_name = kwargs.pop("_backend", None)
    spec, _, _ = self._prepare(args, kwargs, backend_name)
    return spec

  @property
  def specializations(self) -> list[Specialization]:
    table = _tables.get(self.fn.__code__)
    if table is None:
      return []
    return [s for specs in table.specs.values() for s in specs]

  def inspect(self, *args, **kwargs) -> None:
    """Print the typed IR generated for the given example arguments."""
    print(self.specialize(*args, **kwargs).ir)


class StagedFunction:
  """``jit`` applied to something that is not a Python function but that
  Parakeet knows how to compile: ``jit(np.sum)``, ``jit(len)``, ``jit(np.float32)``."""

  def __init__(self, target, **options):
    from . import lib
    from .prims import UFunc

    supported = isinstance(target, UFunc) or lib.lookup(target) is not None
    if not supported:
      try:
        supported = sig.type_value(target) is not None
      except sig.Unsupported:
        supported = False
    if not supported:
      if callable(target):
        raise TypeError(f"@jit: {lib.unsupported_message(target)}")
      raise TypeError(f"@jit expects a function, got {type(target).__name__}")
    self.target = target
    self.options = options
    self.__name__ = getattr(target, "__name__", "function")
    self.__doc__ = getattr(target, "__doc__", None)

  def __repr__(self) -> str:
    return f"<parakeet.jit {self.__name__}>"

  def __call__(self, *args, **kwargs):
    backend = kwargs.pop("_backend", None)
    return call_staged(self.target, args, kwargs, backend, self.options)


def run(fn, *args, **kwargs):
  """Compile ``fn`` (if needed) and call it: ``parakeet.run(f, x, y)``."""
  return jit(fn)(*args, **kwargs)


def _as_jit(fn) -> jit:
  return fn if isinstance(fn, jit) else jit(fn)


def run_python_fn(fn, args, kwargs=None, backend=None):
  """Compile a plain Python function and call it with a list of arguments."""
  return _as_jit(fn)(*args, _backend=backend, **(kwargs or {}))


def specialize(fn, args, kwargs=None, backend=None) -> Specialization:
  """Compile ``fn`` for the types of the given arguments, without running it."""
  return _as_jit(fn).specialize(*args, _backend=backend, **(kwargs or {}))


def typed_repr(fn, args, kwargs=None) -> str:
  """The typed code Parakeet generates for ``fn`` applied to ``args``, as text."""
  return specialize(fn, args, kwargs).ir


def _openmp_available() -> bool:
  return available_backends().get("openmp") == "ok"


# ---------------------------------------------------------------------------
# calling library functions directly from Python
# ---------------------------------------------------------------------------

_wrappers: dict[tuple, jit] = {}


def staged_wrapper(libfn, args: tuple, kwargs: dict, options: dict | None = None) -> jit:
  """Return the cached :class:`jit` wrapper used to stage ``libfn`` for this call shape."""
  kwnames = tuple(sorted(kwargs))
  try:
    key = (libfn, len(args), kwnames, tuple(sorted((options or {}).items())))
    hash(key)
  except TypeError:
    key = (id(libfn), len(args), kwnames, tuple(sorted((options or {}).items())))
  wrapper = _wrappers.get(key)
  if wrapper is None:
    params = [f"a{i}" for i in range(len(args))]
    call = params + [f"{k}={k}" for k in kwnames]
    name = "".join(c if c.isalnum() else "_" for c in getattr(libfn, "__name__", "function"))
    src = f"def {name}({', '.join(params + list(kwnames))}):\n    return __fn__({', '.join(call)})\n"
    filename = f"<parakeet.{name}/{len(args)}{'/' + ','.join(kwnames) if kwnames else ''}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    ns: dict[str, Any] = {"__fn__": libfn}
    exec(compile(src, filename, "exec"), ns)  # noqa: S102 - fixed template, not user input
    wrapper = _wrappers[key] = jit(ns[name], **(options or {}))
  return wrapper


def call_staged(libfn, args: tuple, kwargs: dict, backend: str | None = None, options: dict | None = None):
  """Run a Parakeet library function (``parakeet.map`` ...) from Python by
  compiling a small wrapper for this call shape."""
  wrapper = staged_wrapper(libfn, args, kwargs, options)
  if backend is not None:
    kwargs = {**kwargs, "_backend": backend}
  return wrapper(*args, **kwargs)


def warn_fallback(message: str) -> None:
  if config.device_fallback == "error":
    raise BackendUnavailable(message)
  if config.device_fallback == "warn":
    warnings.warn(message, FallbackWarning, stacklevel=3)


def __getattr__(name: str):
  if name == "openmp_available":  # probing needs a compiler run, so do it lazily
    return _openmp_available()
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
