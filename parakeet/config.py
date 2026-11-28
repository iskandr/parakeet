"""Global configuration.

Change settings by assigning attributes of ``parakeet.config``, temporarily
with ``with parakeet.options(backend="c"): ...``, per function with
``@jit(backend=..., bounds_check=...)`` or through ``PARAKEET_*`` environment
variables (read once at import).
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, field, fields

BACKENDS = ("auto", "c", "multicore", "threads", "openmp", "gcd", "metal", "mps", "cuda", "python")


def _env_bool(name: str, default: bool) -> bool:
  v = os.environ.get(name)
  if v is None:
    return default
  return v.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
  try:
    return int(os.environ[name])
  except (KeyError, ValueError):
    return default


def default_tile_bytes() -> int:
  """Half of the per-core L2 cache, within sane limits (256 KiB if unknown)."""
  l2 = 0
  try:
    if sys.platform == "darwin":
      import subprocess

      for key in ("hw.perflevel0.l2cachesize", "hw.l2cachesize"):
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip()
        if out.isdigit() and int(out) > 0:
          l2 = int(out)
          break
    elif sys.platform.startswith("linux"):
      with open("/sys/devices/system/cpu/cpu0/cache/index2/size") as f:
        text = f.read().strip().upper()
      scale = {"K": 1024, "M": 1024 * 1024}.get(text[-1:], 1)
      l2 = int(text.rstrip("KM")) * scale
  except (OSError, ValueError, ImportError):
    l2 = 0
  if l2 <= 0:
    return 256 * 1024
  return min(max(l2 // 2, 64 * 1024), 1024 * 1024)


@dataclass
class Config:
  #: "auto" (best multi-core strategy), "c" (single-threaded), "multicore",
  #: "threads", "openmp", "gcd", "metal" (alias "mps"), "cuda", or "python"
  #: (run the original function uncompiled, for debugging).
  backend: str = field(default_factory=lambda: os.environ.get("PARAKEET_BACKEND", "auto"))
  #: Check array indices (IndexError instead of memory corruption).
  bounds_check: bool = field(default_factory=lambda: _env_bool("PARAKEET_BOUNDS_CHECK", True))
  #: Fuse array expressions into single loops.
  fusion: bool = field(default_factory=lambda: _env_bool("PARAKEET_FUSION", True))
  #: Tile the loops of matmul-shaped operations (a map over pairs of rows whose
  #: element is a reduction) so each tile of operands stays in cache.
  tiling: bool = field(default_factory=lambda: _env_bool("PARAKEET_TILING", True))
  #: Bytes of operand data one tile may span (default: half of a core's L2 cache).
  tile_bytes: int = field(default_factory=lambda: _env_int("PARAKEET_TILE_BYTES", 0) or default_tile_bytes())
  #: Names of middle-end passes to leave out (see ``parakeet.transforms.pipeline.names()``).
  skip_transforms: tuple = field(
    default_factory=lambda: tuple(
      s.strip() for s in os.environ.get("PARAKEET_SKIP_TRANSFORMS", "").split(",") if s.strip()
    )
  )
  #: Call the BLAS that NumPy uses for float matrix products (CPU backends).
  #: Off: every product is Parakeet's own generated, tiled loop nest.
  blas: bool = field(default_factory=lambda: _env_bool("PARAKEET_BLAS", True))
  #: Worker threads for multi-core backends (default: all cores).
  num_threads: int = field(default_factory=lambda: _env_int("PARAKEET_NUM_THREADS", os.cpu_count() or 1))
  #: Minimum number of cheap loop iterations per chunk before threads are used.
  parallel_grain: int = field(default_factory=lambda: _env_int("PARAKEET_PARALLEL_GRAIN", 8192))
  #: Minimum loop size before a loop is sent to a GPU.
  device_threshold: int = field(default_factory=lambda: _env_int("PARAKEET_DEVICE_THRESHOLD", 32768))
  #: Number of partial results used for reductions on a GPU.
  device_reduce_chunks: int = 8192
  #: What to do when a loop cannot run on the requested GPU: "warn" (run on
  #: the CPU and warn once), "silent" (run on the CPU) or "error".
  device_fallback: str = field(default_factory=lambda: os.environ.get("PARAKEET_DEVICE_FALLBACK", "warn"))
  #: C compiler executable (default: $PARAKEET_CC, $CC, then cc/gcc/clang).
  cc: str | None = field(default_factory=lambda: os.environ.get("PARAKEET_CC") or os.environ.get("CC"))
  #: Extra flags appended to the C compiler command line.
  cflags: tuple[str, ...] = ()
  #: Optimisation level passed to the C compiler.
  opt_level: int = 3
  #: Allow value-changing floating point optimisations (-ffast-math).
  fast_math: bool = False
  #: Use TF32 Tensor Cores (nvcuda::wmma) for float32 matmul on CUDA sm_80+ GPUs.
  #: (Apple Metal float32 simdgroup_matrix is full IEEE FP32 and enabled automatically.)
  tensor_cores: bool = field(default_factory=lambda: _env_bool("PARAKEET_TENSOR_CORES", False))
  #: CUDA kernel compiler: "nvrtc" (default, zero-header NVRTC + CUBIN cache),
  #: "ptx" (direct PTX 7.8 + NVRTC fallback), or "nvcc" (standalone nvcc -O3).
  cuda_compiler: str = field(
    default_factory=lambda: os.environ.get(
      "PARAKEET_CUDA_COMPILER", "ptx" if _env_bool("PARAKEET_CUDA_PTX", False) else "nvrtc"
    )
  )
  #: Specialize tiled/GPU schedules on finite shape buckets & power-of-2
  #: divisibility (on demand, guarded at call time without recompiling
  #: unrelated dimensions or large aligned shapes).
  shape_specialization: bool = field(default_factory=lambda: _env_bool("PARAKEET_SHAPE_SPECIALIZATION", True))
  #: Where compiled modules are cached (default: per-user cache directory).
  cache_dir: str | None = field(default_factory=lambda: os.environ.get("PARAKEET_CACHE_DIR"))
  #: Print the operator IR / typed loop IR / generated source of everything that gets compiled.
  print_operators: bool = field(default_factory=lambda: _env_bool("PARAKEET_PRINT_OPERATORS", False))
  print_ir: bool = field(default_factory=lambda: _env_bool("PARAKEET_PRINT_IR", False))
  print_code: bool = field(default_factory=lambda: _env_bool("PARAKEET_PRINT_CODE", False))
  #: Run the IR verifier before passes, after every pass, and after lowering.
  verify: bool = field(default_factory=lambda: _env_bool("PARAKEET_VERIFY", True))

  def __getattr__(self, name: str):
    # only reached for names that are not settings: the option names of Parakeet 0.x
    if name in LEGACY_NAMES:
      return getattr(self, LEGACY_NAMES[name])
    raise AttributeError(f"parakeet.config has no setting {name!r}")

  def __setattr__(self, name: str, value) -> None:
    name = LEGACY_NAMES.get(name, name)
    if name not in _FIELD_NAMES:
      raise AttributeError(f"parakeet.config has no setting {name!r} (settings: {', '.join(sorted(_FIELD_NAMES))})")
    if name == "backend":
      value = normalize_backend(value)
    if name == "device_fallback" and value not in ("warn", "silent", "error"):
      raise ValueError("device_fallback must be 'warn', 'silent' or 'error'")
    if name == "cuda_compiler" and value not in ("ptx", "nvrtc", "nvcc"):
      raise ValueError("cuda_compiler must be 'ptx', 'nvrtc' or 'nvcc'")
    if name == "num_threads":
      value = max(1, int(value))
    if name == "skip_transforms":
      value = (value,) if isinstance(value, str) else tuple(value)
    if name == "tile_bytes":
      value = max(1024, int(value))
    object.__setattr__(self, name, value)

  def as_dict(self) -> dict:
    return {f.name: getattr(self, f.name) for f in fields(self)}


# option names used by Parakeet 0.x, still accepted
LEGACY_NAMES = {
  "opt_fusion": "fusion",
  "opt_verify": "verify",
  "print_indexified_function": "print_operators",
  "print_specialized_function": "print_ir",
  "print_loopy_function": "print_ir",
  "print_lowered_function": "print_ir",
  "print_generated_code": "print_code",
  "compiler_path": "cc",
}

_FIELD_NAMES = frozenset(f.name for f in fields(Config))


def normalize_backend(name: str) -> str:
  if not isinstance(name, str):
    raise TypeError(f"backend must be a string, got {type(name).__name__}")
  key = name.strip().lower()
  aliases = {"mps": "metal", "seq": "c", "serial": "c", "interp": "python", "pthreads": "threads"}
  if key == "gpu":
    key = "metal" if sys.platform == "darwin" else "cuda"
  key = aliases.get(key, key)
  if key not in BACKENDS:
    raise ValueError(f"unknown backend {name!r}; choose one of: {', '.join(BACKENDS)}")
  return key


config = Config()


@contextlib.contextmanager
def options(**settings):
  """Temporarily override configuration settings.

  >>> with parakeet.options(backend="c", bounds_check=False):
  ...     result = f(x)
  """
  saved = {k: getattr(config, k) for k in settings}
  try:
    for k, v in settings.items():
      setattr(config, k, v)
    yield config
  finally:
    for k, v in saved.items():
      setattr(config, k, v)
