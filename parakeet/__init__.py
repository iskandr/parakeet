"""Parakeet: a runtime compiler for numerical Python.

Decorate an array-oriented function with :func:`jit` and Parakeet compiles a
type-specialised native version the first time it is called::

    import numpy as np
    from parakeet import jit

    @jit
    def scaled_tanh(x, alpha=0.5, beta=0.3):
        return np.tanh(x * alpha) + beta

Array expressions are fused into single loops and run on every core (or on a
GPU with ``backend="metal"`` / ``backend="cuda"``).
"""

from __future__ import annotations

import math as _math

import numpy as _np

from . import errors, lib, ndtypes, prims
from ._version import __version__
from .api import (
  Specialization,
  available_backends,
  jit,
  run,
  run_python_fn,
  seed,
  specialize,
  typed_repr,
)
from .config import config, options
from .errors import (
  BackendError,
  BackendUnavailable,
  CompileError,
  FallbackWarning,
  ParakeetError,
  StaticValueError,
  TypeInferenceError,
  UnsupportedError,
)
from .lib.adverbs import (
  allpairs,
  each,
  filter,  # noqa: A004
  imap,
  ireduce,
  map,  # noqa: A004
  outer_map,
  parfor,
  reduce,
  scan,
)
from .lib.pure import pmap1 as _pmap1
from .lib.pure import pmap2 as _pmap2
from .lib.pure import pmap2_trim as _pmap2_trim
from .ndtypes import (
  Bool,
  Float32,
  Float64,
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
)

pmap1 = jit(_pmap1)
pmap2 = jit(_pmap2)
pmap2_trim = jit(_pmap2_trim)

# Element-wise primitives usable as first-class functions, e.g. reduce(add, x)
from .prims import (  # noqa: E402
  add,
  bitwise_and,
  bitwise_not,
  bitwise_or,
  bitwise_xor,
  divide,
  equal,
  floor_divide,
  fmod,
  greater,
  greater_equal,
  less,
  less_equal,
  logical_and,
  logical_not,
  logical_or,
  logical_xor,
  maximum,
  minimum,
  mod,
  multiply,
  negative,
  not_equal,
  power,
  remainder,
  subtract,
  true_divide,
)

# NumPy functions and dtypes re-exported for convenience (and compatibility
# with earlier versions): inside compiled code they are the compiled versions,
# outside they are plain NumPy.
for _name in (
  "abs sqrt exp exp2 expm1 log log2 log10 log1p sin cos tan arcsin arccos arctan arctan2 sinh cosh "
  "tanh arcsinh arccosh arctanh floor ceil trunc rint round square sign reciprocal hypot "
  "zeros ones empty full zeros_like ones_like empty_like full_like arange linspace eye identity "
  "array asarray copy sum prod mean std var min max argmin argmax any all cumsum cumprod dot vdot "
  "inner outer transpose ravel reshape where clip diff concatenate stack vstack hstack "
  "int8 int16 int32 int64 uint8 uint16 uint32 uint64 float32 float64 bool_ "
  "rad2deg deg2rad logaddexp logaddexp2 isnan isinf isfinite "
  "tile real conjugate shape size ndim"
).split():
  globals()[_name] = getattr(_np, _name)
del _name
rank = _np.ndim  # the name Parakeet has always used for the number of dimensions
norm = _np.linalg.norm
erf, erfc = _math.erf, _math.erfc


def __getattr__(name: str):
  if name == "openmp_available":  # finding out needs the C compiler, so only on request
    from . import api

    return api.openmp_available
  raise AttributeError(f"module 'parakeet' has no attribute {name!r}")


def clear_cache() -> int:
  """Remove all compiled modules from the on-disk cache."""
  from .backends import toolchain

  return toolchain.clear_cache()


__all__ = [
    "BackendError", "BackendUnavailable", "CompileError", "FallbackWarning", "ParakeetError",
    "Specialization", "StaticValueError", "TypeInferenceError", "UnsupportedError", "__version__",
    "allpairs", "available_backends", "clear_cache", "config", "each", "errors", "filter", "imap",
    "ireduce", "jit", "lib", "map", "options", "outer_map", "parfor", "pmap1", "pmap2",
    "pmap2_trim", "reduce", "run", "scan", "seed",
    "ndtypes", "openmp_available", "prims", "run_python_fn", "specialize", "typed_repr",
    "Bool", "Float32", "Float64", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
]  # fmt: skip
