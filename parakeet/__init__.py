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

import numpy as _np

from . import errors, lib
from ._version import __version__
from .api import Specialization, available_backends, jit, run, seed
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
from .scalars import UFUNCS as _UFUNCS

pmap1 = jit(_pmap1)
pmap2 = jit(_pmap2)
pmap2_trim = jit(_pmap2_trim)

# Element-wise primitives usable as first-class functions, e.g. reduce(add, x)
add = _UFUNCS["add"]
subtract = _UFUNCS["sub"]
multiply = _UFUNCS["mul"]
divide = _UFUNCS["div"]
true_divide = _UFUNCS["div"]
floor_divide = _UFUNCS["floordiv"]
remainder = mod = _UFUNCS["mod"]
fmod = _UFUNCS["fmod"]
power = _UFUNCS["pow"]
negative = _UFUNCS["neg"]
minimum = _UFUNCS["min"]
maximum = _UFUNCS["max"]
equal = _UFUNCS["eq"]
not_equal = _UFUNCS["ne"]
less = _UFUNCS["lt"]
less_equal = _UFUNCS["le"]
greater = _UFUNCS["gt"]
greater_equal = _UFUNCS["ge"]
logical_and = _UFUNCS["logical_and"]
logical_or = _UFUNCS["logical_or"]
logical_not = _UFUNCS["logical_not"]
logical_xor = _UFUNCS["logical_xor"]
bitwise_and = _UFUNCS["bitand"]
bitwise_or = _UFUNCS["bitor"]
bitwise_xor = _UFUNCS["bitxor"]
bitwise_not = _UFUNCS["invert"]

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
    "rad2deg deg2rad logaddexp logaddexp2 isnan isinf isfinite"
).split():
    globals()[_name] = getattr(_np, _name)
del _name


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
]  # fmt: skip
