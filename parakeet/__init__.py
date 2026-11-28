"""Parakeet: a runtime compiler for numerical Python.

Decorate an array-oriented function with :func:`jit` and Parakeet compiles a
type-specialised native version the first time it is called::

    import numpy as np
    import parakeet as par

    @par.jit
    def scaled_tanh(x, alpha=0.5, beta=0.3):
        return np.tanh(x * alpha) + beta

Array expressions are fused into single loops and run on every core (or on a
GPU with ``backend="metal"`` / ``backend="cuda"``).
"""

from __future__ import annotations

import math as _math

import numpy as _np

from . import dsl, errors, frontend, lib, ndtypes, nn, optim, prims, syntax, transforms
from ._version import __version__
from .api import (
  DeviceArray,
  Specialization,
  TargetProfile,
  available_backends,
  jit,
  run,
  run_python_fn,
  seed,
  specialize,
  target_profile,
  to_device,
  to_host,
  typed_repr,
)
from .config import config, options
from .dsl import RewriteReport, show_rewrite
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
from .frontend.decorators import macro
from .frontend.diagnose import find_broken_transform
from .index import FrozenHashTable, SortedIndex
from .lib.adverbs import (
  allpairs,
  checkpoint,
  custom_jvp,
  custom_vjp,
  each,
  filter,  # noqa: A004
  filter_reduce,
  fixed_list,
  fold,
  grad,
  hessian,
  hvp,
  imap,
  ipa_attention,
  ireduce,
  jacfwd,
  jacrev,
  jvp,
  jvp_for,
  linear_transpose,
  linearize,
  map,  # noqa: A004
  outer_map,
  parfor,
  reduce,
  scan,
  scatter_add,
  stop_gradient,
  tanimoto_topk,
  topk,
  topk_fold,
  value_and_grad,
  vjp,
  vjp_for,
)
from .lib.bytes_lib import parse_float32, parse_float64, parse_int64, split_bytes
from .lib.expand_lib import filter_pairs, flatmap
from .lib.pure import pmap1 as _pmap1
from .lib.pure import pmap2 as _pmap2
from .lib.pure import pmap2_trim as _pmap2_trim
from .lib.ragged_lib import (
  partition_by_size,
  ragged_partition,
  segment_ids,
  segmented_histogram2d,
  segmented_map,
  segmented_parfor,
  segmented_reduce,
  segmented_scan,
  segmented_softmax,
  varlen_flash_attention,
)
from .lib.spatial_lib import block_sparse_attention, morton_order, spatial_cutoff_sum
from .lib.wavefront_lib import batched_wavefront, soft_align, wavefront
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
from .ragged import RaggedArray
from .spatial import BlockMask
from .target import MatrixUnit
from .transforms.verify import verify
from .tree import tree_array_leaves, tree_leaves, tree_map, tree_map_arrays
from .vocab import Alphabet, Vocab

pmap1 = jit(_pmap1)
pmap2 = jit(_pmap2)
pmap2_trim = jit(_pmap2_trim)
scatter = scatter_add
iscan = scan
iscatter = scatter_add
ifilter_reduce = filter_reduce

# Element-wise primitives usable as first-class functions, e.g. reduce(add, x)
from .prims import (  # noqa: E402
  add,
  bf16_to_f32,
  bitwise_and,
  bitwise_not,
  bitwise_or,
  bitwise_xor,
  bswap,
  clz,
  ctz,
  divide,
  equal,
  f32_to_bf16,
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
  popcount,
  power,
  remainder,
  rotl,
  rotr,
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
  "inner outer transpose ravel reshape where clip diff concatenate stack vstack hstack bincount "
  "sort argsort unique searchsorted "
  "int8 int16 int32 int64 uint8 uint16 uint32 uint64 float32 float64 complex64 complex128 bool_ "
  "rad2deg deg2rad logaddexp logaddexp2 isnan isinf isfinite "
  "tile real imag conj angle conjugate shape size ndim"
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
    "Alphabet", "BackendError", "BackendUnavailable", "BlockMask", "CompileError", "DeviceArray", "FallbackWarning",
    "FrozenHashTable", "MatrixUnit", "ParakeetError", "RaggedArray", "RewriteReport", "SortedIndex",
    "Specialization", "StaticValueError", "TargetProfile",
    "TypeInferenceError", "UnsupportedError", "Vocab", "__version__",
    "allpairs", "available_backends", "batched_wavefront", "bf16_to_f32", "block_sparse_attention",
    "bswap", "checkpoint", "clear_cache", "clz", "config", "ctz", "custom_jvp", "custom_vjp", "each", "errors",
    "f32_to_bf16", "filter", "filter_pairs", "filter_reduce", "fixed_list", "flatmap", "fold", "hessian", "hvp", "ifilter_reduce",
    "imap", "ipa_attention", "ireduce", "iscan", "iscatter", "jacfwd", "jacrev", "jit", "jvp", "jvp_for", "lib",
    "linear_transpose", "linearize", "map", "morton_order", "nn", "optim",
    "options", "outer_map", "parfor", "parse_float32", "parse_float64", "parse_int64", "partition_by_size", "pmap1", "pmap2",
    "pmap2_trim", "popcount", "ragged_partition", "reduce", "rotl", "rotr", "run", "scan", "scatter", "scatter_add", "seed",
    "segment_ids", "segmented_histogram2d", "segmented_map", "segmented_parfor", "segmented_reduce", "segmented_scan", "segmented_softmax",
    "show_rewrite", "soft_align", "spatial_cutoff_sum", "split_bytes", "stop_gradient", "tanimoto_topk", "topk", "topk_fold",
    "tree_array_leaves", "tree_leaves", "tree_map", "tree_map_arrays",
    "varlen_flash_attention", "vjp", "vjp_for", "wavefront",
    "ndtypes", "openmp_available", "prims", "run_python_fn", "specialize", "target_profile",
    "to_device", "to_host", "typed_repr",
    "find_broken_transform", "frontend", "macro", "syntax", "transforms", "grad", "value_and_grad",
    "Bool", "Float32", "Float64", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
]  # fmt: skip
