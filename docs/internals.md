# Compiler architecture

This page covers how Parakeet compiles Python into native CPU and GPU code,
including the low-level typed IR (`spec.operators` and `spec.ir`) that sits
underneath the algebraic rewrite passes.

```text
 Python function + argument types
        │  elaborate (parakeet/frontend/, lib/)
        ▼
 operator IR (IndexMap, IndexReduce, IndexScan, IndexFilter, IndexExpand, IndexScatter, IndexWavefront)
        │  12 middle-end passes + lowering (parakeet/transforms/)  ──►  f.specialize(x).operators / .ir
        ▼
 typed loop IR (parakeet/syntax/ir.py: ParLoop, ForRange, flat buffer loads/stores)
        │  emit C (backends/c.py)            ├─► Metal kernels (backends/metal.py)
        ▼                                    └─► CUDA kernels  (backends/cuda.py)
 system C compiler ─► shared library, cached on disk ─► called through ctypes
```

## Specialization

Calling a `@jit` function classifies its arguments (array dtype, rank, layout;
scalar types; tuple/record structure; callable identity) into a *type key*.
Each distinct key compiles to one cached specialization. Globals and closed-over
values are classified the same way on every call and passed as hidden arguments
so compiled code never reads stale values.

## Elaboration

The frontend is a *staged interpreter* walking the Python AST with compile-time
values and emitting IR only for runtime work:

* **Scalars** become typed IR expressions.
* **Tuples, `NamedTuple`s, fixed-key dicts, dataclasses, slices, functions, and
  dtypes** live at compile time. Helper calls inline directly, leaving no
  Python call frames or tuple allocations in the IR.
* **Arrays** are either *concrete* (a buffer pointer plus offset, shape, and
  strides, so views and slices are pure integer arithmetic) or *delayed* (a
  function from index to value). Delayed expressions fuse into their consumer
  at elaboration time before any middle-end pass runs.

Three situations trigger an elaboration restart with refined information:

1. A loop-carried variable changes type (`x = 0` then `x += 0.5`; restart with
   `x: float`).
2. A delayed array needs eager evaluation because it is read twice, read inside
   a loop, or invalidated by an aliasing store.
3. An argument must be specialized by value (`axis=axis`, tuple lengths, or
   unit broadcast dimensions).

## Middle-end passes

`parakeet/transforms/pipeline.py` runs twelve passes over the operator IR in
order (`parakeet.transforms.pipeline.names()`). Every pass is checked by
`parakeet.verify` when `config.verify` is enabled, and `config.skip_transforms`
can omit passes individually:

1. **Dead code** (`simplify.py`): drops pure operators and unused allocations
   to a fixed point.
2. **Normalize** (`normalize.py`): squeezes unit `keepdims=True` map axes and
   hoists row-invariant inner reductions (`softmax`, `layer_norm`).
3. **Vertical fusion** (`fusion.py`): collapses producer-consumer pairs
   (`Map -> Map` including affine transposes/slices and matmul epilogues,
   `Scan -> Map` into `IndexScan.emit`, `Map -> Filter/Reduce/Scatter`) and
   flattens nested maps.
4. **Filter fusion** (`fusion.py`): merges adjacent `IndexFilter`s and turns
   `Filter -> Reduce` into a zero-allocation guarded `IndexReduce`.
5. **Sibling fusion** (`fusion.py`): merges independent reductions, pure maps
   (including disjoint-union horizontal batching on CUDA), compatible same-key
   scatters, and online `max` + `sum(exp(x - m))` pairs.
6. **Contraction reassociation** (`reassociate.py`): reorders chained bilinear
   contractions when doing so lowers FLOPs (including latent-attention
   absorption).
7. **Monoidal tiling** (`monoid_tile.py`): tiles rescaled semiring reductions
   (attention, Split-K decoding, linear cross-entropy, Gaussian regression) and
   applies affine tile-interval mask pruning.
8. **Row pipeline fusion** (`fusion.py`): collapses multi-stage per-row
   reduction + normalization chains (such as LayerNorm and RMSNorm forward and
   backward) into single-pass warp-cooperative row pipelines on GPU.
9. **Chunkwise scan** (`tile_scan.py`): rewrites affine and bilinear
   matrix-state `for` loops into parallel chunkwise scans.
10. **Pairwise fold tiling** (`pairwise_fold.py`): fuses pairwise score
    producers into ordered tile-summary folds on CPU and GPU (such as 1-NN).
11. **Reduction interchange** (`tiling.py`): hoists strided commutative,
    associative inner reductions outside L1-sized output strips on CPU
    (`PermuteReductions`).
12. **Tiling** (`tiling.py`): rewrites maps wrapping reductions into nested
    cache/register tiles on CPU and shared-memory tiles on CUDA and Metal, plus
    single-pass `atomicAdd` reductions on CUDA.

After these twelve passes, **lowering** (`lowering.py`) converts all operator
statements into scalar loops and the single parallel loop construct, `ParLoop`.

One transform runs during elaboration rather than as a middle-end pass:
**differentiation** (`autodiff.py`, `jvp.py`, `algebraic_vjp.py`), which records
array operators on an algebraic tape during elaboration, applies sufficient-statistic
rematerialization (for attention, cross-entropy, softmax, and norms), and walks
the tape backwards to emit adjoint operators.

## Lowered IR (`spec.operators` and `spec.ir`) {#lowered-ir}

The [primitives guide](primitives.md) and [rewrite rules](rewrites.md) use
Python pseudocode to show algebraic transformations cleanly. Underneath, the
compiler represents functions in a typed, explicit-offset SSA IR with scoped
static memory management (buffers are freed when their allocating scope exits,
with reference counting only when a buffer escapes).

Inspect both stages on any `@par.jit` function via `.specialize(*args)`:

```python
import numpy as np
import parakeet as par


@par.jit
def rms(x):
  return np.sqrt(np.mean(x * x))


spec = rms.specialize(np.ones(1024, dtype=np.float64))
print(spec.operators)  # operator IR before middle-end passes
print(spec.ir)  # lowered loop IR after all 12 passes + lowering
```

In `spec.ir`, `IndexReduce` has been lowered into a chunked `ParLoop` with
explicit flat-buffer offsets (`x_off_2`), typed SSA temporaries (`x_15`,
`acc_14`), and a SIMD inner loop:

```text
def rms(x_1: array[float64, 1d]):
    nchunks_9: int64? = @num_chunks(x_n0_3, True)
    partials_10 = alloc(float64, nchunks_9)
    parallel[reduce] for chunk_11, (start_12, stop_13) in chunks(x_n0_3, n=nchunks_9):
        x_15: float64 = (x_1[(x_off_2 + start_12)] * x_1[(x_off_2 + start_12)])
        acc_14: float64 = x_15
        for i_16 in range((start_12 + 1), stop_13, 1):  # simd
            x_17: float64 = (x_1[(x_off_2 + i_16)] * x_1[(x_off_2 + i_16)])
            acc_14: float64 = (acc_14 + x_17)
        partials_10[chunk_11] = acc_14
    acc_8: float64 = 0.0
    for i_18 in range(0, nchunks_9, 1):
        acc_8: float64 = (acc_8 + partials_10[i_18])
    res_20: float64 = sqrt((acc_8 / float64(x_n0_3)))
    return res_20
```

To trace the IR after every pass that modifies a function, pass a callback to
`trace_passes`:

```python
from parakeet.syntax import ir

prev = None


def on_pass(stage, fn):
  global prev
  text = ir.fmt_function(fn)
  if text != prev:
    print(f"=== {stage} ===\n{text}")
    prev = text


rms.trace_passes(np.ones(1024), _trace=on_pass)
```

## Backends and calling convention

All targets share one C-family scalar emitter (`backends/cfamily.py`) and one
set of C scalar helpers (NumPy floor division/modulo, NaN-propagating min/max,
and RNG) so CPU, Metal, and CUDA agree bit-for-bit on integer results.

* **Host C:** each `ParLoop` body is outlined into a function taking a context
  struct (identical bodies up to alpha-renaming share one compiled function)
  and dispatched through a function pointer supplied at call time (thread pool,
  OpenMP, GCD, or sequential), so one shared library serves every CPU backend.
* **GPU kernels:** eligible `ParLoop`s and tiled plans emit Metal or CUDA
  kernels (`backends/metal.py`, `backends/cuda.py`, `backends/tiled_gpu.py`,
  `backends/monoidal_gpu.py`). On CUDA, alpha-canonical kernel deduplication
  collapses structurally identical kernels across layers (for example, repeated
  transformer blocks) into a single NVRTC compilation. Arrays remain resident
  on the device across consecutive GPU kernels within a call, and across calls
  when wrapped with `par.to_device`.
* **Calling convention:** compiled entry points accept flat arrays of 64-bit
  slots (scalars, and for each array its buffer pointer, offset, shape, and
  strides). Newly allocated output buffers are handed to NumPy without copying
  via a finalizer that releases the native reference; views of inputs return as
  views of caller memory.
