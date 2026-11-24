# How it works

```text
 Python function + argument types
        │  elaborate (parakeet/frontend/, lib/)
        ▼
 operator IR (IndexMap, IndexReduce, IndexScan, IndexFilter, IndexScatter)
        │  middle-end passes + lowering (parakeet/transforms/)  ──►  f.specialize(x).operators / .ir
        ▼
 typed loop IR (parakeet/syntax/ir.py)
        │  emit C (backends/c.py)            ├─► Metal kernels (backends/metal.py)
        ▼                                    └─► CUDA kernels  (backends/cuda.py)
 system C compiler ─► shared library, cached on disk ─► called through ctypes
```

## Specialization

Calling a `@jit` function classifies its arguments — dtype, rank, and layout of
arrays, the types of scalars, the structure of tuples/records, and the identity
of functions — into a *type key*. Each distinct key gets one compiled
*specialization*, cached per function. Values read from globals and closures
are classified the same way on every call and passed as hidden arguments, so
compiled code never sees stale data.

## Elaboration

The frontend is a *staged interpreter*: it walks the function's Python AST with
an environment of compile-time values and emits IR only for what must happen at
run time.

* **Scalars** are IR expressions with a NumPy type.
* **Tuples, NamedTuples, fixed-key dicts, frozen dataclasses, slices, functions,
  closures, and dtypes** exist at compile time. Calling a helper function means
  elaborating its body in place, so by the time IR is produced there are no
  Python calls, closures, or tuple objects left.
* **Arrays** are either *concrete* (a buffer plus offset, shape, and strides as
  scalar expressions — slicing is just arithmetic on those) or *delayed*: a
  function from index to element. `np.tanh(x * a) + b` builds a delayed array
  three layers deep, and no operator is emitted until something consumes it: a
  store, a reduction, or `return`. That consumer's operator evaluates the whole
  expression per element. This handles basic element-wise fusion by
  construction; larger multi-operator rewrites happen next in the middle end.

Three questions cannot be answered in one forward pass, so elaboration is
*restarted* with what it learned:

1. the type of a variable modified in a loop (`x = 0` then `x += 0.5` — restart
   with `x: float`);
2. whether a delayed array must be evaluated eagerly — because it would be
   evaluated twice, inside a loop, or after an array it reads has been modified
   (which would change the answer);
3. which arguments must be compile-time constants (`axis=axis`), in which case
   the specialization is additionally keyed on that argument's value.

## The operator IR

The result of elaboration is a small imperative language: typed scalar
variables, flat buffers with load/store, `if`, counted and unbounded loops,
scoped buffer allocation, and runtime checks. On top of it sit five core
*operator statements*, preserving the adverb algebra of the original NYU
compiler:

* `IndexMap(shape, index, body)`: run `body` for every index; iterations are
  independent.
* `IndexReduce(shape, index, body, folds)`: fold the values `body` computes at
  every index; each `Fold` stores its combiner block, an associative primitive
  name when recognized, and optional multi-state `carries`. With `where`, only
  indices satisfying a predicate participate.
* `IndexFilter(n, index, body, keep)`: compact the indices where `keep` holds,
  in order, into a new array.
* `IndexScan(n, index, body, fold, emit)`: a running fold; `emit` runs at every
  index with the running state.
* `IndexScatter(n, index, body, updates)`: associative index-directed
  scatter-add (`np.bincount`, `parakeet.scatter_add`), supporting multiple
  fused updates over a shared key.

Tiling passes introduce `IndexTile` and `TileStage` nodes on top of the same
IR. `spec.operators` prints a function before the middle-end passes, and
`spec.ir` prints it after lowering. The final pass (`transforms/lowering.py`)
replaces operator statements with loops and with the single parallel construct,
`ParLoop`, which runs its body over chunks of an index range.

Memory management is static: a buffer belongs to the scope that allocated it
and is released when that scope ends, unless it escapes into a longer-lived
variable, in which case a reference count keeps it alive. Parallel loop bodies
never mutate reference counts of shared buffers.

## Middle-end passes

`parakeet/transforms/pipeline.py` lists the eleven passes over the operator
statements in order (`parakeet.transforms.pipeline.names()`). Every pass is
checked by `parakeet.verify` when `config.verify` is enabled, and
`config.skip_transforms` can omit individual passes:

1. **Dead code** (`simplify.py`): removes pure operator statements and unused
   allocations that nothing reads, to a fixed point.
2. **Normalize** (`normalize.py`): squeezes unit `keepdims=True` map axes and
   hoists row-invariant inner reductions (`softmax`, `layer_norm`).
3. **Vertical fusion** (`fusion.py`): collapses producer-consumer operator
   pairs (`Map → Map` including affine transposes/slices and matmul epilogues,
   `Scan → Map` into `IndexScan.emit`, `Map → Filter`, `Map → Reduce`,
   `Map → Scatter`) and flattens nested maps.
4. **Filter fusion** (`fusion.py`): merges adjacent `IndexFilter`s into one
   short-circuiting filter and turns `Filter → Reduce` into a zero-allocation
   guarded `IndexReduce`.
5. **Sibling fusion** (`fusion.py`): merges independent `IndexReduce`s, pure
   `IndexMap`s, compatible `IndexScatter`s sharing a key, and online
   `max` + `sum(exp(x - m))` pairs into single passes.
6. **Contraction reassociation** (`reassociate.py`): reorders chained bilinear
   contractions when doing so saves FLOPs (including latent-attention
   absorption).
7. **Monoidal tiling** (`monoid_tile.py`): tiles rescaled semiring reductions
   (attention, Split-K decoding, linear cross-entropy, Gaussian regression) and
   applies affine tile-interval mask pruning.
8. **Chunkwise scan** (`tile_scan.py`): rewrites affine and bilinear
   matrix-state `for` recurrences into parallel chunkwise scans.
9. **Pairwise fold tiling** (`pairwise_fold.py`): fuses pairwise score
   producers into ordered tile-summary folds on CPU and GPU (such as exact 1-NN).
10. **Reduction interchange** (`tiling.py`): hoists strided commutative,
    associative inner reductions outside L1-sized contiguous output strips on
    CPU (`PermuteReductions`).
11. **Tiling** (`tiling.py`): rewrites maps wrapping reductions into nested
    cache/register tiles on CPU and shared-memory tiles on CUDA and Metal.

After those eleven passes, **lowering** (`lowering.py`) turns operators into
loops and `ParLoop` chunks.

One transform still runs during elaboration rather than as a middle-end pass:

* **Differentiation** (`autodiff.py`, `derivatives.py`, `algebraic_vjp.py`):
  `grad` elaborates the function with a tape that records array operators on
  differentiated values, runs algebraic rematerialization rules (for attention,
  cross-entropy, and norms), and walks the tape backwards to emit adjoint
  operators. Scalars differentiate directly from their expression trees, and
  views map to the same views of their adjoint buffers.

## Backends

All targets are C-like, so one emitter (`backends/cfamily.py`) prints scalar
expressions and control flow while backend modules supply target types, math
intrinsics, tiled GPU codegen (`backends/tiled_gpu.py`, `backends/monoidal_gpu.py`),
and runtime error reporting.

* The **host** program is always C. Each `ParLoop` body is outlined into a
  function taking a context struct (identical bodies up to alpha-renaming share
  one compiled function) and dispatched through a function pointer supplied at
  call time — the thread pool, OpenMP, GCD, or a sequential loop — so one
  compiled library serves every CPU backend.
* For **GPU** backends, eligible `ParLoop`s and tiled plans are also emitted as
  Metal or CUDA kernels. Arrays stay resident on the device across consecutive
  GPU kernels within a call (and across calls when wrapped with
  `parakeet.to_device`).

Scalar helpers (Python/NumPy floor division and modulo, integer powers, shifts,
NaN-propagating min/max, and the RNG) live once as C text and are included in
the C, Metal, and CUDA preludes so all backends agree bit-for-bit on integer
results.

## Calling convention

A compiled entry point takes flat arrays of 64-bit slots: scalars, and for each
array its buffer pointer, offset, shape, and strides. Results return the same
way. Buffers allocated by compiled code are handed to NumPy without copying
using a finalizer that releases the native reference; views of inputs return as
NumPy views of the caller's memory.
