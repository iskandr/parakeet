# How it works

```text
 Python function + argument types
        │  elaborate (parakeet/frontend/, lib/)
        ▼
 typed loop IR (parakeet/syntax/ir.py)  ──►  f.specialize(x).ir
        │  print as C (backends/c.py)        ├─► Metal kernels (backends/metal.py)
        ▼                                    └─► CUDA kernels  (backends/cuda.py)
 system C compiler ─► shared library, cached on disk ─► called through ctypes
```

## Specialization

Calling a `@jit` function classifies its arguments — dtype, rank and layout of
arrays, the types of scalars, the structure of tuples, the identity of functions
— into a *type key*. Each distinct key gets one compiled *specialization*, cached
per function. Values read from globals and closures are classified the same way
on every call and passed as hidden arguments, so compiled code never sees stale
data.

## Elaboration

The frontend is a *staged interpreter*: it walks the function's Python AST with
an environment of compile-time values and emits IR only for what must happen at
run time.

* **Scalars** are IR expressions with a NumPy type.
* **Tuples, slices, functions, closures, dtypes** exist only at compile time.
  Calling a function means elaborating its body in place, so by the time IR is
  produced there are no calls, closures or tuples left.
* **Arrays** are either *concrete* (a buffer plus offset, shape and strides as
  scalar expressions — slicing is just arithmetic on those) or *delayed*: a
  function from index to element. `np.tanh(x * a) + b` builds a delayed array
  three layers deep, and no loop is emitted until something consumes it: a
  store, a reduction, or the `return`. That consumer's loop evaluates the whole
  expression per element. This is how fusion happens — there is no separate
  fusion pass.

Three questions cannot be answered in one forward pass, so elaboration is
*restarted* with what it learned:

1. the type of a variable modified in a loop (`x = 0` then `x += 0.5` — restart
   with `x: float`);
2. whether a delayed array must be evaluated eagerly — because it would be
   evaluated twice, inside a loop, or after an array it reads has been modified
   (which would change the answer);
3. which arguments must be compile-time constants (`axis=axis`), in which case
   the specialization is additionally keyed on that argument's value.

## The IR

The result is a small imperative language: typed scalar variables, flat buffers
with load/store, `if`, counted and unbounded loops, allocation with scoped
release and runtime checks. On top of it sit four *operator statements*, which
is as much of the original's adverb tree as the middle end needs:

* `IndexMap(shape, index, body)`: run `body` for every index; the iterations
  are independent.
* `IndexReduce(shape, index, body, folds)`: fold the values `body` computes
  at every index; each `Fold` has its combiner as a block of IR and, when the
  combiner is a known associative primitive, its name. With `where`, only
  the indices at which a predicate holds count.
* `IndexFilter(n, index, body, keep)`: the indices at which `keep` holds, in
  order, as a new array.
* `IndexScan(n, index, body, fold, emit)`: a running fold; `emit` runs at
  every index with the running value.

`spec.operators` prints a function at this level, `spec.ir` after the passes.
The last pass (`transforms/lowering.py`) replaces the operator statements by
loops and by the single parallel construct — `ParLoop` — that runs its body
for each chunk of an index range. A parallel reduction is a `ParLoop` that
writes one partial result per chunk, combined afterwards.

Memory management is static: a buffer belongs to the loop iteration (or
function) that allocated it and is released when that scope ends, unless it
flows into a longer-lived variable, in which case a reference count keeps it
alive. Parallel loop bodies never touch reference counts of shared buffers.

## Backends

All targets are C-like, so one emitter (`backends/cfamily.py`) prints
expressions and statements; subclasses supply type names, math functions and
failure handling.

* The **host** program is always C. Each `ParLoop` body is outlined into a
  function taking a context struct (bodies with identical IR structure up to
  alpha-renaming share one compiled function), and launched through a function
  pointer supplied at call time — the thread pool, OpenMP, GCD or a plain loop —
  so the same compiled module serves every CPU backend.
* For **GPU** backends each eligible `ParLoop` is additionally printed as a
  Metal or CUDA kernel. At run time the host code calls back into Python, which
  binds the buffers and dispatches the kernel; if the loop is not eligible, is
  too small, or the device fails, the C version runs instead.

Scalar helper functions (Python/NumPy division and modulo semantics, integer
power, shifts, NaN-propagating min/max, the random number generator) exist once
as C text and are included verbatim in the C, Metal and CUDA preludes, which is
why the backends agree bit-for-bit on integer results.

## Calling convention

A compiled entry point takes flat arrays of 64-bit slots: scalars, and for each
array its buffer pointer, offset, shape and strides. Results come back the same
way. Arrays allocated by compiled code are handed to NumPy without copying, with
a finalizer that drops the native reference; views of inputs come back as NumPy
views of the caller's array.

## Transforms

`parakeet/transforms/pipeline.py` lists the passes over the operator
statements, in order; `config.skip_transforms` leaves any of them out:

* **Dead code** (`simplify.py`). Pure `IndexMap`, `IndexReduce`, `IndexFilter`,
  `IndexScan` and unused `Alloc` statements nobody reads are removed to a fixed
  point (the C compiler cannot remove a parallel loop).
* **Vertical fusion** (`fusion.py`). Producer-consumer operator pairs
  (`Map→Map` including epilogues like `np.tanh(a @ b + c)`, `Scan→Map` into
  `IndexScan.emit`, `Map→Filter`, `Map→Reduce`) collapse into one operator and
  drop the intermediate buffer.
* **Filter fusion** (`fusion.py`). Adjacent `IndexFilter`s collapse into a
  single short-circuiting filter (`Filter→Filter`), and a reduction over what an
  `IndexFilter` kept becomes a guarded `IndexReduce` with no index array
  (`Filter→Reduce`).
* **Sibling fusion** (`fusion.py`). Adjacent `IndexReduce`s or independent
  `IndexMap`s over the same index space share one loop.
* **Reduction interchange** (`tiling.py`). `PermuteReductions`: a strided
  commutative+associative `IndexReduce` inside a stride-1 `IndexMap`
  (`np.sum(X, axis=0)`) is interchanged over L1-sized strips of contiguous
  output columns.
* **Tiling** (`tiling.py`). An `IndexMap` of rank two or more whose body is a
  single `IndexReduce` plus straight-line code is run a cache-sized tile of
  its last dimension at a time. Inside a tile the body is instantiated for a
  2×4 block of elements (`syntax/clone.py` copies IR blocks) and the eight
  reductions are fused into one with eight folds.
* **Lowering** (`lowering.py`). Always last: operators to loops. This is
  where a reduction, scan or filter that may run in parallel gets its
  per-chunk partial results and its second pass.

One thing still rewrites the program while it is elaborated:

* **Differentiation** (`autodiff.py`, `derivatives.py`). `grad` elaborates its
  function with a tape that records each array operator applied to a
  differentiated value — at the level of `elementwise`, `reduce`, `matmul` and
  `map`, where each has a one-line derivative rule — then walks the tape
  backwards emitting the adjoint computation with the same building blocks.
  Scalars are differentiated straight from their expression trees, and a view
  of an array is the same view of its adjoint buffer. Assignments and stores
  the tape cannot account for *taint* what they write; reaching a tainted
  value during the backward pass is a compile error.

