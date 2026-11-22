# How it works

```text
 Python function + argument types
        │  elaborate (parakeet/elab.py, arrays.py, lib/)
        ▼
 typed loop IR (parakeet/ir.py)  ──►  f.specialize(x).ir
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
release, runtime checks, and a single parallel construct — `ParLoop` — that runs
its body for each chunk of an index range. Reductions are `ParLoop`s that write
one partial result per chunk, combined afterwards.

Memory management is static: a buffer belongs to the loop iteration (or
function) that allocated it and is released when that scope ends, unless it
flows into a longer-lived variable, in which case a reference count keeps it
alive. Parallel loop bodies never touch reference counts of shared buffers.

## Backends

All targets are C-like, so one emitter (`backends/cfamily.py`) prints
expressions and statements; subclasses supply type names, math functions and
failure handling.

* The **host** program is always C. Each `ParLoop` body is outlined into a
  function taking a context struct, and launched through a function pointer
  supplied at call time — the thread pool, OpenMP, GCD or a plain loop — so the
  same compiled module serves every CPU backend.
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
