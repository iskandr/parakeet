Parakeet
====

[![CI](https://github.com/iskandr/parakeet/actions/workflows/ci.yml/badge.svg)](https://github.com/iskandr/parakeet/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-Parakeet-blue)](https://rubinsteyn.com/parakeet/)

Parakeet is a runtime compiler for an array-oriented subset of Python. It
generates a native specialization of a function for each distinct set of input
types, fuses array operations before lowering them to loops, and runs those
loops across CPU cores or on Apple Metal and NVIDIA CUDA GPUs.

Underneath, Parakeet lowers ordinary NumPy expressions, comprehensions, and its
own data-parallel operators (`map`, `reduce`, `scan`, `filter`, `scatter_add`,
`allpairs`, `dsl.fold`) to the same small algebra of array operators. Because
the compiler keeps that algebraic structure alive until the end, fusion, cache
and GPU tiling, and reverse-mode differentiation (`parakeet.grad`) all fall out
of the same representation.

Parakeet is not a general-purpose compiler for all of Python. It supports the
subset useful for numerical computing, and tells you (with your source line)
when you step outside it.


Example
=======

To accelerate a function, wrap it with Parakeet's **@jit** decorator. NumPy
expressions, explicit loops, and comprehensions all compile to native code:

```python
import numpy as np
from parakeet import jit

alpha = 0.5
beta = 0.3
x = np.array([1, 2, 3])
y = np.tanh(x * alpha) + beta


@jit
def fast(x, alpha=0.5, beta=0.3):
  return np.tanh(x * alpha) + beta


@jit
def loopy(x, alpha=0.5, beta=0.3):
  y = np.empty_like(x, dtype=float)
  for i in range(len(x)):
    y[i] = np.tanh(x[i] * alpha) + beta
  return y


@jit
def comprehension(x, alpha=0.5, beta=0.3):
  return np.array([np.tanh(xi * alpha) + beta for xi in x])


assert np.allclose(fast(x), y)
assert np.allclose(loopy(x), y)
assert np.allclose(comprehension(x), y)
```


Install
====

Parakeet 1.0 requires Python 3.11 or later and a C compiler.

    pip install parakeet                 # CPU backends
    pip install "parakeet[metal]"        # + Apple GPUs
    pip install "parakeet[cuda12]"       # + NVIDIA GPUs


Dependencies
====

Parakeet is written for Python 3.11+ and depends on:

* [NumPy](https://numpy.org) 2.0 or newer
* a C compiler: `clang` from the Xcode command line tools on macOS, `gcc` or `clang` on Linux, MinGW-w64 or LLVM on Windows

That's the whole list. If Parakeet can't find a compiler it says so and tells you how to get one.

If you want to use the Metal backend you need a Mac and the [PyObjC Metal bindings](https://pypi.org/project/pyobjc-framework-Metal/). If you want to use the CUDA backend, you need an NVIDIA graphics card and [CuPy](https://cupy.dev).


How does it work?
====

Your untyped function acts as a template from which Parakeet generates one
*type specialization* for each distinct set of input types.

During elaboration, helper calls are inlined, closures and tuples disappear,
and array expressions stay *delayed* until something consumes them, so
`np.tanh(x * alpha) + beta` becomes a single loop with no temporary arrays.
What comes out is a small operator IR (`IndexMap`, `IndexReduce`, `IndexScan`,
`IndexFilter`, `IndexScatter`). Middle-end passes apply algebraic rewrites on
those operators — vertical and sibling fusion, filter fusion, contraction
reassociation, monoidal and pairwise tiling, chunkwise scans, and cache/GPU
blocking — before lowering to loops, emitting C (plus Metal or CUDA kernels for
eligible parallel loops), compiling with your system C compiler, and caching
the result on disk.

Curious what your function turned into? `parakeet.typed_repr(f, [x])` prints
the typed IR, `parakeet.dsl.show_rewrite(f, x)` summarizes which algebraic
rewrites fired, and `f.specialize(x).source` shows the generated C.


Supported language features
====

Parakeet cannot accelerate arbitrary Python code; it compiles a static-type
numerical subset:

  * Scalar operations (`x + 3 * y`, bitwise and comparison ops, `math`)
  * Control flow (`if`/`elif`/`else`, `for`, `while`, `for`/`while ... else`, `break`, `continue`, early `return`)
  * Nested functions, lambdas and closures; default, keyword and `*args` arguments
  * Nested tuples, `NamedTuple`s, fixed string-key dicts, and plain frozen dataclasses at the JIT boundary
  * Slices, `range`, `zip`, `enumerate`
  * NumPy array expressions (`x[1:, :] + 2 * y[:-1, ::2]`), with broadcasting, boolean masks, integer-array indexing and in-place updates
  * About 200 NumPy functions and methods (`np.ones`, `np.sin`, `x.sum(axis=0)`, `np.einsum`, `np.bincount`, `np.random`, and the usual builtins — see the [full list](https://rubinsteyn.com/parakeet/supported/))
  * List literals and comprehensions (interpreted as array construction and comprehensions)
  * Data-parallel operators (`parakeet.map`, `reduce`, `scan`, `filter`, `filter_reduce`, `scatter_add`, `imap`, `allpairs`, `dsl.fold`)
  * Reverse-mode gradients: `parakeet.grad(f)` and `parakeet.value_and_grad(f)` differentiate through array operators so a whole training step compiles into one function (see [Gradients](https://rubinsteyn.com/parakeet/autodiff/))

Results and dtypes match NumPy 2. The [language reference](https://rubinsteyn.com/parakeet/language/)
lists where Parakeet deliberately differs from CPython: one static type per
variable, static array rank, no recursion, no arbitrary classes, no `try`, no
complex numbers.

Anything outside the subset fails at compile time with an error pointing at your code:

```text
parakeet.errors.UnsupportedError: numpy.fft.fft is not supported in compiled code
  File "demo.py", line 7, in spectrum
    return np.fft.fft(x)
           ^^^^^^^^^^^^^
```

Runtime failures raise the exception NumPy would have raised (`IndexError`,
`ValueError`, `ZeroDivisionError`) instead of crashing the interpreter, even
from inside a parallel loop or GPU kernel.


Backends
===

Parakeet compiles to sequential C, multi-core C, Metal, or CUDA. Switch
backends with `parakeet.config.backend`:

  * *"auto"*: the best multi-core strategy available on your machine (default)
  * *"c"*: lowers all parallel operators to sequential loops
  * *"threads"*: runs parallel operators on Parakeet's own thread pool
  * *"openmp"*: the same, with OpenMP
  * *"gcd"*: the same, with Grand Central Dispatch (macOS)
  * *"metal"* (or *"mps"*): runs eligible parallel operations on an Apple GPU
  * *"cuda"*: runs eligible parallel operations on an NVIDIA GPU
  * *"python"*: runs your original function uncompiled — handy for checking whether a bug is in your code or the compiler

You can also pick a backend per function (`@jit(backend="metal")`), per call
(`f(x, _backend="c")`), or for a block (`with parakeet.options(backend="c"): ...`).
`parakeet.available_backends()` reports which backends work on your machine
and, for any that don't, what is missing.

GPUs only run the loops they can execute natively. Apple GPUs have no
`float64`, so `float64` loops fall back to CPU cores with a warning unless you
use `float32`.


How fast is it?
====

Parakeet wins where NumPy allocates large temporaries or where you would
otherwise fall back to a Python loop. It loses to BLAS on standalone float
matrix products (which it delegates to NumPy's BLAS on CPU by default), and a
single thread loses to NumPy's SIMD `tanh`. On GPUs, launch and transfer
overhead matter; use `parakeet.to_device` to keep inputs resident across calls.

Best of 5 on an Apple M1 Pro (10 cores) with `float64` data (`python benchmarks/run.py`):

| benchmark | NumPy | Parakeet, 1 thread | Parakeet, 10 threads |
|---|---:|---:|---:|
| `np.tanh(x * a) + b`, 4M elements | 14.5 ms | 24.8 ms | 3.5 ms |
| Euclidean norm, 4M elements | 3.4 ms | 1.1 ms | 0.65 ms |
| arc distance, 4M elements | 174 ms | 174 ms | 23 ms |
| Harris corner response, 2000×2000 | 19.4 ms | 4.0 ms | 1.1 ms |
| all-pairs distances, 40k×64×32 | 103 ms | 11.1 ms | 1.6 ms |
| 3×3 convolution, 2000×2000 | 21.2 ms | 9.1 ms | 1.4 ms |
| Julia set, 1000×1000 | 402 ms | 47 ms | 6.1 ms |
| matrix product, 300×300 | 0.16 ms | 4.1 ms | 0.72 ms |

For head-to-head comparisons against PyTorch, JAX, CuPy, Faiss, and PyKeOps on
a four-vCPU worker and an NVIDIA A10 — including exact nearest-neighbor search,
Gaussian kernel regression, causal attention, linear cross-entropy, and GBDT
split finding, along with raw samples and cases where specialist libraries win —
see the [1.0 release benchmarks](https://rubinsteyn.com/parakeet/release-benchmarks/).
The [performance guide](https://rubinsteyn.com/parakeet/performance/) covers
CPU/GPU crossovers, tiling, and Metal/CUDA timings in detail.


More information
===

  * Read the [documentation](https://rubinsteyn.com/parakeet/)
  * Start with [Getting started](https://rubinsteyn.com/parakeet/quickstart/) and the [kernel gallery](https://rubinsteyn.com/parakeet/gallery/)
  * Coming from Parakeet 0.x? The [migration notes](https://rubinsteyn.com/parakeet/migrating/) list what changed
  * Watch the [Parakeet presentation](https://vimeo.com/73895275) from PyData Boston 2013, or read the [HotPar '12 paper](https://www.usenix.org/conference/hotpar12/parakeet-just-time-parallel-accelerator-python) and [Alex Rubinsteyn's 2014 thesis](https://cs.nyu.edu/media/visual/files/Rubinsteyn_Alex.pdf)


History
===

Parakeet was created between 2012 and 2014 at New York University by Alex
Rubinsteyn and Eric Hielscher (with Nathaniel Weinman and advisor Dennis
Shasha) for Python 2. Its central idea was that a small algebra of typed
data-parallel operators — maps, reductions, scans, filters, and outer products —
lets a compiler fuse, tile, and parallelize numerical Python without fragile
loop analysis. When the original Python 2 codebase was retired, its closing
note observed that high-performance numerical code belongs on explicit
data-parallel operators.

Parakeet 1.0 is a ground-up rewrite for Python 3.11+ and NumPy 2 that carries
those same ideas forward: keeping ordinary `@jit` NumPy alongside the explicit
operators, extending the adverb algebra with scatter reductions, multi-state
monoids, chunkwise scans, cache and GPU tiling, and reverse-mode `grad`, and
targeting modern multi-core CPUs, Apple Metal, and NVIDIA CUDA. The original
code lives on in the Git history.

BSD 3-clause licensed; see [LICENSE](LICENSE).
