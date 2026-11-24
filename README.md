Parakeet
====

[![CI](https://github.com/iskandr/parakeet/actions/workflows/ci.yml/badge.svg)](https://github.com/iskandr/parakeet/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-iskandr.github.io%2Fparakeet-blue)](https://iskandr.github.io/parakeet/)

Parakeet is a runtime accelerator for an array-oriented subset of Python. It compiles native versions of a function for each distinct set of input types, fuses the array expressions inside it into single loops, and runs those loops on every core or on a GPU. Parakeet also exposes data parallel operators (`map`, `reduce`, `scan`, `allpairs`) which enable high-level optimizations and parallelization.

Parakeet is not a general-purpose compiler for all of Python. It supports only the language constructs useful for numerical computing, and tells you (with your source line) when you have wandered outside of them.


Example
=======

To accelerate a function, wrap it with Parakeet's **@jit** decorator:

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

From a checkout of the rewrite, run `python -m pip install .` (or `".[metal]"` / `".[cuda12]"` in place of `.`). Once the alpha is published, install the same version from PyPI with

    pip install "parakeet==1.0.0a1"                 # CPU backends
    pip install "parakeet[metal]==1.0.0a1"        # + Apple GPUs
    pip install "parakeet[cuda12]==1.0.0a1"       # + NVIDIA GPUs


Dependencies
====

Parakeet is written for Python 3.11+ and depends on:

* [NumPy](https://numpy.org) 2.0 or newer
* a C compiler: `clang` from the Xcode command line tools on macOS, `gcc` or `clang` on Linux, MinGW-w64 or LLVM on Windows

That's the whole list. If Parakeet can't find a compiler it says so and tells you how to get one.

If you want to use the Metal backend you need a Mac and the [PyObjC Metal bindings](https://pypi.org/project/pyobjc-framework-Metal/). If you want to use the CUDA backend, you need an NVIDIA graphics card and [CuPy](https://cupy.dev).


How does it work?
====

Your untyped function gets used as a template from which multiple *type specializations* are generated
(for each distinct set of input types).
Calls are inlined, closures and tuples disappear at compile time, and array expressions stay *delayed* until something consumes them, which is how `np.tanh(x * alpha) + beta` becomes one loop and no temporaries.
These typed functions are then translated into C (plus Metal or CUDA kernels for the parallel loops), compiled with your system's compiler, and cached on disk.

Curious what your function turned into? `parakeet.typed_repr(f, [x])` shows the typed code and `f.specialize(x).source` shows the C.


Supported language features
====

Parakeet cannot accelerate arbitrary Python code, it only supports a limited subset of the language:

  * Scalar operations (i.e. `x + 3 * y`)
  * Control flow (if-statements, loops, `break`, `continue`, early returns)
  * Nested functions, lambdas and closures; default, keyword and `*args` arguments
  * Tuples
  * Slices
  * NumPy array expressions (i.e. `x[1:, :] + 2 * y[:-1, ::2]`), with broadcasting, boolean masks, integer-array indexing and in-place updates
  * About 200 NumPy library functions and methods like `np.ones`, `np.sin` and `x.sum(axis = 0)`, plus `math`, `np.random` and the usual builtins (look at the [full list](https://iskandr.github.io/parakeet/supported/))
  * List literals (interpreted as array construction)
  * List comprehensions (interpreted as array comprehensions)
  * Parakeet's higher order array operations like `parakeet.imap`, `parakeet.scan`, and `parakeet.allpairs`
  * Gradients: `parakeet.grad(f)` differentiates the array operators, so a training step is one compiled function (see [Gradients](https://iskandr.github.io/parakeet/autodiff/))

Results and their dtypes match NumPy 2. The [language reference](https://iskandr.github.io/parakeet/language/) lists the places where Parakeet deliberately differs from CPython: one type per variable, no recursion, no classes, no `try`, no complex numbers.

Anything outside the subset fails when the function is compiled, with an error that points at your code:

```text
parakeet.errors.UnsupportedError: numpy.fft.fft is not supported in compiled code
  File "demo.py", line 7, in spectrum
    return np.fft.fft(x)
           ^^^^^^^^^^^^^
```

Things that go wrong at runtime raise the exception NumPy would have raised (`IndexError`, `ValueError`, `ZeroDivisionError`) instead of taking down your interpreter, even from inside a parallel loop or a GPU kernel.


Backends
===

Parakeet compiles to sequential C, multi-core C, Metal, or CUDA. To switch between these options change `parakeet.config.backend` to one of:

  * *"auto"*: the best multi-core strategy available on your machine (default)
  * *"c"*: lowers all parallel operators to loops, compiles sequential code
  * *"threads"*: parallel operators run across multiple cores using Parakeet's own thread pool
  * *"openmp"*: the same, with OpenMP
  * *"gcd"*: the same, with Grand Central Dispatch (macOS)
  * *"metal"* (or *"mps"*): launch parallel operations on an Apple GPU
  * *"cuda"*: launch parallel operations on an NVIDIA GPU
  * *"python"*: don't compile anything, just call your function; for when you suspect Parakeet rather than yourself

You can also pick a backend for one function (`@jit(backend = "metal")`), for one call (`f(x, _backend = "c")`), or for a block of code (`with parakeet.options(backend = "c"): ...`). `parakeet.available_backends()` tells you which ones work on your machine and, for the ones that don't, why not.

GPUs only get the loops they can run. Apple GPUs have no `float64`, so on Metal those loops run on the CPU and you get a warning saying so.


How fast is it?
====

Best of 5 on an Apple M1 Pro (10 cores) with `float64` data, from `python benchmarks/run.py`:

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

Parakeet wins where NumPy allocates temporaries or where you would otherwise write a Python loop. It loses to BLAS on matrix products, and a single thread of Parakeet loses to NumPy's vectorized `tanh`. GPUs pay to copy your arrays over and back on every call, so they only help when there is a lot of arithmetic per element; the [performance notes](https://iskandr.github.io/parakeet/performance/) have the Metal and CUDA numbers, including the ones where the GPU is slower.


More information
===

  * Read the [documentation](https://iskandr.github.io/parakeet/)
  * Look at the [examples](examples) and the [benchmarks](benchmarks)
  * Coming from Parakeet 0.x? The [migration notes](https://iskandr.github.io/parakeet/migrating/) list what changed
  * Watch the [Parakeet presentation](https://vimeo.com/73895275) from PyData Boston 2013, or look at the [HotPar slides](https://www.usenix.org/conference/hotpar12/parakeet-just-time-parallel-accelerator-python)


History
===

Parakeet was written between 2012 and 2014 by Alex Rubinsteyn and Eric Hielscher for Python 2. Version 1.0 is a rewrite for modern Python which keeps the interface and the ideas. The original code lives on in the Git history.

BSD 3-clause licensed; see [LICENSE](LICENSE).
