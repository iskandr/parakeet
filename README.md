# Parakeet

[![CI](https://github.com/iskandr/parakeet/actions/workflows/ci.yml/badge.svg)](https://github.com/iskandr/parakeet/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-iskandr.github.io%2Fparakeet-blue)](https://iskandr.github.io/parakeet/)

**Parakeet is a runtime compiler for numerical Python.** Decorate an array-oriented
function with `@jit` and Parakeet compiles a type-specialised native version the
first time you call it: array expressions are fused into single loops, and those
loops run on every CPU core or on a GPU (Apple Metal or NVIDIA CUDA).

```python
import numpy as np
from parakeet import jit


@jit
def fast(x, alpha=0.5, beta=0.3):
    return np.tanh(x * alpha) + beta  # one fused, multi-threaded loop


@jit
def loopy(x, alpha=0.5, beta=0.3):
    y = np.empty_like(x, dtype=float)
    for i in range(len(x)):  # plain loops are compiled too
        y[i] = np.tanh(x[i] * alpha) + beta
    return y


@jit
def comprehension(x, alpha=0.5, beta=0.3):
    return np.array([np.tanh(xi * alpha) + beta for xi in x])


x = np.array([1, 2, 3])
assert np.allclose(fast(x), np.tanh(x * 0.5) + 0.3)
assert np.allclose(loopy(x), fast(x)) and np.allclose(comprehension(x), fast(x))
```

Full documentation: **<https://iskandr.github.io/parakeet/>**

## Install

```bash
pip install parakeet                 # needs Python 3.11+, NumPy 2 and a C compiler
pip install "parakeet[metal]"        # + Apple GPU backend (macOS)
pip install "parakeet[cuda12]"       # + NVIDIA GPU backend (CuPy for CUDA 12)
```

A C compiler is the only system requirement: the Xcode command line tools on
macOS (`xcode-select --install`), `gcc` or `clang` on Linux, MinGW-w64 or LLVM
on Windows. If none is found, Parakeet says so and tells you how to fix it.

## Backends

| `backend=` | What runs your loops | Requirements |
|---|---|---|
| `"auto"` (default) | best multi-core strategy available | C compiler |
| `"c"` | one thread | C compiler |
| `"threads"` | Parakeet's own thread pool | POSIX threads |
| `"openmp"` / `"gcd"` | OpenMP / Grand Central Dispatch | OpenMP runtime / macOS |
| `"metal"` (alias `"mps"`) | Apple GPU, falling back to the CPU per loop | `parakeet[metal]` |
| `"cuda"` | NVIDIA GPU, falling back to the CPU per loop | `parakeet[cuda12]` |
| `"python"` | the original function, uncompiled (for debugging) | – |

```python
@jit(backend="metal")
def distances(X, C):  # float32 in, float32 out
    return parakeet.allpairs(lambda a, b: np.sum((a - b) ** 2), X, C)


parakeet.available_backends()  # what works on this machine, and why not
```

Choose a backend per function (`@jit(backend=...)`), per call (`f(x, _backend="c")`),
globally (`parakeet.config.backend = ...`, or `PARAKEET_BACKEND`), or temporarily
(`with parakeet.options(backend="c"): ...`).

## What can be compiled

A statically typeable, array-oriented subset of Python:

* scalars, tuples, slices, `if`/`for`/`while`/`break`/`continue`, early `return`
* nested functions, lambdas and closures; default, keyword and `*args` arguments
* NumPy arrays of bool, (u)int8–64, float32/64 with views, broadcasting,
  boolean-mask and integer-array indexing, in-place updates
* ~200 NumPy functions and methods (ufuncs, reductions with `axis=`, constructors,
  shape manipulation, `np.linalg.norm`, `np.random`, sorting, ...) plus `math`
  and the usual builtins — see the [full list](https://iskandr.github.io/parakeet/supported/)
* list literals and comprehensions (as arrays)
* Parakeet's data-parallel operators: `parakeet.map`, `reduce`, `scan`,
  `allpairs`, `imap`, `parfor`, ...

Result values *and dtypes* match NumPy (including NumPy 2 promotion rules); the
[language reference](https://iskandr.github.io/parakeet/language/) lists the
deliberate differences. Anything outside the subset fails at compile time with
an error that points at your source line:

```text
parakeet.errors.UnsupportedError: numpy.fft.fft is not supported in compiled code
  File "demo.py", line 7, in spectrum
    return np.fft.fft(x)
           ^^^^^^^^^^^^^
```

Runtime problems raise the exception NumPy would have raised (`IndexError`,
`ValueError`, `ZeroDivisionError`, ...) instead of crashing — bounds checks are
on by default and can be disabled with `@jit(bounds_check=False)`.

## Performance

Best-of-5 wall-clock times on an Apple M1 Pro (10 cores), Python 3.12, NumPy 2.5,
from `python benchmarks/run.py`; "×" is speed-up over the NumPy version of the
same code.

| benchmark (float64) | NumPy | Parakeet, 1 thread | Parakeet, 10 threads |
|---|---:|---:|---:|
| `np.tanh(x * a) + b`, 4M elements | 14.5 ms | 24.8 ms (0.6×) | 3.5 ms (4.2×) |
| Euclidean norm, 4M elements | 3.4 ms | 1.1 ms (3.1×) | 0.65 ms (5.3×) |
| arc distance, 4M elements | 174 ms | 174 ms (1.0×) | 23 ms (7.5×) |
| Harris corner response, 2000×2000 | 19.4 ms | 4.0 ms (4.9×) | 1.1 ms (17×) |
| all-pairs distances, 40k×64×32 | 103 ms | 11.1 ms (9.2×) | 1.6 ms (66×) |
| 3×3 convolution, 2000×2000 | 21.2 ms | 9.1 ms (2.3×) | 1.4 ms (15×) |
| Julia set, 1000×1000 | 402 ms | 47 ms (8.5×) | 6.1 ms (66×) |
| matrix product, 300×300 | 0.16 ms | 4.1 ms | 0.72 ms |

Parakeet is fastest where NumPy allocates temporaries or needs Python loops. It
does **not** replace BLAS (matrix products) and its single-threaded code does not
use vectorised math libraries (`tanh`, `sin`, ...), where NumPy can be ahead.
GPU numbers, and when a GPU is *slower*, are in the
[performance notes](https://iskandr.github.io/parakeet/performance/).

## How it works

Your function is a template. On each call with new argument types Parakeet
elaborates its Python source into a typed loop IR — inlining calls, resolving
closures and tuples at compile time, and keeping array expressions *delayed* so
they fuse into whatever consumes them — then prints that IR as C (plus Metal or
CUDA kernels for parallel loops), compiles it with the system compiler, and
caches the result on disk. `f.specialize(x).ir` shows the IR; `.source` the C.

## History

Parakeet was first written in 2012–2014 by Alex Rubinsteyn and Eric Hielscher at
NYU for Python 2 ([HotPar '12 paper](https://www.usenix.org/conference/hotpar12/parakeet-just-time-parallel-accelerator-python)).
Version 1.0 is a ground-up rewrite for modern Python that keeps the original
interface and ideas; the original code lives on in the Git history (tag
`v0.23.2` and earlier). See the [migration notes](https://iskandr.github.io/parakeet/migrating/).

## License

BSD 3-clause; see [LICENSE](LICENSE).
