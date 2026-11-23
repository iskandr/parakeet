![Parakeet: a runtime compiler for numerical Python](assets/logo.png){ .pk-hero-logo }

**Parakeet** is a runtime compiler for scientific computing in Python which uses
type specialization, data-parallel array operators, and algebraic loop
transformations to turn numerical Python into fused, parallel native code for
multi-core CPUs, Apple GPUs (Metal), and NVIDIA GPUs (CUDA).

## Examples

The two classic `parakeetpython.com` examples — still compiling straight from
ordinary Python to fused, tiled native code — plus Parakeet 1.0's filtered
reductions and reverse-mode gradients:

```python
import numpy as np
import parakeet as pk
from parakeet import jit


@jit
def allpairs_dist(X, Y):
  """All-pairs squared Euclidean distances between rows of X and Y.
  Compiles to a cache-tiled, 2x4 jammed parallel loop with no intermediates
  (1000x1000 float64: ~26 ms on an M1 Max)."""
  def dist(x, y):
    return np.sum((x - y) ** 2)

  return np.array([[dist(x, y) for y in Y] for x in X])


@jit
def smoothing(x, alpha):
  """Exponential smoothing of a time series (explicit stateful loop).
  1,000,000 float64s: ~4 ms vs ~220 ms in CPython (~55x faster)."""
  s = x.copy()
  for i in range(1, len(x)):
    s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]
  return s


# Data-parallel operators and reverse-mode automatic differentiation
pos_sum = pk.filter_reduce(pk.add, lambda v: v > 0.0, x, init=0.0)  # one pass, zero allocations
dloss = pk.grad(lambda w, x: np.mean(np.tanh(x @ w) ** 2))  # differentiates array code & @
```

<div class="grid cards" markdown>

- **Algebraic operator IR.** NumPy expressions, comprehensions, slices and
  loops lower to four data-parallel operators (`IndexMap`, `IndexReduce`,
  `IndexScan`, `IndexFilter`) where fusion, interchange and tiling are simple
  algebraic rewrites — verified after every pass.
- **Every core, or the GPU.** Built-in thread pool, OpenMP or GCD on CPUs;
  Metal on Apple GPUs; CUDA (via CuPy/NVRTC) on NVIDIA GPUs. Loops that cannot
  run on the GPU fall back to the CPU and tell you why.
- **Reverse-mode gradients.** `grad` and `value_and_grad` differentiate array
  code, reductions, `@`, `einsum`, views, `cumsum`, `sort` and `map` — enough
  to train small transformers and convnets.
- **Fails loudly, nothing to set up.** Unsupported code is rejected at compile
  time with your source caret; bad indices and shapes raise ordinary Python
  exceptions. `pip install parakeet` plus a C compiler.

</div>

## Install

```bash
pip install parakeet                 # Python 3.11+, NumPy 2, a C compiler
pip install "parakeet[metal]"        # + Apple GPU backend
pip install "parakeet[cuda12]"       # + NVIDIA GPU backend (CUDA 12)
```

Continue with [Getting started](quickstart.md), or read
[Core ideas, history, and boundaries](philosophy.md) for the PhD thesis design
behind Parakeet and an honest map of what does and does not work yet.

## Limitations (and what changed since 0.x)

Parakeet is **not** a general-purpose Python compiler — trying to compile an
arbitrary Python function will fail loudly. It compiles a statically typed,
array-oriented [subset](language.md) of Python and NumPy.

Several classic limitations from the original `parakeetpython.com` page have
been lifted in 1.0:

* `break`, `continue`, early `return`, `assert`, and `raise` now work (the
  original Structured SSA IR disallowed them).
* Full NumPy 2 broadcasting — including unit dimensions (`8×1×6×1` with
  `7×1×5`) — works statically.
* Python 3.11+ and NumPy 2, with zero compiler-framework dependencies (no
  `llvmpy`, `PyCUDA`, or `nvcc` required).

See [The boundary: what does not work yet](philosophy.md#the-boundary-what-does-not-work-yet)
for the exact limits on types, NumPy/SciPy coverage, `grad`, and GPUs, and
[Performance](performance.md) for CPU and NVIDIA A100 benchmarks (including
where plain NumPy/BLAS is faster).

## History, talks, and theses

Parakeet was created at New York University by **Alex Rubinsteyn**
([@iskandr](https://github.com/iskandr)), with major contributions from **Eric
Hielscher** ([@hielscher](https://github.com/hielscher)) and **Nathaniel
Weinman**, guidance from **Dennis Shasha**, and advice from **Russell Power**.
It was originally hosted at `parakeetpython.com` and is documented in:

* **Alex Rubinsteyn**, [*Runtime Compilation of Array-Oriented Python Programs*](https://cs.nyu.edu/media/visual/files/Rubinsteyn_Alex.pdf) (PhD thesis, NYU, 2014)
* **Eric Hielscher**, *Locality Optimization for Data Parallel Programs* (PhD thesis, NYU, 2013)
* **HotPar '12:** [*Parakeet: A Just-in-Time Parallel Accelerator for Python*](https://www.usenix.org/conference/hotpar12/parakeet-just-time-parallel-accelerator-python)
* **PyData NYC '13:** [*Python on the GPU with Parakeet*](https://vimeo.com/79556629)
* **PyData Boston '13:** [*Parakeet vs. Numba*](https://vimeo.com/73895275)
