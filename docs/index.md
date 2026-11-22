# Parakeet

**A runtime compiler for numerical Python.** Decorate an array-oriented function
with `@jit`; the first call compiles a native version specialised to the argument
types, with array expressions fused into single loops that run on every CPU core
or on a GPU.

```python
import numpy as np
from parakeet import jit


@jit
def scaled_tanh(x, alpha=0.5, beta=0.3):
  return np.tanh(x * alpha) + beta  # one loop, no temporaries, all cores


@jit
def count_inside(points, radius):
  n = 0
  for i in range(len(points)):  # explicit loops are fine too
    if points[i, 0] ** 2 + points[i, 1] ** 2 < radius**2:
      n += 1
  return n
```

<div class="grid cards" markdown>

- **Write NumPy, get loops.** Array expressions, slices, comprehensions and
  explicit loops all compile to the same fused native code — with NumPy's exact
  result types.
- **Every core, or the GPU.** A built-in thread pool, OpenMP or GCD on CPUs;
  Metal on Apple GPUs; CUDA on NVIDIA GPUs. Loops that cannot run on the GPU fall
  back to the CPU and tell you why.
- **Fails loudly, not strangely.** Unsupported code is rejected at compile time
  with your source line; bad indices and shapes raise ordinary Python exceptions.
- **Nothing to set up.** `pip install parakeet` plus a C compiler. Compiled code
  is cached on disk.

</div>

## Install

```bash
pip install parakeet                 # Python 3.11+, NumPy 2, a C compiler
pip install "parakeet[metal]"        # + Apple GPU backend
pip install "parakeet[cuda12]"       # + NVIDIA GPU backend (CUDA 12)
```

Then continue with [Getting started](quickstart.md).

## When to use it

Parakeet shines when NumPy code spends its time allocating temporaries
(`a * b + c * d`), when the natural formulation is a loop that NumPy cannot
express (stencils with conditions, per-pixel iteration, pairwise kernels), or
when you want that loop on all cores or a GPU without rewriting it.

It is **not** a general Python compiler — it compiles a statically typed,
array-oriented [subset](language.md) — and it does not replace BLAS/LAPACK for
dense linear algebra. See [Performance](performance.md) for measurements,
including cases where plain NumPy is faster.
