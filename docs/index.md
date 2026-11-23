# Parakeet

**A runtime compiler and accelerator for numerical Python.** Decorate an
array-oriented function with `@jit`; the first call compiles a native version
specialised to the argument types, with array expressions, comprehensions,
loops, and data-parallel operators fused and compiled to run on every CPU core
or on a GPU.

Write it whichever way reads best — all three compile to the same fused
parallel loop:

```python
import numpy as np
from parakeet import jit


@jit
def array_expr(x, alpha=0.5, beta=0.3):
  return np.tanh(x * alpha) + beta


@jit
def explicit_loop(x, alpha=0.5, beta=0.3):
  y = np.empty_like(x)
  for i in range(len(x)):
    y[i] = np.tanh(x[i] * alpha) + beta
  return y


@jit
def comprehension(x, alpha=0.5, beta=0.3):
  return np.array([np.tanh(xi * alpha) + beta for xi in x])
```

And express parallel reductions, pairwise kernels, filtered reductions, and
gradients directly with Parakeet's data-parallel operators:

```python
import parakeet as pk

dists = pk.allpairs(lambda u, v: np.sum((u - v) ** 2), X, C)  # tiled, no temporaries
pos_sum = pk.filter_reduce(pk.add, lambda v: v > 0.0, x, init=0.0)  # one pass, no mask array
dloss = pk.grad(lambda w, x: np.mean(np.tanh(x @ w) ** 2))  # reverse-mode AD
```

<div class="grid cards" markdown>

- **Algebraic operator IR.** NumPy expressions, comprehensions, slices and
  loops lower to four data-parallel operators (`IndexMap`, `IndexReduce`,
  `IndexScan`, `IndexFilter`) where fusion, interchange and tiling are simple
  algebraic rewrites — verified after every pass.
- **Every core, or the GPU.** A built-in thread pool, OpenMP or GCD on CPUs;
  Metal on Apple GPUs; CUDA on NVIDIA GPUs. Loops that cannot run on the GPU
  fall back to the CPU and tell you why.
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

Then continue with [Getting started](quickstart.md), or read
[Core ideas, history, and boundaries](philosophy.md) for the PhD thesis design
behind Parakeet and an honest map of what does and does not work yet.

## History and lineage

Parakeet was originally created at New York University by **Alex Rubinsteyn**
([@iskandr](https://github.com/iskandr)) and **Eric Hielscher**
([@hielscher](https://github.com/hielscher)) (formerly at `parakeetpython.com`,
HotPar '12, PyData Boston '13, and the two NYU PhD theses: Rubinsteyn 2014 on
adverb-based runtime compilation and Hielscher 2013 on locality optimization).
Parakeet 1.0 is a modern continuation for Python 3.11+ and NumPy 2 that keeps
the original's data-parallel algebra while adding a verified middle end,
Metal/CUDA backends, and reverse-mode differentiation.

## When to use it

Parakeet shines when NumPy code spends its time allocating temporaries
(`a * b + c * d`), when the natural formulation is a loop or pairwise kernel
that NumPy cannot express without huge intermediates, or when you want that
computation across all CPU cores or a GPU without leaving Python.

It is **not** a general-purpose compiler for all of Python — it compiles a
statically typed, array-oriented [subset](language.md). See
[Core ideas, history, and boundaries](philosophy.md#the-boundary-what-does-not-work-yet)
for exact limits and [Performance](performance.md) for CPU and A100 GPU
measurements (including cases where plain NumPy/BLAS is faster).
