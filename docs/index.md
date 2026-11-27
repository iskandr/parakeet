---
title: Parakeet
description: Compile numerical Python and NumPy into fused native code for multi-core CPUs, Apple Metal, and NVIDIA CUDA.
hide:
  - navigation
  - toc
---

<div class="pk-intro" markdown>
<div class="pk-identity" markdown>

# ![Parakeet](assets/logo.png){ .pk-wordmark width="1746" height="520" }

<p class="pk-lead">A data-parallel compiler for numerical Python.</p>

Write ordinary NumPy, Python loops, comprehensions, or explicit data-parallel
operators ([`par.map`](operators.md#map), [`par.reduce`](operators.md#reduce),
[`par.fold`](operators.md#fold), [`par.scan`](operators.md#scan),
[`par.allpairs`](operators.md#allpairs)). Parakeet specializes your function
for its input types, fuses and tiles the array math, and compiles native code
for multi-core CPUs, Apple Metal, and NVIDIA CUDA.

[Get started](quickstart.md){ .md-button .md-button--primary }
[Explore the gallery](gallery.md){ .md-button }

</div>
<div class="pk-example" markdown>

## Exact nearest neighbor

```python
import numpy as np
import parakeet as par


@par.jit
def nearest(x, y):
  delta = x[:, None, :] - y[None, :, :]
  d2 = np.sum(delta ** np.float32(2), axis=2)
  return np.argmin(d2, axis=1)
```

Parakeet streams distances a tile at a time on both CPU and GPU, keeping only
the winning index per tile rather than allocating the `N × M` distance matrix.
Ties preserve NumPy's exact first-index rule.

[Compare timings, scaling, and memory](release-benchmarks.md#at-a-glance-nvidia-l4-4-thread-cpu)

</div>
</div>

## Install

```bash
python -m pip install parakeet
```

Requires Python 3.11+, NumPy 2, and a system C compiler. For GPU backends,
install `parakeet[metal]` on Apple Silicon or `parakeet[cuda12]` with an NVIDIA
GPU.

## What Parakeet gives you

<div class="pk-prim-grid" markdown>

<div class="pk-prim-card" markdown>
<div class="pk-prim-head"><strong>Idiomatic Python &amp; NumPy</strong></div>

Write NumPy expressions, comprehensions, explicit loops, `@dataclass` modules,
or explicit operators (`par.map`, `par.fold`, `par.scan`, `par.scatter_add`,
`par.wavefront`). Unsupported constructs fail loudly with your exact source line.

[Compiled language](language.md) · [Operators](operators.md)
</div>

<div class="pk-prim-card" markdown>
<div class="pk-prim-head"><strong>Automatic Fusion &amp; Tiling</strong></div>

Chained array expressions, boolean filters, same-key histograms, matrix
products, pairwise distances, and online softmax attention fuse and tile into
cache or GPU shared memory without materializing giant intermediates.

[How Parakeet optimizes](primitives.md) · [Algebraic laws](algebra.md)
</div>

<div class="pk-prim-card" markdown>
<div class="pk-prim-head"><strong>Composable Gradients</strong></div>

`par.grad`, `par.value_and_grad`, `par.jvp`, `par.vjp`, and `par.hvp`
differentiate directly through loops, arrays, and `@dataclass` parameters—so
forward pass, backward pass, and optimizer compile into one native step.

[Automatic differentiation](autodiff.md)
</div>

<div class="pk-prim-card" markdown>
<div class="pk-prim-head"><strong>Multi-core CPU, Metal &amp; CUDA</strong></div>

One Python function compiles to sequential C, multi-core threads, OpenMP, GCD,
Apple Metal, or NVIDIA CUDA (with TF32 Tensor Cores). Keep arrays resident on
device across calls with `par.to_device`.

[Backends and GPUs](backends.md) · [Benchmarks](release-benchmarks.md)
</div>

</div>

## Same fused loop, any style

NumPy expressions, comprehensions, and explicit operators compile to the same
single-pass parallel loop with zero temporary arrays:

```python
@par.jit
def scaled_tanh_numpy(x, alpha=0.5, beta=0.3):
  return np.tanh(x * alpha) + beta


@par.jit
def scaled_tanh_comp(x, alpha=0.5, beta=0.3):
  return np.array([np.tanh(xi * alpha) + beta for xi in x])
```

## Where to go next

* **New to Parakeet?** Follow [Getting started](quickstart.md) — five minutes
  from `pip install` to your first compiled function and gradient.
* **Learn by example:** Browse the [Kernel gallery](gallery.md),
  [Train nanoGPT](gallery/nanogpt.md), or
  [Bioinformatics & structural biology](bioinformatics.md).
* **Under the hood:** See [Operators & compilation walkthrough](primitives.md)
  and [Compiler architecture](internals.md) for how Parakeet fuses, tiles, and
  lowers your code.
