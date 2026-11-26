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

Write ordinary NumPy, loops, comprehensions, or explicit data-parallel
operators (`map`, `reduce`, `scan`, `allpairs`). Parakeet specializes your
function for its input types, fuses and tiles the array algebra, and compiles
native code for multi-core CPUs, Apple Metal, and NVIDIA CUDA.

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

Parakeet computes distances a tile at a time on both CPU and GPU, keeping only
the winning reference index per tile rather than allocating the full `N × M`
distance matrix. Ties preserve NumPy's first-index rule.

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

## Algebraic primitives

Rather than shipping hundreds of hand-written C or GPU kernels for individual
NumPy functions, Parakeet lowers ordinary Python, NumPy, and `par.*` calls onto
**seven composable operators**. Inside the compiler IR, each operator carries an
`Index` prefix (`IndexMap`, `IndexReduce`, ...) to mark that it runs over an
explicit N-D coordinate space, while your code stays idiomatic NumPy or `par.*`.
[Read the full primitives guide and end-to-end compilation walkthrough](primitives.md).

<div class="pk-prim-grid" markdown>

<div class="pk-prim-card" markdown>
<div class="pk-prim-head"><strong>Map</strong> <code>IndexMap</code></div>
Independent per-element work over an N-D index space: broadcasting, comprehensions, <code>par.map</code>, <code>par.allpairs</code>.
```python
np.tanh(x * 0.5) + 0.3
```
[Details & fusion rules](primitives.md#map)
</div>

<div class="pk-prim-card" markdown>
<div class="pk-prim-head"><strong>Reduce &amp; Fold</strong> <code>IndexReduce</code></div>
Associative reductions (<code>np.sum</code>, <code>par.reduce</code>) and ordered multi-state folds (<code>np.argmin</code>, <code>par.fold</code>).
```python
np.argmin(d2, axis=1)
```
[Details &amp; <code>reduce</code> vs <code>fold</code>](primitives.md#reduce-and-fold)
</div>

<div class="pk-prim-card" markdown>
<div class="pk-prim-head"><strong>Scan</strong> <code>IndexScan</code></div>
Prefix sums (<code>np.cumsum</code>, <code>par.scan</code>) and affine/bilinear <code>for</code> recurrences (IIR filters, Mamba-2 SSMs).
```python
np.cumsum(x)  # or affine for-loop
```
[Details &amp; chunkwise duality](primitives.md#scan)
</div>

<div class="pk-prim-card" markdown>
<div class="pk-prim-head"><strong>Filter &amp; Expand</strong> <code>IndexFilter</code> <code>IndexExpand</code></div>
Stream compaction (1-to-0/1 boolean masking) and 2-pass ragged expansion (1-to-K <code>par.flatmap</code>).
```python
np.sum(x[x > 0.0])  # 0-alloc FilterReduce
```
[Details &amp; filter fusion](primitives.md#filter-and-expand)
</div>

<div class="pk-prim-card" markdown>
<div class="pk-prim-head"><strong>Scatter</strong> <code>IndexScatter</code></div>
Associative bin accumulation (<code>np.bincount</code>, <code>np.add.at</code>, <code>out[idx[i]] += v</code>, <code>par.scatter_add</code>).
```python
np.bincount(bins, weights=grads, minlength=256)
```
[Details &amp; histogram fusion](primitives.md#scatter)
</div>

<div class="pk-prim-card" markdown>
<div class="pk-prim-head"><strong>Wavefront</strong> <code>IndexWavefront</code></div>
Anti-diagonal and banded 2D dynamic programming for sequence alignment and differentiable DP.
```python
par.soft_align(scores, gamma=1.0)
```
[Details &amp; banded DP](primitives.md#wavefront)
</div>

<div class="pk-prim-card pk-prim-card--wide" markdown>
<div class="pk-prim-head"><strong>Tile</strong> <code>IndexTile</code></div>
Emitted automatically by the middle end for CPU cache/register micro-tiles and GPU shared memory on matrix products, <code>einsum</code>, 1-NN, and online softmax attention.
[Walk through causal attention from 4 lines of NumPy to a single tiled kernel](primitives.md#walkthrough-attention)
</div>

</div>

```python
@par.jit
def scaled_tanh_numpy(x, alpha=0.5, beta=0.3):
  return np.tanh(x * alpha) + beta


@par.jit
def scaled_tanh_comp(x, alpha=0.5, beta=0.3):
  return np.array([np.tanh(xi * alpha) + beta for xi in x])
```

Both versions lower to the same single fused parallel loop with no temporary
arrays. Because programs stay in this operator representation until code
generation, a small set of general passes does the optimization work across
domains:

- **Fusion.** Producer-consumer chains, filtered reductions (`np.sum(x[x > 0])`),
  and sibling histograms sharing the same bin IDs merge into one traversal.
  [See the tree-split example](gallery.md#histograms-and-tree-splits).
- **Tiling and single-pass reductions.** Associative reductions and bilinear
  products tile into CPU cache or GPU shared memory, so Gaussian kernel
  regression, nearest-neighbor search, and attention stream blocks instead of
  allocating giant pairwise matrices. [Read the algebraic laws](algebra.md) or
  [follow the worked compilation walkthrough](primitives.md#worked-end-to-end-compilation-walkthroughs).
- **Differentiation.** `par.grad`, `par.value_and_grad`, and `par.jvp`
  differentiate the primitives directly, compiling a full training step into
  one native function. [Learn about gradients](autodiff.md).
- **Inspection.** Every stage is visible: inspect the operator IR, see which
  rewrites fired with `par.show_rewrite`, or read the generated C, Metal, or
  CUDA source. [Inspect a rewrite](gallery.md#inspecting-algebraic-rewrites).

## Where to go next

- **New to Parakeet?** Start with [Getting started](quickstart.md): five
  minutes from `pip install` to your first compiled function and generated C.
- **Learn by example:** Browse the [Kernel gallery](gallery.md) for runnable
  NumPy and data-parallel kernels, or check the [Release benchmarks](release-benchmarks.md).
- **Go deeper:** Read [Ideas, history, and boundaries](philosophy.md) for the
  design principles, project history, and what Parakeet does and does not
  support.
