# Kernel gallery

Parakeet compiles ordinary NumPy code and explicit data-parallel operators
(`par.map`, `par.allpairs`, `par.fold`, `par.imap`, `par.ireduce`, `par.scatter_add`)
through the **same** `@par.jit` decorator and the **same** algebraic IR.

**How this page is organized:** each section presents one numerical workload.
Where a section has tabs, those tabs show **alternative ways to write the exact
same computation** — broadcasting NumPy vs. explicit Parakeet operators — so
you can compare how both lower to the same fused, tiled schedule.

Start with [exact nearest-neighbor search](#exact-nearest-neighbor), then
[swap the fold state](#inverse-distance-smoothing) to turn search into
kernel smoothing. For a complete end-to-end training script, walk through
[nanoGPT](gallery/nanogpt.md). For genomics, cheminformatics, and 3D protein
kernels, see [Bioinformatics & structural biology](bioinformatics.md).

Every snippet below is included directly from
[`examples/algebraic_gallery.py`](https://github.com/iskandr/parakeet/blob/master/examples/algebraic_gallery.py)
and verified by `python -m examples.algebraic_gallery`.

```python
from typing import NamedTuple
import numpy as np
import parakeet as par
```

## Exact nearest neighbor

Exact nearest-neighbor search finds, for each of `Q` query vectors in
$\mathbb{R}^D$, the index of the closest vector among `R` reference vectors
under Euclidean distance. Writing this with broadcasting NumPy is two lines,
but executing it eagerly allocates a full `Q × R` distance matrix — `64 MiB`
at `Q = 1,024, R = 16,384` — that thrashes cache and GPU memory.

Parakeet fuses the distance calculation directly into an ordered tiled
reduction across `R`, keeping only a small score tile in L1 cache or GPU shared
memory (`16 KiB` per block) and preserving NumPy's exact first-index rule on
equal-distance ties and NaNs.

=== "Broadcasting NumPy"

    ```python
    --8<-- "examples/algebraic_gallery.py:nearest_numpy"
    ```

=== "Nested `par.map` and `argmin`"

    For each query row, map the squared distance over all reference rows and
    take `argmin`. The outer `par.map` runs that search in parallel across
    queries.

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:exact_1nn"
    ```

=== "`par.allpairs` and structured `par.fold`"

    Define the logical `Q × R` domain with `par.allpairs`, pack each
    `(distance, index)` pair into a `NamedTuple`, and reduce with an explicit
    associative merge rule (`first_minimum`).

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:nearest_fold"
    ```

All three spellings compile to the tiled pairwise schedule on CPU, Metal, and
CUDA without allocating the `Q × R` distance matrix.
[Compare timings, scaling, and scratch memory](release-benchmarks.md#scaling-exact-1-nn-with-reference-count).

## Inverse-distance smoothing

Swap the fold state from `(min_distance, index)` to `(weight_sum, weighted_sum)`
and the same squared distances predict a smoothed scalar observation at each
query with weight $w_{ij} = 1 / (1 + \|q_i - r_j\|^2)$. Parakeet computes each
distance once, sums both fields inside each tile, and divides the totals after
the final merge.

=== "Broadcasting NumPy"

    ```python
    --8<-- "examples/algebraic_gallery.py:weighted_numpy"
    ```

=== "`par.allpairs` and `par.fold`"

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:weighted_fold"
    ```

## Gaussian kernel regression

Nadaraya-Watson kernel regression predicts each query's target vector by
weighting reference observations with a Gaussian kernel
$\exp(-\gamma \|q_i - x_j\|^2)$. Because the running maximum, weight sum, and
weighted output form an associative rescaled monoid ([Law 1](algebra.md#law-1-rescaled-semiring-monoids)),
Parakeet streams references in tiles instead of allocating an `N × M` weight
matrix — the exact same monoidal pass that powers FlashAttention.

```python
--8<-- "examples/algebraic_gallery.py:rbf_kernel_regression"
```

## Histograms and tree splits

Histogram-based gradient boosted trees accumulate gradients and Hessians into
feature bins, scan cumulative left/right sums, and pick the split bin with the
highest gain. Because both `par.scatter_add` calls share the same `bin_ids`
index array, Parakeet fuses them into a single privatized traversal over the
observations ([Law 6](algebra.md#law-6-privatized-scatter-multi-state-carries)).

```python
--8<-- "examples/algebraic_gallery.py:gbdt_best_split"
```

## Min-plus matrix product

In shortest-path and tropical algebra, matrix multiplication replaces `(*, +)`
with `(+, min)`: $C_{ij} = \min_r (A_{ir} + B_{rj})$. Writing the inner
reduction explicitly with `par.imap` and `par.ireduce(..., combine=par.minimum)`
avoids an `m × n × k` broadcast tensor and triggers the same cache and register
tiling as standard matrix multiplication.

```python
--8<-- "examples/algebraic_gallery.py:tropical_min_plus"
```

## Causal attention

Write scaled dot-product attention with a causal mask in six lines of NumPy.
Law 1 tiles the softmax-weighted contraction into a single streaming pass with
no `T × T` score matrix, Law 5 proves upper-triangular tiles are completely
masked and skips loading their keys and values, and Law 2 recomputes score
tiles during `par.grad` instead of saving them on the backward tape.

```python
--8<-- "examples/algebraic_gallery.py:causal_flash_attention"
```

For a sliding window, intersect the mask with `(i - j) <= window`; interval
analysis skips tiles outside the window automatically. See
[Algebraic laws](algebra.md).

## Linear cross-entropy

Computing cross-entropy over a large vocabulary `V` from hidden states
`h @ w` normally materializes an `M × V` logit matrix. Law 1 tiles the
contraction and `logsumexp` over vocabulary chunks while gathering the target
logit on the fly, and Law 2 differentiates the loss with zero `M × V` memory.

```python
--8<-- "examples/algebraic_gallery.py:cut_cross_entropy"
```

## Multi-head latent attention

Multi-head latent attention (MLA) projects keys and values from a shared
low-rank latent `c_kv`. When the query length is small (such as autoregressive
decoding), Law 3 reassociates `(q @ (c_kv @ w_uk).T)` by absorbing `w_uk` into
`q` and deferring `w_uv` until after attention — running directly on `c_kv`.

```python
--8<-- "examples/algebraic_gallery.py:deepseek_mla"
```

## Grouped-query attention

When a head-group axis `G` in `q` broadcasts against a single shared KV head
(`1` in `k` and `v`), Law 3 folds `G` into the query tile so all `G` query
heads reuse one shared-memory load of `k` and `v`.

```python
--8<-- "examples/algebraic_gallery.py:gqa_attention"
```

## Exponential IIR filter

A plain Python `for t in range(T):` affine recurrence `s = a[t] * s + b[t]`
has `O(T)` serial depth. Because affine maps compose associatively, Law 4
rewrites the loop into a parallel chunkwise prefix scan (`TileScan`) across
CPU cores or GPU threadblocks with no `O(T)` scratch arrays.

```python
--8<-- "examples/algebraic_gallery.py:iir_exponential_filter"
```

## Mamba-2 state-space recurrence

The same Law 4 chunkwise duality generalizes to bilinear matrix-state
recurrences: a 6-line Python loop over time rewrites into division-free
chunkwise batched GEMMs that automatically lower to CPU register tiling, Metal
`simdgroup_matrix`, and CUDA TF32 WMMA.

```python
--8<-- "examples/algebraic_gallery.py:mamba2_ssd"
```

## nanoGPT end-to-end training

Train a two-layer character-level transformer from scratch in pure NumPy +
Parakeet. The [complete nanoGPT walkthrough](gallery/nanogpt.md) follows the
entire runnable script in order, using `par.map` and `par.value_and_grad` to
compile the forward pass, backward pass, and Adam optimizer into one native
step.

## Inspecting algebraic rewrites

Call `par.show_rewrite(fn, *example_args)` on any compiled or plain Python
function to inspect which algebraic laws fired and compare the operator IR
before and after optimization:

```python
rng = np.random.default_rng(0)
q = rng.standard_normal((64, 8), dtype=np.float32)
x = rng.standard_normal((256, 8), dtype=np.float32)
y = rng.standard_normal((256, 4), dtype=np.float32)
print(par.show_rewrite(rbf_kernel_regression, q, x, y, np.float32(-1)).summary())
```
