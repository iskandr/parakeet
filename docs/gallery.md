# Kernel gallery

These examples show how short NumPy functions and explicit data-parallel
operators compile into fused, tiled native kernels.

Start with [one nearest-neighbor query](#exact-nearest-neighbor), then
[change what the fold remembers](#inverse-distance-smoothing). For a complete
training program, try [nanoGPT](gallery/nanogpt.md). Every snippet below is
included directly from executable examples (`python -m examples.algebraic_gallery`).

```python
import numpy as np
import parakeet
from parakeet import dsl
from typing import NamedTuple
```

The same `@parakeet.jit` decorator compiles both ordinary NumPy and explicit
operators such as `dsl.map`, `dsl.allpairs`, `dsl.imap`, `dsl.ireduce`,
`dsl.iscatter`, and `dsl.fold` (`dsl.kernel` is an alias for `parakeet.jit`).
[Browse the source](https://github.com/iskandr/parakeet/blob/master/examples/algebraic_gallery.py).

The axis-fold examples below use the development version after 1.0.0. To run
them, install the current checkout with `python -m pip install -e .`.

## Exact nearest neighbor

Given `Q` query rows and `R` reference rows with the same feature count, return
the index of the closest reference to each query. For a query at 1 and
references at 0, 2, and 4, the squared distances are `[1, 1, 9]`, so the answer
is index `0` because ties keep the earlier index (and the first NaN wins,
matching NumPy).

=== "Broadcasting NumPy"

    ```python
    --8<-- "examples/algebraic_gallery.py:nearest_numpy"
    ```

=== "Map and argmin"

    For each query, map the same distance function over the references and
    return the position of its smallest value. The outer map repeats that
    search for every query.

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:exact_1nn"
    ```

=== "All pairs and a structured fold"

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:nearest_fold"
    ```

In `nearest_fold`, `allpairs` defines the logical `Q × R` domain of squared
distances, `Winner` pairs each distance with its reference index, and
`first_minimum` merges adjacent winners while keeping left ties and the first
NaN. With the axis-fold extension, both forms compile to the pairwise tile
schedule on CPU, Metal, and CUDA without materializing the full `Q × R`
distance matrix. The published 1.0.0 and post-release comparisons used the
nested-map spelling for the explicit implementation. [Measured timings,
allocations and revision details](release-benchmarks.md#gpu-pairwise-fold-after-100).

## Inverse-distance smoothing

Use the same squared distances to predict a scalar observation at each query.
Each reference contributes `(w, w * value)` with `w = 1 / (1 + distance²)`. The
compiler computes each score once, sums both fields over each tile, and divides
the two totals after the final merge.

=== "Broadcasting NumPy"

    ```python
    --8<-- "examples/algebraic_gallery.py:weighted_numpy"
    ```

=== "Parallel operators"

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:weighted_fold"
    ```

## nanoGPT

Train a two-layer transformer to predict the next character in a short text.
The [complete nanoGPT walkthrough](gallery/nanogpt.md) shows the entire runnable
program in order, from the first import to the final training step. It follows
one input sequence through causal attention, then uses `dsl.map` and
`value_and_grad` to train on a batch.

## Gaussian kernel regression

Predict each query's value by weighting reference values by squared distance
(Nadaraya-Watson smoothing). The running maximum, weight sum, and weighted
output form an associative monoid, so the compiler streams references in tiles
instead of allocating an `N × M` weight matrix. That is the same monoidal
reduction at the heart of softmax attention.

```python
--8<-- "examples/algebraic_gallery.py:rbf_kernel_regression"
```

## Histograms and tree splits

Accumulate gradients and Hessians into the same bins, take prefix sums, and
pick the best split — the core loop of histogram-based gradient boosted trees.
When the two scatter-adds share the same bin-index expression, Parakeet fuses
them into a single traversal of the observations.

```python
--8<-- "examples/algebraic_gallery.py:gbdt_best_split"
```

## Min-plus matrix product

Replace multiplication with addition and addition with minimum. Each cell takes
the shortest two-leg path through an intermediate vertex. Writing the reduction
directly over `(m, n)` and `k` avoids an `m × n × k` temporary and triggers the
same cache and register tiling as ordinary matrix multiplication.

```python
--8<-- "examples/algebraic_gallery.py:tropical_min_plus"
```

## Attention and linear cross-entropy

Both computations normalize scores over a large inner dimension (`T` keys or `V`
vocabulary entries) and fold them into a much smaller output. Tiling the
monoidal reduction avoids materializing the `T × T` attention matrix or the
`M × V` logit matrix, and the backward pass recomputes score tiles from the
inputs instead of saving them on the tape.

=== "Causal attention"

    ```python
    --8<-- "examples/algebraic_gallery.py:causal_flash_attention"
    ```

=== "Linear cross-entropy"

    ```python
    --8<-- "examples/algebraic_gallery.py:cut_cross_entropy"
    ```

For a sliding window, restrict the mask to keys within the window. When
interval analysis proves a whole tile is masked out, the tile is skipped before
loading keys or values. See [Algebraic laws](algebra.md).

## Shared keys and compressed attention

Multi-head latent attention writes expanded keys and values as matrix products
with a shared low-rank latent `c_kv`. When the query length is small enough
that reassociating the products saves FLOPs, the compiler absorbs `w_uk` into
`q` and applies `w_uv` after attention — operating directly on `c_kv`.
Grouped-query attention folds the query-head group into the query tile so
shared keys and values are loaded once per group.

=== "Latent attention"

    ```python
    --8<-- "examples/algebraic_gallery.py:deepseek_mla"
    ```

=== "Grouped-query attention"

    ```python
    --8<-- "examples/algebraic_gallery.py:gqa_attention"
    ```

## Affine scans and state-space models

Two successive affine updates compose into another affine update. That lets
Parakeet split a plain Python `for t in range(T):` recurrence into fixed-size
chunks, summarize each chunk in parallel, scan the much shorter sequence of
chunk summaries, and finish each chunk independently. The same idea handles
both a scalar exponential filter and a matrix-state recurrence.

=== "Scalar IIR filter"

    ```python
    --8<-- "examples/algebraic_gallery.py:iir_exponential_filter"
    ```

=== "Mamba-2 recurrence"

    ```python
    --8<-- "examples/algebraic_gallery.py:mamba2_ssd"
    ```

## Inspect a rewrite

Pass example arrays to `dsl.show_rewrite` to see which algebraic laws fired and
compare the operator IR before and after the middle-end passes.

```python
rng = np.random.default_rng(0)
q = rng.standard_normal((64, 8), dtype=np.float32)
x = rng.standard_normal((256, 8), dtype=np.float32)
y = rng.standard_normal((256, 4), dtype=np.float32)
print(dsl.show_rewrite(rbf_kernel_regression, q, x, y, np.float32(-1)).summary())
```

## Classic Parakeet examples

The repository also includes the classic numerical, image-processing, and
training scripts in [`examples/`](https://github.com/iskandr/parakeet/tree/v1.0.0/examples) — many carried forward from Parakeet 0.x:

| Script | What it demonstrates |
|---|---|
| [`tanh_rescale.py`](https://github.com/iskandr/parakeet/blob/v1.0.0/examples/tanh_rescale.py) | NumPy expression, explicit loop, and comprehension compiling to the same fused loop |
| [`matmult.py`](https://github.com/iskandr/parakeet/blob/v1.0.0/examples/matmult.py) | Matrix multiplication four ways: `@`, `allpairs`, comprehension, and loops |
| [`kmeans.py`](https://github.com/iskandr/parakeet/blob/v1.0.0/examples/kmeans.py) | K-means clustering with comprehensions and no materialized distance matrix |
| [`julia.py`](https://github.com/iskandr/parakeet/blob/v1.0.0/examples/julia.py) | Data-parallel map over pixels with a data-dependent inner loop |
| [`growcut.py`](https://github.com/iskandr/parakeet/blob/v1.0.0/examples/growcut.py) | GrowCut image-segmentation cellular automaton |
| [`finite_difference.py`](https://github.com/iskandr/parakeet/blob/v1.0.0/examples/finite_difference.py) | 2-D wave equation stencil with explicit loops |
| [`convolution.py`](https://github.com/iskandr/parakeet/blob/v1.0.0/examples/convolution.py) | 3×3 image convolution via nested comprehensions and helper inlining |
| [`decision_tree.py`](https://github.com/iskandr/parakeet/blob/v1.0.0/examples/decision_tree.py) | Parallel feature-wise split search laid out like CudaTree |
| [`convnet.py`](https://github.com/iskandr/parakeet/blob/v1.0.0/examples/convnet.py) | Small convolutional network trained end-to-end with `parakeet.grad` |
| [`nanogpt.py`](https://github.com/iskandr/parakeet/blob/v1.0.0/examples/nanogpt.py) | Character-level transformer trained with `value_and_grad` |

## Pass-ablation timings

These Apple Silicon measurements hold the CPU backend constant (with BLAS
disabled) and toggle a single middle-end pass at a time, isolating what each
rewrite contributes. For head-to-head comparisons across libraries and raw
measurement files, see [Release benchmarks](release-benchmarks.md).

| Workload | Pass disabled | Pass enabled | Pass |
|---|---:|---:|---|
| Mamba-2 recurrence, length 2,048, widths 64 | 58.25 ms | 1.08 ms | Chunkwise scan |
| Latent attention, 4 queries and 2,048 keys, widths 64 and 256 | 1.86 ms | 0.22 ms | Contraction reassociation |
| Affine IIR filter, length 262,144 | 0.63 ms | 0.27 ms | Chunkwise scan |
<<<<<<< HEAD
=======

??? details "Earlier training-step timings on an NVIDIA L4"

    These warmed end-to-end `examples/nanogpt.py` timings include forward,
    backward, and Adam. The small model has batch 16, sequence length 64, width
    128, and two layers. The medium model has batch 32, sequence length 128,
    width 256, and four layers.

    | Model | Parakeet TF32 | Parakeet FP32 | PyTorch eager | Compiled PyTorch | JAX JIT |
    |---|---:|---:|---:|---:|---:|
    | Small | 6.44 ms | 8.04 ms | 9.15 ms | 3.58 ms | 1.58 ms |
    | Medium | 32.15 ms | 47.05 ms | 19.28 ms | 15.15 ms | 9.18 ms |

    [Training benchmark source](https://github.com/iskandr/parakeet/blob/v1.0.0/tools/colab_nanogpt_bench.py).
>>>>>>> 9de8781 (De-slop docs, fix unrendered math and stale IR descriptions, and emphasize 2012-2014 Parakeet continuity)
