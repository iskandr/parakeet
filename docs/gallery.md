# Kernel gallery

The same `@parakeet.jit` compiles broadcasting NumPy and explicit data-parallel
operators. Use `dsl.map` when the inputs line up row by row, `dsl.allpairs`
when every left row meets every right row, and `dsl.fold` when you can describe
how two partial results combine. `dsl.kernel` is an alias for `parakeet.jit`.

Start with [one nearest-neighbor query](#exact-nearest-neighbor), then
[change what the fold remembers](#inverse-distance-smoothing). For a complete
training program, try [nanoGPT](gallery/nanogpt.md). The snippets
come from executable examples. From an installed source checkout, run
`python -m examples.algebraic_gallery` for the numerical checks.

```python
import numpy as np
import parakeet
from parakeet import dsl
from typing import NamedTuple
```

The axis-fold examples below use the development version after 1.0.0. To run them,
install the current checkout with `python -m pip install -e .`.

## Exact nearest neighbor

Given `Q` query rows and `R` reference rows with the same feature count,
return `Q` reference indices. For a query at 1 and references at 0, 2 and 4,
the squared distances are `[1, 1, 9]`. The answer is index `0` because it
is the first minimum.

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

Read the explicit form from `nearest_fold`. `allpairs` defines a logical
`Q × R` array of squared distances. Folding along `axis=1` visits the
references for one query. With `indexed=True`, `Winner` receives both the
distance and its column index. Two adjacent groups can each report a winner.
`first_minimum` keeps the left winner on ties and preserves the first NaN.
Finally, `emit` extracts the index. This merge rule lets the compiler process
tiles without storing the full distance matrix.

With the axis-fold extension, both forms use the pairwise tile schedule on supported CPU,
Metal and CUDA targets. The published 1.0.0 and post-release comparison used
the nested-map spelling for the explicit implementation. [Measured timings,
allocations and revision details](release-benchmarks.md#gpu-pairwise-fold-after-100).

## Inverse-distance smoothing

Keep the query/reference pairs and `squared_distance` from nearest neighbor.
Now each reference carries a scalar observation. A point at squared distance
`d2` casts a vote with weight `1 / (1 + d2)`. To combine two groups, add their
weight totals and their weighted observation totals. Divide once, after all
groups have contributed. This implements a weighted mean for each query over
a nonempty reference set.

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

Predict each query's value by weighting the reference values by distance. The
maximum, weight sum and weighted output can be updated a tile at a time. This
is also the reduction at the heart of softmax attention.

```python
--8<-- "examples/algebraic_gallery.py:rbf_kernel_regression"
```

## Histograms and tree splits

Accumulate gradients and Hessians into the same bins. When their index
expressions agree, the compiler can combine the two histogram traversals.
Prefix sums then give the score for each possible split.

```python
--8<-- "examples/algebraic_gallery.py:gbdt_best_split"
```

## Min-plus matrix product

Replace multiplication with addition and addition with minimum. Each output
cell takes the cheapest two-leg path through an intermediate vertex. For an
`m × k` matrix and a `k × n` matrix, the result has shape `m × n`. The indexed
reduction avoids an `m × n × k` temporary.

```python
--8<-- "examples/algebraic_gallery.py:tropical_min_plus"
```

## Attention and linear cross-entropy

Both computations can normalize scores over tiles. In attention, the running
state includes the weighted value vector. Cross-entropy needs the log
normalizer and the target score. The backward pass can recompute the scores
from the inputs.

=== "Causal attention"

    ```python
    --8<-- "examples/algebraic_gallery.py:causal_flash_attention"
    ```

=== "Linear cross-entropy"

    ```python
    --8<-- "examples/algebraic_gallery.py:cut_cross_entropy"
    ```

For a sliding window, restrict the mask to keys within the window. When the
mask and fill permit it, whole tiles can be skipped. [Algebraic laws](algebra.md).

## Shared keys and compressed attention

DeepSeek's multi-head latent attention writes expanded keys and values as
matrix products. When the shape makes it cheaper, the compiler reassociates
these products to operate on compressed values. Grouped-query attention
shares keys and values across several query heads.

=== "Latent attention"

    ```python
    --8<-- "examples/algebraic_gallery.py:deepseek_mla"
    ```

=== "Grouped-query attention"

    ```python
    --8<-- "examples/algebraic_gallery.py:gqa_attention"
    ```

## Affine scans and state-space models

Two successive affine updates have an affine composition. A block of updates
can summarize its effect on the incoming state, then blocks can combine their
summaries in sequence order. The scalar filter and matrix-state recurrence
below expose two uses of this rule.

=== "Scalar IIR filter"

    ```python
    --8<-- "examples/algebraic_gallery.py:iir_exponential_filter"
    ```

=== "Mamba-2 recurrence"

    ```python
    --8<-- "examples/algebraic_gallery.py:mamba2_ssd"
    ```

## Inspect a rewrite

Pass example arrays to the inspector to see which laws fired and compare the
operator IR before and after compilation. This example uses the Gaussian
regression function above.

```python
rng = np.random.default_rng(0)
q = rng.standard_normal((64, 8), dtype=np.float32)
x = rng.standard_normal((256, 8), dtype=np.float32)
y = rng.standard_normal((256, 4), dtype=np.float32)
print(dsl.show_rewrite(rbf_kernel_regression, q, x, y, np.float32(-1)))
```

## Gallery measurements

These earlier Apple Silicon measurements hold the CPU backend constant, with
BLAS disabled, and toggle one compiler pass at a time. The release benchmark
page compares complete implementations across libraries and provides the raw
samples. [Release benchmarks](release-benchmarks.md).

| Workload | Pass disabled | Pass enabled | Pass |
|---|---:|---:|---|
| Mamba-2 SSD, length 2,048, widths 64 | 58.25 ms | 1.08 ms | Chunkwise scan |
| Latent attention, 4 queries and 2,048 keys, widths 64 and 256 | 1.86 ms | 0.22 ms | Contraction reassociation |
| Affine IIR filter, length 262,144 | 0.63 ms | 0.27 ms | Chunkwise scan |
