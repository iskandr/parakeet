# Kernel gallery

These examples compile ordinary NumPy functions and explicit data-parallel operators.
Each code sample comes from the runnable gallery. From an installed source checkout,
run the examples and their numerical checks with:

```bash
python examples/algebraic_gallery.py
```

The examples use `import numpy as np`, `import parakeet` and
`from parakeet import dsl`. [Browse the source](https://github.com/iskandr/parakeet/blob/v1.0.0/examples/algebraic_gallery.py).

## Exact nearest neighbor

Find the closest reference row for each query. The outer map assigns one output
per query. The inner map computes distances, and `argmin` returns the first
reference index at the minimum.

```python
--8<-- "examples/algebraic_gallery.py:exact_1nn"
```

The CPU compiler fuses the score and reduction in both this form and its
broadcast NumPy equivalent. At 1,024 queries by 4,096 references, each active
query strip needs an 8 KiB tile and 192 bytes of row summaries. In the release
benchmark, the broadcast CUDA version used a 16 MiB distance matrix and the
nested DSL required CPU fallback. [Timings and allocation details](release-benchmarks.md#pairwise-workloads).

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

??? details "Earlier training-step timings on an NVIDIA L4"

    These warmed end-to-end nanoGPT timings include forward, backward and Adam.
    The small model has batch 16, sequence length 64, width 128 and two layers.
    The medium model has batch 32, sequence length 128, width 256 and four layers.

    | Model | Parakeet TF32 | Parakeet FP32 | PyTorch eager | Compiled PyTorch | JAX JIT |
    |---|---:|---:|---:|---:|---:|
    | Small | 6.44 ms | 8.04 ms | 9.15 ms | 3.58 ms | 1.58 ms |
    | Medium | 32.15 ms | 47.05 ms | 19.28 ms | 15.15 ms | 9.18 ms |

    [Training benchmark source](https://github.com/iskandr/parakeet/blob/v1.0.0/tools/colab_nanogpt_bench.py).
