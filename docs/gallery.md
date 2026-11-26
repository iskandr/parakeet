# Kernel gallery

Parakeet compiles ordinary NumPy code, plain Python loops, and explicit
data-parallel operators (`par.map`, `par.allpairs`, `par.fold`, `par.imap`,
`par.ireduce`, `par.scatter_add`) through the **same** `@par.jit` decorator and
the **same** algebraic IR.

**How this page is organized:** the examples progress from straightforward
pairwise NumPy fusion and histograms to custom semiring reductions, parallel
recurrences, and modern attention and loss kernels; introducing Parakeet's
capabilities one layer at a time. Where a section has tabs, those tabs show
**alternative ways to write the exact same computation** (idiomatic NumPy vs.
plain loops vs. explicit Parakeet operators) that all compile to the same
fused, tiled schedule.

For a complete end-to-end training walkthrough, see
[nanoGPT](gallery/nanogpt.md). For genomics, cheminformatics, and 3D protein
kernels, see [Bioinformatics & structural biology](bioinformatics.md). For the
underlying identities behind these rewrites, see [Algebraic laws](algebra.md).

Every snippet below is included directly from
[`examples/algebraic_gallery.py`](https://github.com/iskandr/parakeet/blob/master/examples/algebraic_gallery.py)
and verified by `python -m examples.algebraic_gallery`.

```python
from typing import NamedTuple
import numpy as np
import parakeet as par
```

---

## 1. Pairwise fusion and automatic tiling

### Exact nearest neighbor

Exact nearest-neighbor search finds, for each of `Q` query vectors in `R^D`,
the index of the closest vector among `R` reference vectors under Euclidean
distance. Writing this with broadcasting NumPy is two lines, but executing it
eagerly allocates a full `Q × R` distance matrix (`64 MiB` at `Q = 1,024,
R = 16,384`) that thrashes cache and GPU memory.

Parakeet fuses the distance calculation directly into an ordered tiled
reduction across `R`, keeping only a small score tile in L1 cache or GPU shared
memory (`16 KiB` per block) and preserving NumPy's exact first-index rule on
equal-distance ties and NaNs.

=== "Broadcasting NumPy"

    ```python
    --8<-- "examples/algebraic_gallery.py:nearest_numpy"
    ```

=== "Nested `par.map` and `np.argmin`"

    For each query row, map the squared distance over all reference rows and
    take `np.argmin`. The outer `par.map` runs that search in parallel across
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

All three spellings compile to the same tiled pairwise schedule on CPU, Metal,
and CUDA without allocating the `Q × R` distance matrix.
[Compare timings, scaling, and scratch memory](release-benchmarks.md#scaling-exact-1-nn-with-reference-count).

### Inverse-distance smoothing

Swap the fold state from `(min_distance, index)` to `(weight_sum, weighted_sum)`
and the same squared distances predict a smoothed scalar observation at each
query with weight `w[i, j] = 1 / (1 + ||q[i] - r[j]||²)`. Parakeet computes each
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

### Gaussian kernel regression

Nadaraya-Watson kernel regression predicts each query's target vector by
weighting reference observations with a Gaussian kernel
`exp(-gamma * ||q[i] - x[j]||²)`. Because the running maximum, weight sum, and
weighted output form an associative
[rescaled monoid](algebra.md#law-1-rescaled-semiring-monoids), Parakeet streams
references in tiles instead of allocating an `N × M` weight matrix, using the
same online monoidal pass that powers tiled attention below.

```python
--8<-- "examples/algebraic_gallery.py:rbf_kernel_regression"
```

---

## 2. Histograms and scatter-add fusion

### Histograms and tree splits

Histogram-based gradient boosted trees accumulate gradients and Hessians into
feature bins, scan cumulative left/right sums, and pick the split bin with the
highest gain. You can write the bin accumulations as a declarative NumPy
comprehension, with `np.add.at`, as a plain Python `for` loop, or with explicit
`par.scatter_add` (including keyed generator expressions); all four lower to
Parakeet's `O(N)` `IndexScatter` operator. Because both accumulations share the
same `bin_ids` array, Parakeet fuses them into a single
[thread-privatized traversal](algebra.md#law-6-privatized-scatter-multi-state-carries)
over the observations.

=== "Declarative comprehension"

    A pull-style comprehension `[np.sum(grads[bin_ids == b]) for b in range(nbins)]`
    (or `[sum(grads[i] for i in range(len(bin_ids)) if bin_ids[i] == b) for b in range(nbins)]`)
    would cost `O(nbins * N)` naively. Parakeet inverts it
    automatically to an `O(N)` `IndexScatter` and fuses both histograms into
    one pass.

    ```python
    --8<-- "examples/algebraic_gallery.py:gbdt_best_split_comprehension"
    ```

=== "NumPy `np.add.at`"

    ```python
    --8<-- "examples/algebraic_gallery.py:gbdt_best_split_numpy"
    ```

=== "Python `for` loop"

    When an indexed `+=` loop writes only into fresh buffers that are never
    read inside the loop, Parakeet proves independence, lifts each update to
    `IndexScatter`, and fuses sibling histograms sharing `b = bin_ids[i]` into
    one parallel traversal automatically.

    ```python
    --8<-- "examples/algebraic_gallery.py:gbdt_best_split_loop"
    ```

=== "Explicit `par.scatter_add`"

    ```python
    --8<-- "examples/algebraic_gallery.py:gbdt_best_split"
    ```

---

## 3. Custom semirings with index operators

### Min-plus matrix product

In shortest-path and tropical algebra, matrix multiplication replaces `(*, +)`
with `(+, min)`: `C[i, j] = min_r (A[i, r] + B[r, j])`. Writing the index space
explicitly with `par.imap` and `par.ireduce(..., combine=np.minimum)` avoids an
`m × n × k` broadcast tensor and triggers the same cache and register tiling as
standard matrix multiplication.

```python
--8<-- "examples/algebraic_gallery.py:tropical_min_plus"
```

---

## 4. Sequential recurrences to parallel scans

### Exponential IIR filter

A plain Python `for t in range(T):` affine recurrence `s = a[t] * s + b[t]`
has `O(T)` serial depth. Because affine maps compose associatively, Parakeet's
[chunkwise scan pass](algebra.md#law-4-chunkwise-state-space-duality) rewrites
the loop into a parallel prefix scan (`TileScan`) across CPU cores or GPU
threadblocks with no `O(T)` scratch arrays.

```python
--8<-- "examples/algebraic_gallery.py:iir_exponential_filter"
```

### Mamba-2 state-space recurrence

The same chunkwise state-space duality generalizes to bilinear matrix-state
recurrences: a 6-line Python loop over time rewrites into division-free
chunkwise batched GEMMs that automatically lower to CPU register tiling, Metal
`simdgroup_matrix`, and CUDA TF32 WMMA.

```python
--8<-- "examples/algebraic_gallery.py:mamba2_ssd"
```

---

## 5. Attention and large-vocabulary losses

### Causal attention

Write scaled dot-product attention with a causal mask in six lines of NumPy.
Three general passes compose automatically:

* **Online monoidal tiling** fuses the softmax-weighted contraction into a
  single streaming pass with no `T × T` score matrix
  ([details](algebra.md#law-1-rescaled-semiring-monoids)).
* **Tile-interval mask pruning** proves upper-triangular tiles are completely
  masked and skips loading their keys and values
  ([details](algebra.md#law-5-affine-tile-interval-mask-pruning)).
* **Rematerialized differentiation** recomputes score tiles on the fly during
  `par.grad`, `par.jvp`, and `par.hvp` instead of storing `T × T` tensors
  ([details](algebra.md#law-2-algebraic-rematerialization-in-reverse-mode-ad)).

```python
--8<-- "examples/algebraic_gallery.py:causal_flash_attention"
```

For a sliding window, intersect the mask with `(idx[:, None] - idx[None, :]) <= window`;
interval analysis skips tiles outside the window automatically.

### Linear cross-entropy

Computing cross-entropy over a large vocabulary `V` from hidden states
`h @ w` normally materializes an `M × V` logit matrix. Online monoidal tiling
chunks the contraction and `logsumexp` over the vocabulary while gathering the
target logit on the fly, and algebraic rematerialization differentiates the
loss with zero `M × V` memory.

```python
--8<-- "examples/algebraic_gallery.py:cut_cross_entropy"
```

### Multi-head latent attention

Multi-head latent attention (MLA) projects keys and values from a shared
low-rank latent `c_kv`. When the query length is small (such as autoregressive
decoding), [bilinear contraction reassociation](algebra.md#law-3-bilinear-contraction-reassociation-shared-kv-tiling)
absorbs `w_uk` into `q` and defers `w_uv` until after attention; running
tiled attention directly over the compressed latent `c_kv`.

```python
--8<-- "examples/algebraic_gallery.py:deepseek_mla"
```

### Grouped-query attention

When a head-group axis `G` in `q` broadcasts against a single shared KV head
(`1` in `k` and `v`), shared-KV head-group tiling folds `G` into the query tile
so all `G` query heads reuse one shared-memory load of `k` and `v`.

```python
--8<-- "examples/algebraic_gallery.py:gqa_attention"
```

---

## 6. End-to-end training and inspecting rewrites

### Dataclasses, methods, and polymorphic dispatch

Parakeet lowers `@dataclass` records (frozen or non-frozen), pure methods,
`@property`, `__call__`, structural `match`/`case`, compile-time `isinstance`,
`functools.partial`, and `*args`/`**kwargs` away completely during elaboration,
letting you write idiomatic scientific Python types that still compile to
fused, tiled, differentiable native code (`examples/protein_and_layers.py`):

```python
--8<-- "examples/protein_and_layers.py:protein_and_layers"
```

### Unboxed complex FFT, Wirtinger AD, and FASTA/CSV byte parsing

`examples/complex_fft_and_fasta.py` shows an iterative Cooley-Tukey radix-2 FFT
and Wirtinger gradient (`par.value_and_grad`) in unboxed `complex64` alongside
zero-`PyObject` FASTA/CSV byte ingestion via `par.split_bytes`, byte LUT
remapping, and `par.parse_float32`.

### Adaptive CudaTree BFS -> DFS GBDT

`examples/cudatree_gbdt.py` combines level-wise BFS histogram construction with
histogram subtraction (`large_child = parent - small_child`),
`par.partition_by_size`, `par.ragged_partition`, and a single fused
thread-per-subtree DFS completion pass using thread-local `par.fixed_list`
scratch stacks (`benchmarks/tree_models.py` benchmarks balanced and skewed
workloads).

### nanoGPT end-to-end training

Train a two-layer character-level transformer from scratch in pure NumPy +
Parakeet. The [complete nanoGPT walkthrough](gallery/nanogpt.md) follows the
entire runnable script in order, using `par.map` and `par.value_and_grad` to
compile the forward pass, backward pass, and Adam optimizer into one native
step.

### Inspecting algebraic rewrites

Call `par.show_rewrite(fn, *example_args)` on any compiled or plain Python
function to inspect which compiler optimizations fired and compare the operator
IR before and after optimization:

```python
rng = np.random.default_rng(0)
q = rng.standard_normal((64, 8), dtype=np.float32)
x = rng.standard_normal((256, 8), dtype=np.float32)
y = rng.standard_normal((256, 4), dtype=np.float32)
print(par.show_rewrite(rbf_kernel_regression, q, x, y, np.float32(-1)).summary())
```
