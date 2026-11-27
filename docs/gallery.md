# Kernel gallery

Parakeet compiles NumPy expressions, Python loops, and explicit data-parallel
operators ([`par.map`](operators.md#map), [`par.allpairs`](operators.md#allpairs),
[`par.fold`](operators.md#fold), [`par.imap`](operators.md#imap),
[`par.ireduce`](operators.md#ireduce), [`par.scatter_add`](operators.md#scatter_add))
through one `@par.jit` decorator. Where a section has tabs, each spelling
compiles to the same fused, tiled schedule.

Every snippet below is included verbatim from
[`examples/algebraic_gallery.py`](https://github.com/iskandr/parakeet/blob/master/examples/algebraic_gallery.py)
and verified by `python -m examples.algebraic_gallery`. See also
[Train nanoGPT](gallery/nanogpt.md) and
[Bioinformatics & structural biology](bioinformatics.md).

```python
from typing import NamedTuple
import numpy as np
import parakeet as par
```

---

## 1. Pairwise fusion and automatic tiling

### Exact nearest neighbor

Find the closest reference vector among `R` references for each of `Q` queries
in `Rᴰ`. Naive NumPy allocates a `Q × R` distance matrix (`64 MiB` at
`Q = 1,024, R = 16,384`); Parakeet streams `16 KiB` tiles in L1/SRAM and keeps
NumPy's exact first-index rule on ties and NaNs.

=== "Broadcasting NumPy"

    ```python
    --8<-- "examples/algebraic_gallery.py:nearest_numpy"
    ```

=== "Nested `par.map` and `np.argmin`"

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:exact_1nn"
    ```

=== "`par.allpairs` and structured `par.fold`"

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:nearest_fold"
    ```

All three spellings compile to the same tiled pairwise schedule on CPU, Metal,
and CUDA. [Compare timings and scratch memory](release-benchmarks.md#scaling-exact-1-nn-with-reference-count).

### Inverse-distance smoothing

Swap the fold state from `(min_dist, idx)` to `(weight_sum, weighted_sum)` to
smooth targets with weight `w[i, j] = 1 / (1 + ‖q[i] − r[j]‖²)` in a single
tiled pass:

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

Nadaraya-Watson smoothing weights references by `exp(-gamma * ‖q[i] − x[j]‖²)`.
[Law 1 rescaled-monoid tiling](algebra.md#law-1-rescaled-semiring-monoids)
streams tiles without allocating an `N × M` weight matrix:

```python
--8<-- "examples/algebraic_gallery.py:rbf_kernel_regression"
```

---

## 2. Histograms and scatter-add fusion

### Histograms and tree splits

Gradient-boosted trees accumulate gradients and Hessians into feature bins and
pick the split with maximum gain. All four spellings below lower to `O(N)`
`IndexScatter` and fuse both histograms into one
[thread-privatized pass](algebra.md#law-6-privatized-scatter-ordered-multi-state-carries):

=== "Declarative comprehension"

    Inverts `[np.sum(grads[bin_ids == b]) for b in range(nbins)]` automatically
    from `O(nbins · N)` to one `O(N)` fused histogram pass:

    ```python
    --8<-- "examples/algebraic_gallery.py:gbdt_best_split_comprehension"
    ```

=== "NumPy `np.add.at`"

    ```python
    --8<-- "examples/algebraic_gallery.py:gbdt_best_split_numpy"
    ```

=== "Python `for` loop"

    Lifts independent indexed `+=` updates into one fused parallel histogram:

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

Tropical matrix multiplication replaces `(*, +)` with `(+, min)`:
`C[i, j] = min_r (A[i, r] + B[r, j])`. Writing the index space explicitly with
`par.imap` and `par.ireduce(..., combine=np.minimum)` avoids an `m × n × k`
broadcast tensor and triggers cache and register tiling:

```python
--8<-- "examples/algebraic_gallery.py:tropical_min_plus"
```

---

## 4. Sequential recurrences to parallel scans

### Exponential IIR filter

A plain `for t in range(T):` affine recurrence `s = a[t] * s + b[t]` rewrites
via [Law 4 chunkwise duality](algebra.md#law-4-chunkwise-state-space-duality)
into a parallel prefix scan across CPU cores or GPU threadblocks:

```python
--8<-- "examples/algebraic_gallery.py:iir_exponential_filter"
```

### Mamba-2 state-space recurrence

The same pass turns a 6-line bilinear matrix-state loop into division-free
chunkwise GEMMs that lower directly to CPU register tiles, Metal
`simdgroup_matrix`, and CUDA TF32 WMMA:

```python
--8<-- "examples/algebraic_gallery.py:mamba2_ssd"
```

---

## 5. Attention and large-vocabulary losses

### Causal attention

Four lines of NumPy automatically compose
[online monoidal tiling](algebra.md#law-1-rescaled-semiring-monoids) (no `T × T`
matrices), [tile-interval block skipping](algebra.md#law-5-affine-tile-interval-mask-pruning)
(upper triangle never loaded), and
[rematerialized AD](algebra.md#law-2-algebraic-rematerialization-in-reverse-mode-ad)
under `par.grad`, `par.jvp`, and `par.hvp`:

```python
--8<-- "examples/algebraic_gallery.py:causal_flash_attention"
```

Intersect the mask with `(idx[:, None] - idx[None, :]) <= window` for an
automatic `O(N · W)` sliding window.

### Linear cross-entropy

Tiles `h @ w` and `logsumexp` over the vocabulary `V` without allocating an
`M × V` logit matrix (forward and backward):

```python
--8<-- "examples/algebraic_gallery.py:cut_cross_entropy"
```

### Multi-head latent attention

[Law 3 contraction reassociation](algebra.md#law-3-bilinear-contraction-reassociation-shared-kv-tiling)
absorbs `w_uk` into `q` and defers `w_uv` during short-query decode so tiled
attention runs directly over the compressed latent `c_kv`:

```python
--8<-- "examples/algebraic_gallery.py:deepseek_mla"
```

### Grouped-query attention

Shared-KV head-group tiling folds the head-group axis `G` into the query tile
so all `G` query heads reuse one shared-memory load of `k` and `v`:

```python
--8<-- "examples/algebraic_gallery.py:gqa_attention"
```

---

## 6. Dataclasses & more runnable programs

### Dataclasses, methods, and polymorphic dispatch

`@dataclass` modules, pure methods, `@property`, `__call__`, and structural
`match`/`case` inline completely during elaboration with zero `PyObject`
overhead:

```python
--8<-- "examples/protein_and_layers.py:protein_and_layers"
```

### Full example programs

| Example | What it demonstrates | Source / Walkthrough |
|---|---|---|
| **nanoGPT from scratch** | 2-layer char-level transformer with fused forward, `par.value_and_grad`, and Adam | [Walkthrough](gallery/nanogpt.md) · [`examples/nanogpt.py`](https://github.com/iskandr/parakeet/blob/master/examples/nanogpt.py) |
| **Bioinformatics & structural biology** | FASTQ k-mers, 6-channel BAM pileup, 2048-bit Tanimoto, Kabsch RMSD, Gotoh, Myers, SSHash RNA | [Bioinformatics gallery](bioinformatics.md) · [`examples/bio_gallery.py`](https://github.com/iskandr/parakeet/blob/master/examples/bio_gallery.py) |
| **Complex FFT & FASTA/CSV bytes** | Unboxed `complex64` Cooley-Tukey FFT, Wirtinger AD, zero-`PyObject` byte parsing | [`examples/complex_fft_and_fasta.py`](https://github.com/iskandr/parakeet/blob/master/examples/complex_fft_and_fasta.py) |
| **Adaptive CudaTree GBDT** | Level-wise BFS histograms + histogram subtraction + thread-per-subtree DFS via `par.fixed_list` | [`examples/cudatree_gbdt.py`](https://github.com/iskandr/parakeet/blob/master/examples/cudatree_gbdt.py) |
| **Inspecting rewrites** | Call `par.show_rewrite(fn, *args).summary()` on any kernel above | [Walkthrough](primitives.md#inspecting-your-own-functions) |
