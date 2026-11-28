# Kernel gallery

Every featured kernel below is shown in Parakeet's standard recurring styles so
you can compare how each idiom compiles:

| Suffix | Style |
|---|---|
| `*_numpy` | NumPy broadcasting and array expressions |
| `*_comp` | Python comprehensions |
| `*_loop` | Explicit Python `for` loops |
| `*_par` | Explicit Parakeet operators (`par.map`, `par.allpairs`, `par.fold`, `par.imap`, `par.ireduce`, `par.scatter_add`, `par.scan`) |
| `*_helpers` | Wrapped `par.nn` helpers (where applicable) |

Every snippet below is included verbatim from
[`examples/algebraic_gallery.py`](https://github.com/iskandr/parakeet/blob/master/examples/algebraic_gallery.py)
and verified for exact numerical agreement by `python -m examples.algebraic_gallery`.
See also [Train nanoGPT](gallery/nanogpt.md) and
[Bioinformatics & structural biology](bioinformatics.md).

```python
import operator
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

=== "NumPy (`_numpy`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:nearest_numpy"
    ```

=== "Comprehension (`_comp`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:nearest_comp"
    ```

=== "Loop (`_loop`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:nearest_loop"
    ```

=== "Operators (`_par`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:nearest_par"
    ```

`nearest_numpy`, `nearest_comp`, and `nearest_par` all lower to the same tiled
pairwise schedule on CPU, Metal, and CUDA.
[Compare timings and scratch memory](release-benchmarks.md#scaling-exact-1-nn-with-reference-count).

### Inverse-distance smoothing

Smooth reference targets with weight `w[i, j] = 1 / (1 + ‖q[i] − r[j]‖²)` in a
single tiled pass without materializing the `Q × R` weight matrix:

=== "NumPy (`_numpy`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:smooth_numpy"
    ```

=== "Comprehension (`_comp`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:smooth_comp"
    ```

=== "Loop (`_loop`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:smooth_loop"
    ```

=== "Operators (`_par`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:squared_distance"

    --8<-- "examples/algebraic_gallery.py:smooth_par"
    ```

### Gaussian kernel regression

Nadaraya-Watson smoothing weights references by `exp(-gamma * ‖q[i] − x[j]‖²)`.
[Law 1 rescaled-monoid tiling](algebra.md#law-1-rescaled-semiring-monoids)
streams tiles without allocating an `N × M` weight matrix:

=== "NumPy (`_numpy`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:rbf_regression_numpy"
    ```

=== "Helpers (`_helpers`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:rbf_regression_helpers"
    ```

=== "Comprehension (`_comp`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:rbf_regression_comp"
    ```

=== "Loop (`_loop`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:rbf_regression_loop"
    ```

=== "Operators (`_par`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:rbf_regression_par"
    ```

---

## 2. Histograms and scatter-add fusion

### Histograms and tree splits

Gradient-boosted trees accumulate gradients and Hessians into feature bins and
pick the split with maximum gain. All four styles below lower to `O(N)`
`IndexScatter` and fuse both histograms into one
[thread-privatized pass](algebra.md#law-6-privatized-scatter-ordered-multi-state-carries):

=== "NumPy (`_numpy`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:gbdt_split_numpy"
    ```

=== "Comprehension (`_comp`)"

    Inverts `[np.sum(grads[bin_ids == b]) for b in range(nbins)]` automatically
    from `O(nbins · N)` to one `O(N)` fused histogram pass:

    ```python
    --8<-- "examples/algebraic_gallery.py:gbdt_split_comp"
    ```

=== "Loop (`_loop`)"

    Lifts independent indexed `+=` updates into one fused parallel histogram:

    ```python
    --8<-- "examples/algebraic_gallery.py:gbdt_split_loop"
    ```

=== "Operators (`_par`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:gbdt_split_par"
    ```

---

## 3. Custom semirings with index operators

### Min-plus matrix product

Tropical matrix multiplication replaces `(*, +)` with `(+, min)`:
`C[i, j] = min_r (A[i, r] + B[r, j])`. Writing the index space explicitly with
`par.imap` and `par.ireduce(..., combine=np.minimum)` avoids an `m × n × k`
broadcast tensor and triggers cache and register tiling:

=== "NumPy (`_numpy`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:tropical_min_plus_numpy"
    ```

=== "Comprehension (`_comp`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:tropical_min_plus_comp"
    ```

=== "Loop (`_loop`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:tropical_min_plus_loop"
    ```

=== "Operators (`_par`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:tropical_min_plus_par"
    ```

---

## 4. Sequential recurrences to parallel scans

### Exponential IIR filter

A plain `for t in range(T):` affine recurrence `s = a[t] * s + b[t]` works for
arbitrary `a` (including zero or negative coefficients) and rewrites via
[Law 4 chunkwise duality](algebra.md#law-4-chunkwise-state-space-duality) into
a parallel prefix scan. For positive decays (`a > 0`), the closed-form prefix
product can also be written directly with NumPy or `par.scan`:

=== "Loop (`_loop` — arbitrary `a`, Law 4 `TileScan`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:iir_filter_loop"
    ```

=== "NumPy (`_numpy` — positive `a > 0`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:iir_filter_numpy"
    ```

=== "Comprehension (`_comp` — positive `a > 0`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:iir_filter_comp"
    ```

=== "Operators (`_par` — positive `a > 0`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:iir_filter_par"
    ```

### Mamba-2 state-space recurrence

The same Law 4 pass turns a 6-line bilinear matrix-state loop into
division-free chunkwise GEMMs that lower directly to CPU register tiles, Metal
`simdgroup_matrix`, and CUDA TF32 WMMA:

```python
--8<-- "examples/algebraic_gallery.py:mamba2_ssd"
```

---

## 5. Attention and large-vocabulary losses

### Causal attention

Four lines of NumPy (or a single call to `par.nn.scaled_dot_product_attention`)
compose [online monoidal tiling](algebra.md#law-1-rescaled-semiring-monoids)
(no `T × T` matrices), [tile-interval block skipping](algebra.md#law-5-affine-tile-interval-mask-pruning)
(upper triangle never loaded), and
[rematerialized AD](algebra.md#law-2-algebraic-rematerialization-in-reverse-mode-ad)
under `par.grad`, `par.jvp`, and `par.hvp`:

=== "NumPy (`_numpy`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:causal_attention_numpy"
    ```

=== "Helpers (`_helpers`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:causal_attention_helpers"
    ```

=== "Comprehension (`_comp`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:causal_attention_comp"
    ```

=== "Loop (`_loop`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:causal_attention_loop"
    ```

=== "Operators (`_par`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:causal_attention_par"
    ```

Intersect the mask with `(idx[:, None] - idx[None, :]) <= window` for an
automatic `O(N · W)` sliding window.

### Linear cross-entropy

Tiles `h @ w` and `logsumexp` over the vocabulary `V` without allocating an
`M × V` logit matrix (forward and backward):

=== "NumPy (`_numpy`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:cross_entropy_numpy"
    ```

=== "Helpers (`_helpers`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:cross_entropy_helpers"
    ```

=== "Comprehension (`_comp`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:cross_entropy_comp"
    ```

=== "Loop (`_loop`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:cross_entropy_loop"
    ```

=== "Operators (`_par`)"

    ```python
    --8<-- "examples/algebraic_gallery.py:cross_entropy_par"
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

## 6. Objects, methods, and full programs

### Dataclasses, methods, and polymorphic dispatch

`@dataclass` types and `par.nn.Module` subclasses inline pure methods,
`@property`, `__call__`, and structural `match`/`case` completely during
elaboration with zero `PyObject` overhead:

```python
--8<-- "examples/protein_and_layers.py:protein_and_layers"
```

### Full example programs

| Example | What it demonstrates | Source / Walkthrough |
|---|---|---|
| **nanoGPT (`par.nn` & functional)** | 2-layer char-level transformer both with `par.nn.Module`/`par.optim.Adam` and in minimalist from-scratch functional style | [Walkthrough](gallery/nanogpt.md) · [`examples/nanogpt.py`](https://github.com/iskandr/parakeet/blob/master/examples/nanogpt.py) · [`examples/nanogpt_functional.py`](https://github.com/iskandr/parakeet/blob/master/examples/nanogpt_functional.py) |
| **Bioinformatics & structural biology** | FASTQ k-mers via `par.Alphabet`, 6-channel BAM pileup, 2048-bit Tanimoto, Kabsch RMSD, Gotoh, Myers, SSHash RNA | [Bioinformatics gallery](bioinformatics.md) · [`examples/bio_gallery.py`](https://github.com/iskandr/parakeet/blob/master/examples/bio_gallery.py) |
| **Complex radix-2 FFT** | Unboxed `complex64` Cooley-Tukey FFT and Wirtinger spectral-filter AD | [`examples/fft.py`](https://github.com/iskandr/parakeet/blob/master/examples/fft.py) |
| **FASTA & CSV byte parsing** | Zero-`PyObject` byte parsing with `par.split_bytes`, `par.Alphabet.mask`, and `par.parse_float32` | [`examples/fasta_io.py`](https://github.com/iskandr/parakeet/blob/master/examples/fasta_io.py) |
| **Adaptive CudaTree GBDT** | Level-wise BFS histograms + histogram subtraction + thread-per-subtree DFS via `par.fixed_list` | [`examples/cudatree_gbdt.py`](https://github.com/iskandr/parakeet/blob/master/examples/cudatree_gbdt.py) |
| **Inspecting rewrites** | Call `par.show_rewrite(fn, *args).summary()` on any kernel above | [Walkthrough](primitives.md#inspecting-your-own-functions) |
