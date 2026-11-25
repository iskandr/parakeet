# Algebraic laws

Parakeet started in 2012 with a small algebra of data-parallel operators —
maps, reductions, scans, and filters — and a set of fusion and tiling rules
over them. Those rules eliminate temporary arrays and improve cache locality
for ordinary array expressions, comprehensions, and matrix products.

Many modern numerical and machine-learning kernels need one step further:
eliminating quadratic `O(N²)` or `O(M · V)` intermediates, turning long serial
recurrences into parallel block passes, and avoiding saving giant intermediates
for reverse-mode differentiation. Rather than adding special-purpose "layer"
ops to the compiler, Parakeet encodes these transformations as six general
algebraic laws over the same five operator statements (`IndexMap`,
`IndexReduce`, `IndexScan`, `IndexFilter`, `IndexScatter`), multi-state `Carry`
folds, and `IndexTile` / `TileStage` tiling.

Because the passes match operator algebra instead of model names, the same
rules fire on Gaussian kernel regression, shortest-path products, exact
nearest-neighbor search, histogram split-finding, and IIR filters as on
attention, cross-entropy, and state-space models.

## The six laws at a glance

| Law | Algebraic idea | Middle-end pass | Example workloads |
|---|---|---|---|
| **Law 1** — Rescaled semiring monoids | Associative rescaling fold `(m₁, l₁, o₁) ⊕ (m₂, l₂, o₂)` | `monoidal tiling` | Softmax attention, Split-K decoding, linear cross-entropy, Gaussian (RBF) kernel regression, 1-pass `logsumexp` |
| **Law 2** — Algebraic AD rematerialization | Sufficient-statistic VJPs that recompute score tiles on the fly | `algebraic_vjp` (in `autodiff`) | Attention backward (no `N × N` saved), cross-entropy backward (no `M × V` saved), 1-pass LayerNorm / RMSNorm VJP |
| **Law 3** — Bilinear contraction reassociation & shared-KV tiling | `(U @ Vᵀ) @ Wᵀ = U @ (Vᵀ @ Wᵀ)` + folding shared-KV query groups into tiles | `contraction reassociation` | Multi-head latent attention absorption, grouped-query attention, low-rank `(X @ A) @ B` chains |
| **Law 4** — Chunkwise state-space duality | Affine composition `(a₂, b₂) ∘ (a₁, b₁) = (a₂ a₁, a₂ b₁ + b₂)` and block duality | `chunkwise scan` | Parallel IIR / EMA / Kalman filters, Mamba-2 and gated linear attention recurrences |
| **Law 5** — Affine tile-interval pruning | Interval bounds `[i₀, i₀+Bₘ-1] × [j₀, j₀+Bₙ-1]` over index predicates | `TileInterval` in `analysis` | Causal and sliding-window masks, banded matrices, spatial cutoffs, runtime boolean tile skipping |
| **Law 6** — Privatized scatter & multi-state carries | Commutative/associative scatter-add and ordered multi-field folds | `IndexScatter`, `Fold.carries`, `pairwise fold tiling` | GBDT histogram splits, exact 1-NN search, `argmin`/`argmax` with values, sparse updates |

---

## Law 1 — Rescaled semiring monoids

Any reduction whose output depends on a global normalizing statistic (a row
maximum `m`, a sum of exponentials `l`, or a norm) can fuse with a downstream
weighted sum in a single tiled pass whenever its running state forms an
associative monoid `(S, ⊕, e)`.

For softmax-weighted contractions (attention and Gaussian kernel regression),
the state carries the running maximum `m`, normalizing sum `l`, and
unnormalized output vector `o`:

```text
m = max(m₁, m₂)
(m₁, l₁, o₁) ⊕ (m₂, l₂, o₂) = (
    m,
    l₁ · exp(m₁ - m) + l₂ · exp(m₂ - m),
    o₁ · exp(m₁ - m) + o₂ · exp(m₂ - m),
)
```

Because `⊕` is associative, the same monoid covers four workloads without
allocating an `N × M` or `M × V` matrix:

1. **Tiled softmax attention (`flash_monoid`):** streams tiles of `K` and `V`
   through CPU cache or GPU shared memory against a tile of `Q`, never
   materializing the score matrix `S` or probability matrix `P`.
2. **Split-K decoding (`split_k=True`):** when the query grid is too small to
   fill a GPU (for instance, autoregressive decode with small `T_q` and long
   `T_kv`), Parakeet splits `T_kv` across threadblocks computing partial
   `(mₚ, lₚ, oₚ)` summaries and merges them with the same `⊕`.
3. **Tiled linear cross-entropy (`monoid_lse`):** tiles `H @ W` over the
   vocabulary dimension `V` using the scalar `(m, l)` monoid while gathering
   the target score on the fly, avoiding an `M × V` logit matrix.
4. **Gaussian / RBF kernel regression (`score_op="sqdist"`):** recognizes
   `-γ · ||xᵢ - yⱼ||²` feeding a normalized exponential sum and compiles
   Nadaraya-Watson smoothing into one `O(N + M)`-memory kernel.

---

## Law 2 — Algebraic rematerialization in reverse-mode AD

A naive reverse-mode tape saves every forward intermediate, which would bring
back `O(N²)` memory for attention and `O(M · V)` memory for linear
cross-entropy. Parakeet's tape optimizer (`parakeet/transforms/algebraic_vjp.py`)
runs at the operator level before lowering and substitutes sufficient-statistic
adjoints where an algebraic identity exists:

* **Attention backward (`flash_monoid_bwd`):** saves only `(Q, K, V, O, lse)` —
  no `N × N` tensor is kept. A cheap `O(N · D)` pass computes
  `Dᵢ = sum_d dO[i, d] · O[i, d]`, and the tiled backward kernel recomputes
  each score tile `P = exp(S - lse)` and `dS = P · (dO @ Vᵀ - D)` in on-chip
  memory while accumulating `dK` and `dV`.
* **Linear cross-entropy backward (`cce_bwd`):** saves only `(H, W, targets, lse)`
  and recomputes tile probabilities `p[i, v] = exp(hᵢ · wᵥ - lseᵢ)` while
  accumulating `dH` and `dW`.
* **LayerNorm and RMSNorm VJP:** collapses multi-pass mean/variance backward
  graphs into a single pass over the saved normalized output and inverse scale.
* **Standalone softmax VJP:** drops the `max` adjoint completely using
  `dS = P · (dP - sum(P · dP))`.

---

## Law 3 — Bilinear contraction reassociation & shared-KV tiling

For a chain of two bilinear contractions sharing a single-use intermediate `T`:

```text
T[u, k] = sum_r U[u, r] · V[k, r]
Y[p, u] = sum_k W[p, k] · T[u, k]
```

reassociating the summation order computes the same result:

```text
M[p, r] = sum_k W[p, k] · V[k, r]
Y[p, u] = sum_r M[p, r] · U[u, r]
```

`parakeet/transforms/reassociate.py` compares the multiply-add counts
`U·K·(R + P)` against `P·R·(K + U)` and chooses the cheaper order:

* **Latent attention absorption:** given `K = C_kv @ W_uk`, `V = C_kv @ W_uv`,
  and `O = softmax(Q @ Kᵀ) @ V`, Law 3 emits a runtime shape-guarded branch
  that absorbs `W_uk` into `Q` (`Q_abs = Q @ W_ukᵀ`) and moves `W_uv` after
  attention (`O = (P @ C_kv) @ W_uv`). Because monoidal tiling (Law 1) runs
  right after Law 3 on both branches, short-query decode runs tiled attention
  directly over the compressed latent `C_kv` without materializing expanded
  `K` or `V`, while long-query prefill automatically takes the lower-FLOP
  expanded branch.
* **Low-rank matrix chains:** `(X @ A) @ B` and `X @ (A @ B)` pick the
  lower-FLOP association at runtime, and remain differentiable under `grad`.
* **Shared-KV head-group tiling:** when a batch axis `g` of `Q` (size `G`) does
  not vary in `K` or `V` (grouped-query attention), Parakeet folds `g` into the
  query row index (`i_eff = g * T_q + i`) so all `G` query heads share one
  shared-memory load of `K` and `V`.

---

## Law 4 — Chunkwise state-space duality

A sequential loop of length `T` has `O(T)` serial depth and cannot use multiple
cores or GPU threadblocks directly. `parakeet/transforms/tile_scan.py`
recognizes two associative recurrence patterns inside plain Python
`for t in range(T):` loops:

1. **Affine recurrences (`s[t] = a[t] * s[t - 1] + b[t]`):** covers exponential
   moving averages, IIR filters, and scalar Kalman filters. The loop is
   rewritten into a 3-stage chunkwise scan (chunk size `C = 64`) using affine
   composition `(a₂, b₂) ∘ (a₁, b₁) = (a₂ a₁, a₂ b₁ + b₂)`, reducing serial
   depth by 64× with no `O(T)` temporary arrays.
2. **Bilinear matrix-state recurrences:**
   ```python
   for t in range(T):
     S = a[t] * S + k[t][:, None] * v[t][None, :]
     out[t] = q[t] @ S
   ```
   Rewritten into a 4-step division-free chunkwise scan (`C = 64`):
   - **Step 0:** parallel cumulative decay products inside each chunk (exact
     even when `a[t] == 0` or `a[t] < 0`, and when `T` is not a multiple of `C`).
   - **Step 1:** intra-chunk quadratic attention-form products
     `Y_intra[c] = ((Q[c] @ K[c]ᵀ) * L[c]) @ V[c]`.
   - **Step 2:** chunk state summary `B[c] = K_scaled[c]ᵀ @ V[c]`.
   - **Step 3:** 64× shorter inter-chunk recurrence
     `S_in[c + 1] = A[c] * S_in[c] + B[c]`.
   - **Step 4:** off-diagonal readout `Y_off[c] = Q[c] @ S_in[c]` combined with
     `Y_intra[c]`.

   Because `chunkwise scan` runs before `tiling`, the matrix products in Steps 1,
   2, and 4 automatically flow into CPU register tiling, Metal
   `simdgroup_matrix`, and CUDA `nvcuda::wmma` Tensor Cores.

---

## Law 5 — Affine tile-interval mask pruning

In `parakeet/transforms/analysis.py`, `TileInterval` propagates coordinate
bounds `[i₀, i₀ + B_M - 1]` and `[j₀, j₀ + B_N - 1]` through affine index
expressions and boolean connectives:

* **Causal masks (`j <= i`):** proves tiles with `j₀ > i₀ + B_M - 1` are
  entirely false (skipped before loading `K` or `V`) and tiles with
  `j₀ + B_N - 1 <= i₀` are entirely true (run through a branchless inner loop).
* **Sliding windows (`(j <= i) & (i - j <= W)`):** skips both future tiles and
  tiles outside the window, reducing `O(N²)` work to `O(N · W)` from ordinary
  mask code.
* **Runtime boolean masks:** when a mask comes from a runtime array, GPU
  kernels cooperatively check whether any entry in the tile is true and skip
  empty tiles before loading values.

---

## Law 6 — Privatized scatter & multi-state carries

Two patterns show up constantly outside dense linear algebra: accumulating into
shared bins (`np.bincount`, `parakeet.scatter_add`) and folding structured
state (`argmin`, Welford variance, custom `dsl.fold`).

* **Keyed histogram fusion:** sibling `IndexScatter` statements sharing the
  same index expression (such as gradient and Hessian histograms in a decision
  tree split search) merge into one traversal over the data, lowered with
  thread-privatized bins on CPU and parallel atomics/privatization on GPU.
* **Ordered multi-state carries and pairwise tile folds:** `Fold.carries`
  tracks multiple scalar accumulators in one reduction pass while preserving
  chunk order (so leftmost ties in `argmin` or non-commutative compositions in
  `dsl.fold` stay exact). `pairwise fold tiling` fuses a pairwise score
  producer directly into an ordered fold over bounded tiles on both CPU and
  GPU — eliminating the full `N × M` distance matrix in both broadcasting NumPy
  and nested-map 1-NN.

---

## Inspecting rewrites with `parakeet.dsl`

Call [`parakeet.dsl.show_rewrite`](api.md) on any function to see which laws
fired, which tiled plans were selected, and the operator IR before and after
the middle-end passes:

```python
import numpy as np
from parakeet import dsl


@dsl.kernel
def causal_attention(q, k, v):
  causal = dsl.imap(lambda i, j: i >= j, (q.shape[0], k.shape[0]))
  s = dsl.where(causal, (q @ k.T) * dsl.f32(0.125), dsl.f32(-1e9))
  e = dsl.exp(s - dsl.max(s, axis=-1, keepdims=True))
  return (e / dsl.sum(e, axis=-1, keepdims=True)) @ v


q = np.zeros((128, 64), np.float32)
print(dsl.show_rewrite(causal_attention, q, q, q).summary())
```

```text
Parakeet Algebraic Rewrite Report for `causal_attention`
  Laws triggered : Law 1: Rescaled Semiring Monoid Tile Fusion, Law 5: Affine Tile-Interval Mask Pruning
  Monoidal plans : flash_monoid
  Operators      : IndexMap 5 -> 1 | IndexTile 1 | IndexScatter 0 -> 0 | Carry folds 0
```

See the [Kernel gallery](gallery.md) for runnable examples and timings across
all six laws.
