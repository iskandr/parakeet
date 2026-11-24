# Algebraic laws & frontier kernels

Modern high-performance kernels — FlashAttention-2/3, FlashDecoding, FlexAttention,
Apple Cut Cross-Entropy (CCE), DeepSeek Multi-Head Latent Attention (FlashMLA),
Llama-3 Grouped-Query Attention (GQA), and Mamba-2 State-Space Duality (SSD) — rarely
get their order-of-magnitude speedups from writing tighter scalar loops. They get
them by eliminating $O(N^2)$, $O(N \cdot V)$, or $O(T)$ memory round-trips and
sequential bottlenecks using a small set of **universal algebraic laws**.

Parakeet encodes these laws directly as general IR passes over its five
data-parallel operators (`IndexMap`, `IndexReduce`, `IndexScan`, `IndexScatter`,
`IndexTile`) plus multi-state `Carry` folds and `TileStage` SRAM staging. None of
these passes hard-code "Transformer" or "LLM" layers: the exact same rewrite rules
fire on RBF/Gaussian kernel regression, tropical shortest-path algorithms, exact
nearest-neighbor search, GBDT histogram split-finding, and Kalman/IIR filters.

## Overview of the six algebraic laws

| Law | Algebraic Identity | IR Pass / Primitive | Frontier Kernels Synthesized Automatically | Non-Transformer Workloads Accelerated |
|---|---|---|---|---|
| **Law 1** — Rescaled Semiring Monoids | Associative monoid $(m_1, l_1, \mathbf{o}_1) \oplus (m_2, l_2, \mathbf{o}_2)$ with rescaling | `monoidal tiling` (`IndexTile` + `Carry` + `MonoidalTilePlan`) | **FlashAttention**, **FlashDecoding** (Split-K), **Cut Cross-Entropy**, 1-pass `logsumexp`/`softmax` | **RBF / Gaussian kernel regression** (Nadaraya-Watson), log-likelihood & partition functions |
| **Law 2** — Algebraic AD Rematerialization | Sufficient-statistic VJP identities ($D_i = \sum_d dO_{id} O_{id}$, SRAM logit recompute, 1-pass Norm VJP) | `algebraic_vjp.py` (tape optimizer in `autodiff.py`) | **FlashAttention backward** (0 $N\times N$ saved), **CCE backward** (0 $M\times V$ saved), **1-pass LayerNorm / RMSNorm VJP** | Differentiable kernel smoothing, large-vocabulary classifiers |
| **Law 3** — Bilinear Contraction Reassociation & Shared-KV Tiling | $(U V^\top) W^\top = U (V^\top W^\top)$ + batch-axis folding into query tile | `contraction reassociation` (`reassociate.py`) + `_maybe_fold_gqa_group` | **DeepSeek MLA / FlashMLA** weight absorption, **Llama-3 GQA** head-group tiling, **LoRA** `(X @ A) @ B` | Low-rank / Kronecker tensor chains, multi-query bilinear scoring |
| **Law 4** — Chunkwise State-Space Duality (`TileScan`) | Affine composition monoid $(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1, a_2 b_1 + b_2)$ + semiseparable block duality | `chunkwise scan` (`tile_scan.py`) | **Mamba-2 SSD**, **Gated Linear Attention (GLA)**, **RG-LRU / Griffin** | **Parallel IIR / EMA / Kalman filters**, **GBDT histogram prefix sums** |
| **Law 5** — Affine Tile-Interval Mask Pruning | Interval arithmetic on tile coordinates $[i_0, i_0+B_M-1] \times [j_0, j_0+B_N-1]$ | `TileInterval` + `tile_predicate_guards` (`analysis.py`) | **FlexAttention** causal, sliding-window, and block-sparse attention (zero wasted tiles + branchless interior) | Spatial cutoff kernels, banded matrices, tabular range masks |
| **Law 6** — Privatized Associative Scatter & Multi-State `Carry` | Commutative/associative scatter-add + multi-accumulator `Fold.carries` | `IndexScatter`, `Fold.carries` (`fusion.py`, `lowering.py`) | Sparse embedding & MoE token dispatch, `argmin`/`argmax` with values | **GBDT histogram building** (XGBoost/LightGBM core), **exact 1-NN search**, particle-in-cell |

---

## Law 1 — Rescaled semiring monoids

Any reduction whose output depends on a global normalizing statistic (a maximum $m$,
sum of exponentials $l$, mean $\mu$, or norm) can be fused with downstream
contractions in a **single SRAM-tiled pass** whenever its state forms an associative
monoid $(S, \oplus, e)$.

For softmax-weighted contractions (FlashAttention, FlashDecoding, RBF kernel
regression), the monoid carries running row maximum $m$, denominator $l$, and
unnormalized output vector $\mathbf{o} \in \mathbb{R}^D$:

$$
(m_1, l_1, \mathbf{o}_1) \oplus (m_2, l_2, \mathbf{o}_2)
= \Big(m,\; l_1 e^{m_1 - m} + l_2 e^{m_2 - m},\; \mathbf{o}_1 e^{m_1 - m} + \mathbf{o}_2 e^{m_2 - m}\Big),
\quad m = \max(m_1, m_2)
$$

Because $\oplus$ is strictly associative, the **exact same monoid** simultaneously
produces:

1. **FlashAttention** (`kind="flash_monoid"`): streams tiles of $K_j, V_j$ through
   GPU `__shared__` / `threadgroup` memory against hoisted $Q_i$ without ever
   allocating the $O(N^2)$ score matrix $S$ or probability matrix $P$.
2. **FlashDecoding** (`split_k=True`): when the query grid is small relative to device
   SMs (e.g. autoregressive decode $T_q \le 16$, $T_{kv} \ge 128$), Parakeet splits
   $T_{kv}$ across parallel threadblocks computing partial $(m_p, l_p, \mathbf{o}_p)$
   and recombines them using the exact same $\oplus$.
3. **Cut Cross-Entropy** (`kind="monoid_lse"`): tiles $H W^\top$ over vocabulary $V$
   with the scalar $(m, l)$ monoid plus on-the-fly target dot-product gather, never
   materializing the $O(M \cdot V)$ logit matrix.
4. **RBF / Gaussian Kernel Regression** (`score_op="sqdist"`): recognizes
   $-\gamma \|x_i - y_j\|_2^2$ as a tiled contraction feeding softmax and compiles
   Nadaraya-Watson smoothing into a single $O(N + M)$-memory kernel.

---

## Law 2 — Algebraic rematerialization in reverse-mode AD

Standard reverse-mode automatic differentiation saves every forward intermediate on
the tape, which immediately re-introduces $O(N^2)$ memory for attention and
$O(M \cdot V)$ memory for cross-entropy. Parakeet's tape optimizer
(`parakeet/transforms/algebraic_vjp.py`) runs at the high-level operator level before
lowering and rewrites known structures into **sufficient-statistic algebraic VJPs**:

* **Attention backward (`flash_monoid_bwd`):** saves only $(Q, K, V, O, \text{lse})$ —
  zero $N \times N$ tensors saved. A cheap $O(ND)$ pre-pass computes
  $D_i = \sum_d dO_{id} O_{id}$, and the SRAM-tiled backward kernel recomputes each
  tile $P_{ij} = \exp(S_{ij} - \text{lse}_i)$ and $dS_{ij} = P_{ij}(dO_i V_j^\top - D_i)$
  on-chip while accumulating $dK, dV$ in registers.
* **Cut Cross-Entropy backward (`cce_bwd`):** saves only $(H, W, y, \text{lse})$ —
  zero $M \times V$ logits saved. A 2D-tiled GPU kernel recomputes
  $p_{iv} = \exp(h_i \cdot w_v - \text{lse}_i)$ in SRAM and accumulates $dH, dW$.
* **LayerNorm & RMSNorm VJP:** collapses multi-pass mean/variance/sqrt backward
  graphs into a single algebraic VJP over saved $(\hat{x}, \sigma)$, eliminating
  redundant reductions.
* **Standalone Softmax VJP:** eliminates the `max` backward entirely via
  $dS = P \odot (dP - \sum P \odot dP)$.

---

## Law 3 — Bilinear contraction reassociation & shared-KV tiling

For any chain of two bilinear contractions sharing a single-use intermediate $T$:

$$
T[u, k] = \sum_{r=0}^{R-1} U[u, r]\, V[k, r], \qquad
Y[p, u] = \sum_{k=0}^{K-1} W[p, k]\, T[u, k]
$$

reassociating the contraction order yields the identical tensor:

$$
M[p, r] = \sum_{k=0}^{K-1} W[p, k]\, V[k, r], \qquad
Y[p, u] = \sum_{r=0}^{R-1} M[p, r]\, U[u, r]
$$

`parakeet/transforms/reassociate.py` compares the exact multiply-add counts
$U K (R + P)$ vs $P R (K + U)$:

* **DeepSeek MLA weight absorption:** Given naive
  `K = C_kv @ W_uk; V = C_kv @ W_uv; O = softmax(Q @ K.T) @ V`, Law 3 emits a
  cost-guarded branch that absorbs $W_{uk}$ into $Q$ (`Q_abs = Q @ W_uk.T`) and moves
  $W_{uv}$ after attention (`O_lat = P @ C_kv; O = O_lat @ W_uv`). Because
  `monoidal tiling` (Law 1) runs immediately after Law 3 and recurses into both
  branches, the decode branch automatically fuses into **FlashMLA / FlashDecoding
  directly over the compressed latent `C_kv`** — never decompressing `K` or `V` into
  HBM. When $T_q$ is large (prefill), the runtime guard picks the lower-FLOP
  decompressed branch automatically.
* **General low-rank chains (LoRA, Kronecker):** `(X @ A) @ B` and `X @ (A @ B)`
  automatically select the FLOP-optimal order at runtime with full reverse-mode AD
  support.
* **GQA head-group tiling:** When a trailing batch axis `g` of `Q` (extent `G`) does
  not appear in `K` or `V` (e.g. `Q: (B, H_kv, G, T_q, D)`, `K, V: (B, H_kv, 1, T_kv, D)`),
  Parakeet folds `g` into the query row tile (`i_eff = g * T_q + i`), serving all `G`
  query heads from a single SRAM load of `K` and `V`.

---

## Law 4 — Chunkwise state-space duality (`TileScan`)

Sequential recurrences over length $T$ have $O(T)$ serial depth and cannot saturate
multi-core CPUs or GPUs directly. `parakeet/transforms/tile_scan.py` recognizes two
general associative recurrence structures inside plain Python `for t in range(T):`
loops:

1. **Affine linear recurrences** ($s_t = a_t s_{t-1} + b_t$ — EMA, IIR filters,
   scalar Kalman filters, RG-LRU, GBDT histogram prefix sums):
   Rewrites the loop into a 3-stage parallel chunkwise scan (`C = 64`) using the
   associative affine composition monoid
   $(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1,\; a_2 b_1 + b_2)$ — cutting sequential
   depth by $64\times$ with zero $O(T)$ temporary arrays.
2. **Bilinear matrix-state recurrences** (Mamba-2 SSD & Gated Linear Attention):
   ```python
   for t in range(T):
     S = a[t] * S + k[t][:, None] * v[t][None, :]
     out[t] = q[t] @ S
   ```
   Rewrites the 6-line sequential loop into **4-step division-free Chunkwise
   State-Space Duality** (`C = 64`):
   - **Step 0:** parallel division-free cumulative decay products (exact even when
     $a_t = 0$ or $a_t < 0$, and for $T$ not divisible by $C$).
   - **Step 1:** intra-chunk quadratic attention-form GEMMs
     $M^{(c)} = (Q^{(c)} {K^{(c)}}^\top) \odot L^{(c)}$,
     $Y_{\text{intra}}^{(c)} = M^{(c)} V^{(c)}$.
   - **Step 2:** chunk state summary GEMM $B^{(c)} = (\tilde{K}^{(c)})^\top V^{(c)}$.
   - **Step 3:** $64\times$-shorter inter-chunk recurrence
     $S_{\text{in}}^{(c+1)} = A^{(c)} S_{\text{in}}^{(c)} + B^{(c)}$.
   - **Step 4:** off-diagonal readout GEMM $Y_{\text{off}}^{(c)} = Q^{(c)} S_{\text{in}}^{(c)}$
     and parallel combine.

   Because `chunkwise scan` runs right before `tiling`, all four heavy steps compile
   automatically to **NVIDIA CUDA `nvcuda::wmma` Tensor Cores**, **Apple Metal
   `simdgroup_matrix`**, and **CPU register micro-tiles**.

---

## Law 5 — Affine tile-interval mask pruning

In `parakeet/transforms/analysis.py`, `TileInterval` propagates bounding intervals
$[i_0, i_0 + B_M - 1]$ and $[j_0, j_0 + B_N - 1]$ through arbitrary affine index
expressions and boolean connectives:

* **Causal mask** (`j <= i`): proves `j0 > i0 + BM - 1` is **all-false** (entire
  upper-triangular block skipped before loading `K` or `V`!) and `j0 + BN - 1 <= i0`
  is **all-true** (branchless inner loop!).
* **Sliding-window / FlexAttention** (`(j <= i) & (i - j <= W)`): automatically skips
  both future tiles and past tiles outside window `W` — turning $O(N^2)$ attention
  into $O(N \cdot W)$ without writing a custom sparse kernel.
* **Runtime boolean masks:** when a mask comes from a runtime boolean array, the GPU
  kernel performs a cooperative SRAM ballot across the tile and skips all-false tiles
  before any dot-product or value loads.

---

## Inspecting rewrites with `parakeet.dsl`

Use [`parakeet.dsl.show_rewrite`](api.md) on any function to see which algebraic laws
fired, which `MonoidalTilePlan`s were synthesized, and the before/after IR:

```python
import numpy as np
from parakeet import dsl


@dsl.kernel
def causal_attention(q, k, v):
  causal = dsl.imap(lambda ij: ij[0] >= ij[1], (q.shape[0], k.shape[0]))
  s = np.where(causal, (q @ k.T) * np.float32(0.125), np.float32(-1e9))
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v


q = np.zeros((128, 64), np.float32)
print(dsl.show_rewrite(causal_attention, q, q, q).summary())
```

```text
Parakeet Algebraic Rewrite Report for `causal_attention`
  Laws triggered : Law 1: Rescaled Semiring Monoid Tile Fusion, Law 5: Affine Tile-Interval Mask Pruning
  Monoidal plans : flash_monoid
  Operators      : IndexMap 5 -> 1 | IndexTile 1 | IndexScatter 0 | Carry folds 0
```

See the [Gallery of examples](gallery.md) for runnable benchmarks across all six laws.
