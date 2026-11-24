# Gallery of examples

Every example below is **5–15 lines of ordinary NumPy or `parakeet.dsl`** — yet
Parakeet's six algebraic laws automatically compile it into a fused, SRAM-tiled,
or chunkwise-dual native kernel for multi-core CPUs, Apple Metal GPUs, and NVIDIA
CUDA GPUs.

Run the entire gallery locally with exact numerical verification and timings:

```bash
python examples/algebraic_gallery.py
```

## Why these examples matter

| Workload | Lines in Parakeet | Typical Hand-Tuned Kernel | Algebraic Laws Fired | Key Win |
|---|---:|---:|---|---|
| **Causal / Sliding-Window FlashAttention (fwd + bwd)** | **7 lines** | ~1,500+ LOC CUDA / ~250 LOC Triton | Laws 1 + 2 + 5 | Zero $O(N^2)$ HBM allocation; $D_i$-recomputed backward; block-sparse tile skipping |
| **Apple Cut Cross-Entropy (fwd + bwd)** | **6 lines** | ~400 LOC Triton / CUDA | Laws 1 + 2 | Zero $O(M \cdot V)$ logit matrix; SRAM-recomputed $dH, dW$ |
| **DeepSeek MLA Weight Absorption (FlashMLA)** | **6 lines** | ~600 LOC CUDA / Triton | Laws 3 + 1 | Absorbs $W_{uk}, W_{uv}$ into $Q$/output; attends directly over latent $C_{kv}$ |
| **Llama-3 GQA Shared-KV Head-Group Tiling** | **5 lines** | ~300 LOC Triton | Laws 3 + 1 | Folds head group $G$ into query tile; $G\times$ less KV-cache HBM reads |
| **Mamba-2 State-Space Duality (SSD)** | **6 lines** | ~500 LOC Triton | Law 4 | Rewrites serial loop into 4 batched Tensor-Core GEMMs (**48× faster** than serial C!) |
| **RBF / Gaussian Kernel Regression** | **5 lines** | ~250 LOC KeOps CUDA | Law 1 (`sqdist`) | Zero $O(N \cdot M)$ pairwise matrix (**17.9× faster** than NumPy on CPU!) |
| **Tropical Min-Plus Matrix Product (`min/+`)** | **5 lines** | ~200 LOC C++/CUDA | Register/SRAM Tiling | One `(min, +)` semiring matrix product with zero $O(N^3)$ temporary tensor |
| **Exact 1-NN Search** | **4 lines** | ~150 LOC CUDA / FAISS flat | `Carry` + Vertical Fusion | Zero $O(N \cdot M)$ distance matrix (**14.9× faster** than NumPy on CPU!) |
| **GBDT Histogram + Prefix-Sum Best Split** | **9 lines** | ~400 LOC XGBoost/LightGBM C++ | `IndexScatter` + Scan + `Carry` | Privatized parallel histogram + prefix-sum split gain (**2.9× faster** than NumPy!) |
| **Batched nanoGPT Training Step (fwd+bwd+Adam)** | **~45 lines** | ~2,000+ LOC custom kernels | Laws 1 + 2 + 5 + WMMA | **1.42× faster than PyTorch eager** on NVIDIA L4 (`6.44 ms` vs `9.15 ms`)! |

---

## 1. Causal & sliding-window FlashAttention (fwd + $D_i$ backward)

**7 lines — zero $O(N^2)$ memory in forward OR backward:**

```python
import numpy as np
from parakeet import dsl, value_and_grad


@dsl.kernel
def causal_flash_attention(q, k, v):
  causal = dsl.imap(lambda ij: ij[0] >= ij[1], (q.shape[-2], k.shape[-2]))
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))
  s = np.where(causal, np.matmul(q, np.swapaxes(k, -1, -2)) * scale, np.float32(-1e9))
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return np.matmul(e / np.sum(e, axis=-1, keepdims=True), v)
```

For sliding-window FlexAttention, change only the predicate line:

```python
mask = dsl.imap(lambda ij: (ij[0] >= ij[1]) & ((ij[0] - ij[1]) <= window), (T, T))
```

* **Law 1** fuses `Q @ K.T -> softmax -> P @ V` into a single SRAM-tiled `flash_monoid`
  kernel carrying `(m, l, O)` in registers.
* **Law 5** propagates tile intervals `[i0, i0+31] × [j0, j0+31]` through `i >= j` and
  `i - j <= W`, skipping all empty tiles before loading `K`/`V` and removing branches
  from interior tiles.
* **Law 2** (`value_and_grad`) saves only `(Q, K, V, O, lse)` and emits the
  $D_i = \sum_d dO_{id} O_{id}$ recomputed `flash_monoid_bwd` kernel.

---

## 2. Apple Cut Cross-Entropy (zero $M \times V$ logits)

**6 lines — never materializes the `[BT, V]` vocabulary logit matrix:**

```python
def cut_cross_entropy(h, w, targets):
  logits = h @ w
  m = np.max(logits, axis=-1, keepdims=True)
  z = logits - m
  lse = np.log(np.sum(np.exp(z), axis=-1))
  picked = z[np.arange(h.shape[0]), targets]
  return -np.mean(picked - lse)
```

* **Law 1** compiles forward into `monoid_lse`, tiling `h @ w` over vocabulary chunks
  in SRAM while tracking `(m, l)` and the target logit.
* **Law 2** compiles `value_and_grad(cut_cross_entropy, argnum=(0, 1))` into 2D-tiled
  `cce_bwd`, recomputing `p_iv = exp(h_i · w_v - lse_i)` on-chip with zero $M \times V$
  saved activation tensor.

---

## 3. DeepSeek MLA weight absorption (FlashMLA)

**6 lines — automatically switches between latent-absorbed decode and prefill:**

```python
@dsl.kernel
def deepseek_mla(q, c_kv, w_uk, w_uv):
  k = c_kv @ w_uk
  v = c_kv @ w_uv
  scale = np.float32(1.0 / np.sqrt(np.float32(q.shape[-1])))
  s = (q @ k.T) * scale
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ v
```

* **Law 3** reassociates `Q @ (C_kv @ W_uk)^T` → `(Q @ W_uk^T) @ C_kv^T` and
  `P @ (C_kv @ W_uv)` → `(P @ C_kv) @ W_uv` under an exact FLOP cost guard.
* **Law 1** immediately fuses `(Q_abs @ C_kv^T) -> softmax -> (P @ C_kv)` into
  `flash_monoid` / FlashDecoding directly over the tiny latent `C_kv`!
* On decode (`T_q=4, T_kv=2048, d_c=64, d_h=256`), Law 3 + Law 1 runs **8.8× faster**
  than without reassociation (`0.19 ms` vs `1.69 ms` on CPU).

---

## 4. Mamba-2 State-Space Duality (SSD) & parallel IIR filters

**6 lines of sequential Python → 4 batched Tensor-Core GEMMs (48× speedup):**

```python
@dsl.kernel
def mamba2_ssd(q, k, v, a):
  T = q.shape[0]
  out = np.empty((T, v.shape[1]), dtype=np.float32)
  S = np.zeros((k.shape[1], v.shape[1]), dtype=np.float32)
  for t in range(T):
    S = a[t] * S + k[t][:, None] * v[t][None, :]
    out[t] = q[t] @ S
  return out
```

* **Law 4** (`chunkwise scan`) transforms the sequential loop into division-free
  4-step Chunkwise State-Space Duality (`C = 64`), exact even when `a[t] == 0` (hard
  state reset) or `a[t] < 0` (oscillatory SSM).
* The four heavy contractions (`Q_c @ K_c^T`, `M @ V_c`, `K_tail^T @ V_c`, `Q_c @ S_in`)
  are contiguous 3D batched GEMMs that automatically lower to CUDA `nvcuda::wmma`
  Tensor Cores, Apple Metal `simdgroup_matrix`, and CPU register tiles — **48.3× faster**
  than the sequential C loop (`1.04 ms` vs `50.23 ms`) and **7.9× faster** than NumPy!

The exact same pass also parallelizes arbitrary scalar/batched affine recurrences
(EMA, IIR filters, scalar Kalman filters, RG-LRU):

```python
@dsl.kernel
def iir_exponential_filter(a, b):
  T = a.shape[0]
  out = np.empty_like(b)
  s = np.float32(0.0)
  for t in range(T):
    s = a[t] * s + b[t]
    out[t] = s
  return out
```

---

## 5. Fused RBF / Gaussian kernel regression (Nadaraya-Watson)

**5 lines — zero $O(N \cdot M)$ distance matrix, 17.9× faster than NumPy:**

```python
@dsl.kernel
def rbf_kernel_regression(x_query, x_train, y_train, neg_gamma):
  d2 = np.sum((x_query[:, None, :] - x_train[None, :, :]) ** np.float32(2.0), axis=-1)
  s = d2 * neg_gamma
  e = np.exp(s - np.max(s, axis=-1, keepdims=True))
  return (e / np.sum(e, axis=-1, keepdims=True)) @ y_train
```

Standard PyTorch, JAX, and SciPy materialize the entire $N \times M$ pairwise squared
distance matrix and weight matrix in HBM (OOMing at $N = M = 65{,}536$). Because
**Law 1** recognizes squared-distance contractions (`score_op="sqdist"`), Parakeet
automatically compiles this 5-line function into the exact same 1-pass SRAM-tiled
`flash_monoid` kernel — **$O(N + M)$ memory, 17.9× faster than NumPy on CPU**!

---

## 6. One tropical min-plus matrix product & exact 1-NN search

**5 lines — arbitrary semiring matmul & `Carry`-fused nearest neighbor:**

```python
@dsl.kernel
def tropical_min_plus(a, b):
  m, n, k = a.shape[0], b.shape[1], a.shape[1]
  return dsl.imap(
    lambda ij: dsl.ireduce(lambda r: a[ij[0], r] + b[r, ij[1]], k, combine=parakeet.minimum),
    (m, n),
  )


@dsl.kernel
def exact_1nn(x_query, x_ref):
  return dsl.map(
    lambda q: np.argmin(dsl.map(lambda r: np.sum((q - r) ** np.float32(2.0)), x_ref)),
    x_query,
  )
```

* `tropical_min_plus` tiles the `(min, +)` semiring contraction in registers/SRAM with
  zero $O(N^3)$ intermediate tensor.
* `exact_1nn` vertically fuses the squared-distance reduction directly into an
  index-carrying `Carry` fold (`argmin`), never allocating the $N \times M$ distance
  matrix (**14.9× faster than NumPy** on CPU!).

---

## 7. GBDT histogram building & prefix-sum best-split finding

**9 lines — the inner loop of XGBoost / LightGBM:**

```python
@dsl.kernel
def gbdt_best_split(bin_ids, grads, hess, nbins, reg_lambda):
  g_hist = dsl.iscatter(nbins, bin_ids, grads)
  h_hist = dsl.iscatter(nbins, bin_ids, hess)
  g_left = np.cumsum(g_hist)
  h_left = np.cumsum(h_hist)
  g_right = g_left[nbins - 1] - g_left
  h_right = h_left[nbins - 1] - h_left
  gain = (g_left * g_left) / (h_left + reg_lambda) + (g_right * g_right) / (h_right + reg_lambda)
  best_bin = np.argmax(gain)
  return best_bin, gain[best_bin]
```

* `dsl.iscatter` lowers to thread-privatized parallel `IndexScatter` on CPU/GPU.
* `np.cumsum` lowers to parallel `IndexScan`, and `np.argmax` fuses the gain
  expression into a single `Carry` reduction — **2.9× faster than NumPy** at
  $N = 200{,}000$ samples.

---

## Verified benchmarks

### Same-device pass-on vs pass-off rewrite speedups (Apple Silicon CPU, `blas=False`)

To isolate the speedup of each algebraic pass on the **same device** (holding the
backend and CPU constant, toggling only the named pass via `skip_transforms`):

| Workload | Pass OFF (ms) | Pass ON (ms) | Rewrite Speedup | Pass Isolated |
|---|---:|---:|---:|---|
| **Mamba-2 SSD** (`T=2048, D=64, V=64`) | 58.25 ms | **1.08 ms** | **54.1×** | Law 4 (`chunkwise scan`) |
| **DeepSeek MLA Decode** (`Tq=4, Tkv=2048, dc=64, dh=256`) | 1.86 ms | **0.22 ms** | **8.4×** | Law 3 (`contraction reassociation`) |
| **Affine IIR Filter** (`T=262,144`) | 0.63 ms | **0.27 ms** | **2.3×** | Law 4 (`chunkwise scan`) |

### CPU vs eager NumPy (`examples/algebraic_gallery.py`, Apple Silicon, `blas=False`)

| Workload | Parakeet (ms) | NumPy (ms) | Speedup | Laws Fired |
|---|---:|---:|---:|---|
| **RBF Kernel Regression** (`512 × 2048 × 32`) | **4.37 ms** | 78.26 ms | **17.9×** | Law 1 (`sqdist`, zero $N\times M$ allocation) |
| **Exact 1-NN Search** (`512 × 2048 × 32`) | **1.69 ms** | 25.24 ms | **14.9×** | `Carry` + Vertical Fusion |
| **Mamba-2 SSD** (`T=2048, D=64, V=64`) | **1.74 ms** | 13.70 ms | **7.9×** | Law 4 (Chunkwise SSD) |
| **GBDT Hist + Best-Split** (`N=200k, B=256`) | **0.39 ms** | 1.11 ms | **2.9×** | `IndexScatter` + `IndexScan` + `Carry` |
| **One Min-Plus Matrix Product** (`256 × 256`) | **3.64 ms** | 6.24 ms | **1.7×** | Semiring Tiling |

### Apple Metal GPU vs eager CPU NumPy (`backend="metal"`, `blas=False`)

*(Cross-device comparison: Parakeet compiled to Apple Metal GPU vs the eager CPU NumPy
broadcast expression on the same Apple Silicon machine.)*

| Workload | Parakeet Metal GPU (ms) | Eager CPU NumPy (ms) | Ratio | Memory / Law Highlight |
|---|---:|---:|---:|---|
| **RBF Kernel Regression** (`1024 × 4096 × 32`) | **6.06 ms** | 312.76 ms | **51.6×** | Zero $N \times M$ matrix (Law 1 `sqdist`) |
| **One Min-Plus Matrix Product** (`512 × 512`) | **4.70 ms** | 36.77 ms | **7.8×** | Zero $N^3$ tensor (SRAM semiring tile) |
| **Causal FlashAttention** (`B=2, H=4, T=512, D=64`) | **7.49 ms** | 7.24 ms | **1.0×** | Zero $O(T^2)$ HBM allocation (Laws 1 + 5) |

The gallery's nested-map 1-NN requests a temporary row inside the query loop. On Metal and CUDA that loop requires CPU fallback. The broadcast NumPy form runs on both GPUs and allocates a full float32 distance matrix. The CPU schedule tiles both forms. See the [same-device release benchmarks](release-benchmarks.md).

### NVIDIA L4 GPU — End-to-End Batched nanoGPT Training Step (`tools/colab_nanogpt_bench.py`)

Unmodified NumPy `pk_gpt_loss` automatically compiles `flash_monoid`, `flash_monoid_bwd`,
`monoid_lse`, `cce_bwd`, 1-pass LayerNorm VJP, Split-K WMMA, and C-level `cuLaunchKernel`
dispatch with `0.0 up / 1.0 dn` transfers per step and exact loss parity:

| Config | Parakeet TF32 WMMA | Parakeet FP32 | PyTorch Eager | `torch.compile` | `jax.jit` |
|---|---:|---:|---:|---:|---:|
| `small` (`B=16, T=64, C=128, L=2`) | **6.44 ms** (compile 7.45 s) | **8.04 ms** | 9.15 ms | 3.58 ms (compile 20.0 s) | 1.58 ms (compile 19.3 s) |
| `med` (`B=32, T=128, C=256, L=4`) | **32.15 ms** (compile 29.6 s) | 47.05 ms | 19.28 ms | 15.15 ms (compile 29.7 s) | 9.18 ms (compile 23.4 s) |

*Parakeet TF32 WMMA runs **1.42× faster than PyTorch eager** on `small` while compiling
**2.7× faster** than `torch.compile` and `jax.jit`, with zero $O(T^2)$ attention or
$O(BT \cdot V)$ logit matrices materialized in HBM.*
