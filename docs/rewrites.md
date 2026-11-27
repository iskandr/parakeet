# Rewrite rules catalog

This reference catalogs the algebraic rewrites Parakeet applies between Python
source and native loops, and notes how Parakeet 1.x compares with the two NYU
PhD theses behind the original compiler:

* **[R]** [Alex Rubinsteyn](https://rubinsteyn.com/),
  [*Runtime Compilation of Array-Oriented Python Programs*](https://cs.nyu.edu/media/publications/rubinsteyn_alex.pdf)
  (NYU, 2014) — type specialization, data-parallel operators, and fusion rules.
* **[H]** [Eric Hielscher](https://github.com/hielscher),
  [*Locality Optimization for Data Parallel Programs*](https://cs.nyu.edu/media/publications/eric_h.pdf)
  (NYU, 2013; [arXiv:1304.1835](https://arxiv.org/abs/1304.1835)) — cache/register
  tiling and loop interchange over operator trees.

*(For conceptual motivation and visual walkthroughs, start with
[Operators & compilation walkthrough](primitives.md) and
[Algebraic laws](algebra.md).)*

## Two rewrite stages

1. **During elaboration:** array expressions remain *delayed* (index-to-value
   functions), so simple producer-consumer chains fuse on construction.
2. **In the middle end:** 12 passes in `parakeet.transforms.pipeline.PASSES`
   rewrite the emitted operator IR (`IndexMap`, `IndexReduce`, `IndexScan`,
   `IndexFilter`, `IndexExpand`, `IndexScatter`, `IndexWavefront`) before loop
   lowering.

---

## Fusion rules {#fusion}

![Vertical, filter-reduce, and sibling fusion collapse memory passes](assets/diag-fusion.svg)

### Vertical fusion (producer into consumer)

| Rule (`[R]` Fig. 6.1) | Python example | Implementation notes |
|---|---|---|
| `Map(g, Map(f, X))` → `Map(g ∘ f, X)` | `np.tanh(x * a) + b` | Elaboration + middle end; includes affine coordinate inversion for transpose, shifted-slice, and stencil fusion |
| `Map(g, IndexMap(f, σ))` → `IndexMap(g ∘ f, σ)` | `par.imap(...) * 2` | Elaboration + middle end |
| `OuterMap(g, Map(f, X₁), X₂)` → fused `OuterMap` | `par.allpairs(d, x*s, y)` | Elaboration |
| `Reduce(g, ⊕, Map(f, X))` → `Reduce(g ∘ f, ⊕, X)` | `np.sum(x * x)` | Elaboration + middle end |
| `Scatter(⊕, idx, Map(f, X))` → fused `IndexScatter` | `np.bincount(idx, weights=w*s)` | `transforms/fusion.py` |
| `Scan(g, ⊕, ε, Map(f, X))` → fused `Scan` | `np.cumsum(x * 2)` | Elaboration + middle end |
| `Map(g, Scan(f, ⊕, ε, X))` → `Scan(f, ⊕, g ∘ ε, X)` | `np.cumsum(x) * 2 + 1` | Consumer fuses into `IndexScan.emit` (`transforms/fusion.py`) |
| `Filter(g, p, Map(f, X))` → `Filter(g ∘ f, p ∘ f, X)` | `(x * 2)[x * 2 > 0]` | Predicate and values share fused map |
| `Reduce(g, ⊕, Filter(f, p, X))` → `FilterReduce` | `np.sum(x[x > 0])` | Guarded `IndexReduce`, zero index allocation (`transforms/fusion.py`) |
| `Filter(q, Filter(p, X))` → `Filter(p ∧ q, X)` | `x[x > 0][x[x > 0] < 1]` | Single short-circuiting compaction (`transforms/fusion.py`) |

**Safety guards (extending `[R]`):**

* **Alias safety:** every delayed array snapshots the store counter when
  created; if a potentially aliasing store occurs before consumption,
  elaboration restarts with that expression materialized eagerly.
* **Duplication guard:** `[R]` has no multi-use guard. Here, an expensive
  delayed array may be fused at most once; a second consumer or a use inside a
  loop restarts and evaluates it once.
* **Filter-reduce guards:** requires single use of the compacted indices and an
  explicit identity `init` (so `np.min(x[mask])` still raises `ValueError` on an
  empty mask). Inside the fused loop, the gather bounds check is eliminated.

### Nested and horizontal (sibling) fusion

| Category | Rule (`[R]` Figs. 6.2–6.3) | Python example | Notes |
|---|---|---|---|
| **Nested maps** | `IndexMap(λi. IndexMap(f, σ₂), σ₁)` → `IndexMap(f, σ₁ ++ σ₂)` | `[[f(x, y) for y in Y] for x in X]` | Flattens independent inner shapes in elaboration (cheap bodies) or `CombineNestedMaps` (expensive bodies) to expose full N-D parallelism |
| **Sibling reductions** | `Reduce(f, ⊕₁, X); Reduce(g, ⊕₂, X)` → fused 2-fold `Reduce` | `x.min(), x.max()` | Same index space, same seed status, both stay vectorizable (e.g. float `sum` and `max` stay separate to protect SIMD) |
| **Sibling maps** | `Map(f, X); Map(g, X)` → fused 2-output `Map` | `np.sin(x), np.cos(x)` | Single memory read; on CUDA, independent contiguous maps of differing sizes also batch horizontally into one disjoint-union kernel (e.g. Adam steps) |
| **Sibling scatters** | Two `IndexScatter`s sharing `idx` → fused multi-bin `IndexScatter` | GBDT grad + hess histograms | Single traversal over `idx` (`transforms/fusion.py`) |
| **Online fusion** | `m = max(x); s = sum(exp(x - m))` → 1-pass `(m, s)` fold | `logsumexp`, `softmax` | Uses `Fold.carries` (`fuse_online_logsumexp`) |

---

## Reductions, scans, and linear algebra

* **Associativity:** reductions parallelize and vectorize only when the combiner
  is recognizably one of `+`, `*`, `min`, `max`, `and`, `or`, `&`, `|`, `^`
  (however written). Any other user function folds strictly left-to-right.
* **Multi-axis reductions:** reduced axes are transposed last (zero-copy view)
  and folded in one pass per outer coordinate.
* **Parallel scans & filters:** associative scans use 3-stage chunk fold →
  prefix chunk totals → chunk scan; boolean filters use 3-stage count → prefix
  offsets → parallel scatter.
* **Matrix products & `einsum`:** `a @ b` lowers to `Map((i, j) ↦ Reduce(+, a[i, :] * b[:, j]))`,
  so prologues and epilogues fuse directly. Contiguous stride-1 inner loops are
  ensured by transposing `b` once when needed (`transpose(transpose(x)) → x`).
  Stacked `A @ B` reshapes to a single tall product. Standalone CPU float
  `A @ B` dispatches to NumPy's `cblas_?gemm` unless `config.blas = False`.

---

## Tiling and locality

| `[H]` thesis concept | Parakeet 1.x implementation |
|---|---|
| `PermuteReductions` (interchange strided inner reduction with stride-1 outer map) | **Target-aware reduction interchange** (`transforms/tiling.py`): on CPU, hoists strided associative inner reductions (`np.sum(X, axis=0)`) outside L1-sized output strips; skipped on GPU so warps coalesce across columns |
| `TiledMap` over nested adverbs | **Generalized `IndexTile` + `TileStage`** (`transforms/tiling.py`, `backends/tiled_gpu.py`): handles any rank `≥ 2` `IndexMap` wrapping one `IndexReduce` (arbitrary folds, `Carry` states, and prologues/epilogues) → cache + register tiles on CPU, `__shared__` / `threadgroup` tiles on CUDA and Metal |
| Register tiling & unrolling | **N-D register jamming** (`lowering.jam_register_tile`, `2×4` interior + exact remainder faces on CPU) and `4×4` / `2×2` per-thread register tiles on CUDA/Metal |
| Hardware matrix units | Automatic FP32 `simdgroup_matrix` on Apple Metal; opt-in TF32 `nvcuda::wmma` on CUDA `sm_80+` (`config.tensor_cores = True`) with double-buffered `cp.async` |
| `TiledReduce` / `TiledScan` | L1 column-strip `PermuteReductions`, invariant row hoisting (`normalize`), and chunkwise `TileScan` for affine/bilinear recurrences (general cache-tiled arbitrary `TiledScan` is not implemented) |
| Online tile-size search | Analytical compile-time sizing from `TargetProfile` (`parakeet/target.py`) and `config.tile_bytes` |

*(Stencils are not maps of reductions over two distinct operands, so they are
never tiled—matching `[H]`'s finding that tiling slows down blur stencils.)*

---

## Scalars, views, and shape-bucket specialization

Built on the fly during IR construction (leaving CSE, LICM, unrolling, and SIMD
vectorization to the system C/CUDA/Metal compiler):

* **Algebraic scalar simplifications:** constant folding (NumPy semantics),
  `x + 0`, `x * 1`, `x // 1 → x`, integer `x * 0 → 0` (not floats, preserving
  `-0.0`/`nan`), `x ** 2 → x * x`, `x ** 0.5 → sqrt(x)`, constant `select` /
  boolean short-circuiting.
* **Zero-copy views:** slices of slices, transposes, contiguous `reshape`, and
  `expand_dims` reduce purely to offset/stride arithmetic.
* **Redundant check elimination:** dimensions already proven equal by a prior
  broadcast check are not re-checked; loop-bounded indices omit bounds checks.
* **Value & shape-bucket specialization (`[R]` §6.4 extended):**
  - Size-1 dimensions are always static constants (fixing `[R]`'s
    `200×1 + 1×300` broadcasting corner case statically).
  - Arguments required at compile time (`axis=`, `keepdims=`, tuple lengths)
    specialize automatically.
  - Tiled/GPU passes propagate finite power-of-2 shape buckets and divisibility
    (`DimInfo(lo, hi, align)` via `config.shape_specialization`), guarding only
    queried dimensions so aligned sizes (`128, 256, 512, 1024, 4096`) share one
    branchless fast-path specialization.

---

## Operator differentiation rules

`par.grad` attaches one reverse rule per operator (see
[Automatic differentiation](autodiff.md) and [Law 2](algebra.md#law-2-algebraic-rematerialization-in-reverse-mode-ad)):

| Operator | Adjoint given output cotangent `ȳ` |
|---|---|
| `Map(f, X)` element-wise | `Map(∂f/∂x · ȳ, X)`, summed over broadcast axes |
| `Reduce(+, X)` | `ȳ` broadcast to `X` |
| `Reduce(max, X)` | `ȳ` where `X == result`, split equally across ties |
| `Reduce(*, X)` | Zero-safe `ȳ · result / X` |
| `Reduce(f, X)` (arbitrary `f`) | `O(√N)` checkpointed reverse replay |
| `Scan(+, X)` | Reverse `Scan(+)` of `ȳ` |
| `A @ B` | `ȳ @ Bᵀ` and `Aᵀ @ ȳ` |
| View (slice, transpose, reshape) | Matching view of `ȳ` |
| Gather `X[idx]` / Scatter `scatter_add` | `scatter_add(ȳ, idx)` / gather `ȳ[idx]` |
| `sort(X)` | Scatter `ȳ` by the inverse sorting permutation |
| `Map(f, X)` (general `f`) | Per-element differentiated `f` with thread-privatized adjoints |
| Runtime `if c: A else: B` | Both pure branches evaluated; values and adjoints selected by `c` |

---

## Summary vs. the 2013–2014 theses

| Added in Parakeet 1.x | Present in `[R]` / `[H]` only |
|---|---|
| Composable forward/reverse/higher-order AD + Law 2 rematerialization | IR-to-IR differentiation pass over lowered loops (1.x uses an algebraic tape during elaboration) |
| Rescaled-monoid tiling (attention, CCE, RBF), Law 3 reassociation, Law 4 chunkwise scans, Law 5 interval pruning, Law 6 scatter & pairwise folds | Recursive tiling of arbitrary deeper adverb nests and general `TiledScan` for arbitrary non-affine scans |
| Alias & duplication safety guards on fusion; static size-1 broadcast specialization; `DimInfo` shape-bucket specialization | Online runtime tile-size search |
| Parallel scans, filters, `RaggedArray` expand, wavefront DP, `einsum`, and runtime bounds/shape exceptions | Custom low-level scalar optimizer (1.x delegates scalar opts to clang/gcc/NVRTC/MSL) |
| Unified CPU + Apple Metal + NVIDIA CUDA with device residency and hardware matrix units | — |

---

**Next:** [Compiler architecture](internals.md) walks through how elaboration,
all 12 middle-end passes, lowered loop IR, and C/Metal/CUDA codegen fit
together.
