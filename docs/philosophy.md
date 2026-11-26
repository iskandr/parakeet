# Ideas, history, and boundaries

## Five ideas that shape Parakeet

### 1. Specialize on types and shapes at call time

Python is dynamically typed, yet numerical functions are almost always called
with the same dtypes and ranks at a given call site. Parakeet treats your
Python function as a template and compiles a native *specialization* for each
distinct combination of argument dtypes, array ranks, memory layouts
(contiguous vs. strided), and container structure.

When an expression's type or rank depends on a value; such as `axis=`, a tuple
length, or a unit dimension of length `1` that controls broadcasting; Parakeet
specializes on that value automatically. That resolves the broadcasting corner
case noted in Rubinsteyn's thesis (`200×1 + 1×300`) statically at compile time.

### 2. Lower everything to a small index-operator core

Instead of maintaining hundreds of hand-written C or GPU kernels for every
NumPy function, Parakeet defines array expressions, comprehensions, matrix
products, `einsum`, and user-written adverbs in terms of a small set of core
index-operator statements:

* `IndexMap`: independent computation over an index space
* `IndexReduce`: associative or sequential folds (including multi-carry state
  such as `argmin`/`argmax` and online `logsumexp`, plus guarded filtered
  reductions)
* `IndexScan`: prefix folds with an optional emit stage
* `IndexFilter`: 1-to-0/1 stream compaction of indices matching a predicate
* `IndexExpand`: 1-to-K ragged compaction (`flatmap`, `filter_pairs`)
* `IndexScatter`: associative index-directed scatter-add (`np.bincount`,
  `parakeet.scatter_add`)
* `IndexWavefront`: anti-diagonal / banded 2D semiring dynamic programming
  (`wavefront`, `batched_wavefront`, `soft_align`)
* `IndexTile`: explicit cache and GPU shared-memory tile staging emitted by
  the middle end

You can inspect a function at this level at any time via
`f.specialize(*args).operators` or `parakeet.show_rewrite(f, *args)`.

### 3. Fuse algebraically before lowering to loops

Because programs stay in this small operator IR until the end of the pipeline,
optimizations are algebraic identities rather than fragile loop-nest analyses:

* **Vertical fusion** collapses producer-consumer chains: map into map
  (including transposes, slices, stencils, and matmul epilogues like
  `np.tanh(a @ b + c)`), map into reduce/scan/filter/scatter, map after scan,
  filter into filter, and filter into reduce (turning `np.sum(x[x > 0])` into
  one zero-allocation loop).
* **Nested map flattening** turns a map inside a map into one multi-dimensional
  `IndexMap`, exposing the full index space to CPU threads and GPU grids.
* **Sibling and online fusion** lets independent consumers over the same index
  space share a pass (`x.min(), x.max()`, `np.sin(x), np.cos(x)`, or sibling
  histograms with the same bin IDs) and merges dependent `max` + `sum(exp(x - m))`
  pairs into a single associative pass.
* **Dead-operator elimination** drops pure operators and allocations that
  nothing reads before loops are emitted.

Parakeet 1.0 adds the safety conditions the 2014 thesis omitted: cost and
single-use guards so fusion never silently duplicates expensive work, alias
checks, and explicit associativity checks so non-associative user folds stay
strictly left-to-right.

### 4. Tile for cache and GPU shared memory from the same IR

Following Hielscher's 2013 thesis, locality transformations operate directly on
the operator statements before loop lowering:

* **Reduction interchange** hoists strided inner reductions (such as
  `np.sum(X, axis=0)`) outside L1-sized strips of contiguous output columns on
  CPU, while leaving them coalesced across threads on GPU.
* **Invariant hoisting** squeezes unit `keepdims=True` axes and lifts row-level
  reductions so normalizations like `softmax` and `layer_norm` stay L1-resident.
* **Cache, register, and GPU SRAM tiling** rewrites maps wrapping reductions
  (`a @ b`, `allpairs`, `einsum`, tropical products, and fused prologues or
  epilogues) into nested cache and register tiles on CPU and cooperatively
  staged shared-memory tiles on CUDA and Metal. On Apple Silicon Metal, float32
  contractions use `simdgroup_matrix` automatically; on CUDA `sm_80+`,
  `config.tensor_cores=True` enables `nvcuda::wmma` TF32 Tensor Cores.
* **Algebraic laws for modern reductions** extend the exact same tiling
  machinery to rescaled semiring monoids (online softmax, Gaussian kernel
  regression, attention, cut cross-entropy), contraction reassociation,
  chunkwise affine/bilinear scans, and tile-interval mask pruning. See
  [Algebraic laws](algebra.md).
* **Persistent GPU residency** (`parakeet.to_device` / `to_host`) keeps
  arrays and parameter tuples on CUDA or Metal across calls without repeated
  host copies.

### 5. Inspectable, fail-loud compilation

* `parakeet.verify` checks scoping, types, and buffer ownership before passes,
  after every pass, and after lowering.
* `f.specialize(*args)` exposes `.operators`, `.ir`, `.source`, and `.kernels`.
* `par.show_rewrite(f, *args)` summarizes which algebraic optimizations fired.
* `parakeet.find_broken_transform` bisects the middle-end passes one by one.
* Unsupported Python or NumPy features fail at compile time with the exact
  source caret and inlined call chain instead of silently falling back.

---

## What does *not* work yet

Parakeet is a numerical compiler, not a drop-in replacement for all of CPython
or SciPy. Knowing the boundary up front avoids surprises.

### Python language and types

| Works | Not supported yet |
|---|---|
| `bool`, `int8`–`int64`, `uint8`–`uint64`, `float32`, `float64` scalars and arrays (any rank or strides) | Complex numbers (`complex64`/`complex128`), native `float16` compute (`float16` is widened to `float32`), structured/record dtypes, object or string arrays |
| Nested tuples, `NamedTuple`s, fixed string-key dicts, `@dataclass`es (frozen or non-frozen) with pure methods / `@property` / `__call__`, `slice`, `range`, `zip`, `enumerate` | Dynamic dictionary keys, `set`, runtime string manipulation, arbitrary non-dataclass user classes with heap mutation |
| List literals and comprehensions as fixed-shape arrays | Mutable or dynamically resized Python lists (`.append`, `.extend`, ragged lists) |
| `if`/`elif`/`else`, `for`, `while`, `for`/`while ... else`, `break`, `continue`, early `return`, structural `match`/`case`, `isinstance`/`hasattr`/`getattr`, nested `def`/`lambda`, closures, `*args`/`**kwargs`, `functools.partial`, `operator.*`, `assert`, `raise` | `try`/`except`, `with`, generators (`yield`), `global`/`nonlocal`, `import` inside `@jit`, recursion |
| Static array rank; compile-time specialization on `axis`, `keepdims`, `dtype`, tuple lengths | Value-dependent output rank (e.g. `np.squeeze(x)` without `axis=`), variables whose types cannot unify |

### NumPy and SciPy coverage

About 200 core NumPy functions and ufuncs are built in (see
[Supported functions](supported.md)), covering arithmetic, reductions, scans,
sorting, masks and fancy indexing, `@`/`dot`/`einsum`, `np.linalg.norm`, and
parallel `np.random`. Not yet built in:

* Matrix decompositions and solvers (`np.linalg.svd`, `solve`, `inv`, `eig`,
  `cholesky`, `qr`, `det`, `lstsq`)
* Spectral transforms (`np.fft.*`)
* Variable-cardinality set operations (`np.unique`, `np.lexsort`, `np.isin`)
* SciPy modules (`scipy.special`, `scipy.optimize`, `scipy.ndimage`, ...)

### Automatic differentiation (`grad`, `jvp`, `vjp`, `hvp`)

Parakeet provides reverse-mode (`par.grad`, `par.value_and_grad`, general
non-scalar `par.vjp`), forward-mode (`par.jvp`, `par.linearize`), linear
transposes (`par.linear_transpose`), and higher-order differentiation
(`par.hvp`, `par.jacfwd`, `par.jacrev`, `par.hessian`) across element-wise
operations, reductions (`sum`, zero-safe `prod`, `min`, `max`, general folds),
`cumsum` and general `par.scan`, `sort` and `topk`, `FrozenHashTable.get` and
`SortedIndex.get`, `@`, `einsum`, views and concatenation, gather and scatter
(`np.add.at`, canonical `for`-loop scatter-add, `par.scatter_add`),
`map`/`allpairs` with user functions, pure runtime `if` branches, and
checkpointed `for`/`while` loops; enough to train transformers
([`examples/nanogpt.py`](https://github.com/iskandr/parakeet/blob/master/examples/nanogpt.py))
and convnets
([`examples/convnet.py`](https://github.com/iskandr/parakeet/blob/master/examples/convnet.py)).
Current boundaries:

* **Elaboration-time tape & symbolic JVP:** reverse-mode differentiation
  records operators on a tape during elaboration (with an algebraic tape
  optimizer for zero-$O(T^2)$ attention VJP/HVP, linear cross-entropy, softmax,
  and norms) rather than running as a separate IR-to-IR pass over lowered loops.
* **Loops over differentiated values:** small static `for` loops (`<= 32`)
  unroll for cross-iteration fusion; runtime-bound and long `for` loops use
  IR-level `O(sqrt(T))` checkpointed reverse loops; data-dependent `while`
  loops use fixed-slot stride-doubling checkpointing; forward-mode `jvp`
  streams tangents in O(1) memory; and `par.checkpoint`, `par.stop_gradient`,
  `@par.custom_vjp`, and `@par.custom_jvp` provide explicit control over
  rematerialization and custom/implicit rules. Loops with `break`/`continue`
  over active state remain refused loudly.
* **Pure runtime `if` branches in reverse mode:** both branches execute and are
  selected by the condition, so branches touching differentiated values cannot
  contain array stores, `return`, `break`, or `raise`.
* **Missing derivative rules (refused loudly):** `cumprod`, `convolve`,
  arbitrary indexed overwrites (`out[idx] = v`; use `np.add.at`,
  `for i: out[idx[i]] += v`, or `par.scatter_add` for scatter-add), and
  overwriting an array still needed by an earlier backward rule.
* **Higher-order derivatives:** forward-over-reverse `par.hvp` and
  `par.hessian` compose through element-wise ops, reductions, `@`, views,
  checkpointed loops, LayerNorm, and zero-$O(T^2)$ attention; reverse-over-reverse
  `grad(grad(...))` also supports `np.stack`, `np.cumsum`, gather indexing,
  scatter-add, and scalar 1-to-1 `par.map`, while refusing non-scalar user
  `map` bodies loudly.

### GPU and locality boundaries

* **Apple Metal has no hardware `float64`:** `float64` loops fall back to CPU
  cores with a warning; use `float32` on Metal.
* **No per-thread heap allocation on GPU:** a parallel loop that allocates
  temporary arrays or selects between buffer aliases inside its body runs on
  CPU instead of CUDA or Metal.
* **Dense GEMM vs. cuBLAS:** CPU backends call NumPy's BLAS for standalone
  float `A @ B`. GPU backends run Parakeet's own shared-memory tiled kernels
  (including Metal `simdgroup_matrix` and optional CUDA TF32 WMMA), which fuse
  prologues and epilogues in one kernel but trail hand-tuned cuBLAS on large
  standalone matrix multiplies.
* **Analytical tile sizing:** tile sizes are picked analytically from target
  cache and shared-memory budgets (`config.tile_bytes`) rather than Hielscher's
  online runtime search. Chunkwise `TileScan` handles affine and bilinear
  recurrences; general cache-tiled `TiledScan` for arbitrary scans is not
  implemented.

---

## History

Parakeet started in OCaml as an array compiler modeling the array language Q,
later shifted to modeling Python, and was eventually reimplemented in Python at
New York University by **Alex Rubinsteyn** (`@iskandr`), with major
contributions from **Eric Hielscher** (`@hielscher`) and **Nathaniel Weinman**,
guidance from **Dennis Shasha**, and advice from **Russell Power**. That earlier
work was hosted at `parakeetpython.com` and documented in:

* **Alex Rubinsteyn**, [*Runtime Compilation of Array-Oriented Python Programs*](https://cs.nyu.edu/media/publications/rubinsteyn_alex.pdf)
  (PhD thesis, NYU, 2014): type and value specialization, data-parallel adverbs,
  and algebraic fusion.
* **Eric Hielscher, Alex Rubinsteyn, and Dennis Shasha**,
  [*Locality Optimization for Data Parallel Programs*](https://arxiv.org/abs/1304.1835)
  (NYU CS Technical Report TR2013-955 / Eric Hielscher's 2013 NYU PhD thesis
  work): `TiledMap`, `TiledReduce`, and `TiledScan` for unified cache tiling,
  register blocking, and loop interchange over adverb trees.
* Papers and talks: [HotPar '12](https://www.usenix.org/conference/hotpar12/workshop-program/presentation/rubinsteyn)
  (*Parakeet: A Just-In-Time Parallel Accelerator for Python*),
  [PyData Boston '13](https://vimeo.com/73895275), and
  [PyData NYC '13](https://vimeo.com/79556629) (*Python on the GPU with Parakeet*).

When the Python 2 codebase was retired, its README noted that general Python is
a poor fit for high-performance numerical code and that compilers work much
better over a small set of explicit data-parallel operators. Parakeet 1.x is a
rewrite for Python 3.11+ and NumPy 2 that keeps `@jit` and NumPy alongside
those explicit operators, fixes the soundness gaps from the earlier design, and
adds Apple Metal, NVIDIA CUDA, composable automatic differentiation, and tiled
monoidal reductions.
