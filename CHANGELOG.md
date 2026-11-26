# Changelog

## 1.6.0 (November 25, 2026)

Unboxed `complex64`/`complex128` arithmetic and Wirtinger AD, plus compiled
byte/ASCII parsing helpers (`split_bytes`, `parse_int64`, `parse_float32`,
`parse_float64`) for FASTA/CSV ingestion (Python Semantics Wave 3, closing
`#81`).

### Unboxed `complex64` and `complex128`

- **Zero-copy `(re, im)` lowering:** `np.complex64`, `np.complex128`, and
  Python `complex` values lower cleanly onto pairs of real `float32`/`float64`
  views and scalar expressions, working out of the box across CPU, multicore,
  Apple Metal (`complex64`), and CUDA without IR or backend changes.
- **Complex math and linear algebra:** Supports `+`, `-`, `*`, `/`, `**`,
  `==`, `!=`, `abs`, `conj`/`conjugate`, `.real`, `.imag`, `np.angle`,
  `np.exp`, `np.log`, `np.sqrt`, `np.sin`, `np.cos`, `np.matmul`/`@`, `np.dot`,
  `np.vdot`, `np.sum`, `np.mean`, `np.cumsum`, `np.where`, `np.concatenate`,
  `np.stack`, indexing/slicing, and in-place updates with NEP 50 promotion.
- **Wirtinger automatic differentiation:** `par.grad`, `par.value_and_grad`,
  and `par.jvp` differentiate real-valued losses over `complex64`/`complex128`
  arrays automatically.

### Byte and ASCII parsing helpers

- **`par.split_bytes(buf, sep, skip_empty=False)`:** Data-parallel delimiter
  splitting of `uint8` / `bytes` buffers into `RaggedArray[uint8]` with zero
  `PyObject` allocation.
- **`par.parse_int64`, `par.parse_float32`, `par.parse_float64`:** Compiled
  ASCII integer and floating-point parsers (with optional signs, decimals, and
  exponents) operating on single `uint8` slices or in parallel over
  `RaggedArray[uint8]`.
- **New runnable example:** Added `examples/complex_fft_and_fasta.py`
  (Cooley-Tukey radix-2 FFT, spectral filter Wirtinger gradient, and compiled
  FASTA/CSV byte ingestion).

## 1.5.0 (November 25, 2026)

Local Python lists, GPU-safe fixed-capacity scratch buffers, dynamic-shape
array/ragged loop rebinding, and zero-allocation `itertools` combinators
(Python Semantics Wave 2, closing `#80`).

### Lists, growable CPU vectors, & GPU scratch stacks

- **Compile-time static `list`:** Straight-line and unrolled `outs = []` with
  `.append()`, `.extend()`, `.pop()`, `.clear()`, indexing/slicing, and
  consumption by `np.stack`, `np.concatenate`, `np.array`, and
  `RaggedArray.from_sequences` lowers away before IR emission and composes
  seamlessly with `par.grad`.
- **Runtime growable CPU `list[T]`:** Lists mutated inside runtime `for`/`while`
  loops or dynamic `if` branches automatically promote to a refcounted
  homogeneous growable buffer (`buf`, `len`, `cap`) with geometric doubling and
  zero-copy conversion to `ArrayVal` (`par.config.verify=True` clean).
- **`par.fixed_list(capacity, dtype)`:** Fixed-capacity stack-allocated scratch
  buffer (`ir.Alloc(..., stack=True)`) emitted natively as thread-local stack
  arrays on CPU, multicore, CUDA, and Apple Metal with zero `malloc` and runtime
  bounds/capacity checks.
- **Dynamic-shape loop rebinding:** Verified runtime shape rebinding for both
  `ArrayVal` (e.g., `arr = arr[arr > thresh]`) and `RaggedArray` across `while`
  and `for` loops under `par.config.verify=True`.
- **Static `itertools` & generators:** Zero-allocation compilation for
  `itertools.pairwise`, `itertools.islice`, `itertools.chain`,
  `itertools.product`, and `itertools.batched` in `for` loops and generator
  expressions (`sum(...)`, `np.array(...)`).

## 1.4.0 (November 25, 2026)

Broader idiomatic Python semantics lowered cleanly at elaboration time with
zero `PyObject` overhead: non-frozen `@dataclass` records with pure methods,
`@property`, `__call__`, and functional field rebinding; full structural
`match`/`case` destructuring; compile-time `isinstance`/`hasattr`/`getattr`
dead-branch pruning; and `*args`/`**kwargs`, `functools.partial`, and
`operator.*` call ergonomics (Python Semantics Wave 1, closing `#79`).

### Non-frozen `@dataclass`, methods & `@property`

- **Non-frozen `@dataclass` records:** Both frozen and regular `@dataclass`
  objects compile to unboxed compile-time records across `@par.jit`, `par.grad`,
  and `par.jvp` (with discrete metadata fields treated as zero-tangent static
  leaves).
- **Functional local field rebinding:** `obj.x = new_x`, nested `obj.a.b = v`,
  and augmented updates `obj.x += v` rebuild the compile-time record chain
  functionally (value semantics, zero heap mutation); including across loops
  and branches.
- **Pure methods, `@property`, `@staticmethod`, `@classmethod`, & `__call__`:**
  Methods and properties on `@dataclass` and `NamedTuple` records inline like
  ordinary helper functions; bound methods on records can also be passed
  directly from Python into `@par.jit` and `par.map`.

### Structural `match`/`case`, `isinstance`/`hasattr`, & call ergonomics

- **Extended structural `match`/`case`:** Added `MatchMapping` (`case {"k": v, **rest}:`),
  starred array patterns (`case [first, *mid, last]:`), and class patterns for
  non-frozen dataclasses, `RaggedArray`, `np.ndarray`, `dict`, and `tuple`.
- **Compile-time `isinstance`, `hasattr`, & `getattr`:** Resolved statically
  against the specialization key (including Python 3.10+ union types `A | B`,
  user dataclasses, `NamedTuple`, and `RaggedArray`) with guaranteed dead-branch
  pruning.
- **Call ergonomics:** Full `**kwargs` collection and `**d` unpacking,
  `functools.partial` (both inside `@par.jit` and passed from Python), and
  `operator.*` callables (`operator.add`, `operator.mul`, `operator.itemgetter`,
  `operator.attrgetter`, etc.) including associative `par.reduce`/`par.scan`
  combiners.
- **New runnable gallery example:** Added `examples/protein_and_layers.py`
  (`ProteinBackbone` + `MLPBlock` with polymorphic dense/`RaggedArray` dispatch
  and reverse-mode AD).

## 1.3.0 (November 25, 2026)

Composable forward-mode, general reverse-mode, and higher-order automatic
differentiation (`jvp`, `vjp`, `linearize`, `linear_transpose`, `hvp`,
`jacfwd`, `jacrev`, `hessian`, `@custom_jvp`), zero-$O(T^2)$ rematerialized
attention JVP/HVP, idiomatic `np.add.at` and `for`-loop scatter lifting, and a
documentation and examples overhaul (AD Upgrade Wave 2, closing `#76`, `#77`,
and `#78`).

### Forward-mode AD, general VJPs & linear transposes

- **`par.jvp` & `@par.custom_jvp`:** Symbolic forward-mode differentiation over
  scalars, arrays, matmul/einsum, reductions (tie-averaged `min`/`max`,
  zero-safe `prod`), scans, folds, `map`, gather/scatter, `if`/`else`,
  runtime `for`/`while` loops (in $O(1)$ extra memory), `NamedTuple` / frozen
  `@dataclass` records, and `RaggedArray`s.
- **General non-scalar `par.vjp`, `par.linearize`, & `par.linear_transpose`:**
  Full reverse-mode pullbacks for array, tuple, record, and `RaggedArray`
  outputs, plus exact linear operator transposes satisfying
  `⟨JVP(x; v), u⟩ == ⟨v, VJP(x; u)⟩`.
- **Higher-order AD (`par.hvp`, `par.jacfwd`, `par.jacrev`, `par.hessian`):**
  Forward-over-reverse Hessian-vector products across MLPs, LayerNorm,
  checkpointed `for` loops, and FlashAttention. Also lifted reverse-over-reverse
  `grad(grad(...))` cliffs for `np.stack`, `np.cumsum`, gather indexing,
  scatter-add, and scalar 1-to-1 `par.map`.

### Composable algebraic AD & indexing adjoints

- **Zero-$O(T^2)$ attention JVP & HVP:** Fused dual `(S, dS) -> (O, dO)`
  monoidal tiling and parallel rematerialized HVP tangent emission drop all
  quadratic $O(T^2)$ allocations during both `par.jvp` and `par.hvp` (reported
  by `par.show_rewrite`).
- **Composable `ALGEBRAIC_AD_RULES` registry:** Unified FlashAttention,
  Cut Cross-Entropy, softmax, and LayerNorm / RMSNorm rematerialization rules
  and enabled algebraic tape optimization inside checkpointed runtime `for`
  loops.
- **`topk`, `sort`, `FrozenHashTable`, & `SortedIndex` AD:** Added reverse-mode
  and forward-mode differentiation for `par.topk` / `par.topk_fold`, `np.sort`,
  `FrozenHashTable.get`, and `SortedIndex.get`.
- **Multi-store vertical fusion fix:** Hardened `match_single_output_map` and
  `match_single_output_scan` to reject multi-store prefixes, fixing a compiler
  bug exposed by fused primal+tangent `IndexMap`s.

### Idiomatic scatter lowering & documentation overhaul

- **`np.add.at` and `for`-loop scatter lifting:** `np.add.at(out, idx, vals)`
  and canonical `for i in range(n): out[idx[i]] += vals[i]` loops (including
  multi-histogram bodies sharing an index binding `b = bin_ids[i]`) now lift
  directly to `IndexScatter`, fuse sibling histograms into one privatized
  traversal, and differentiate cleanly under both `grad` and `jvp`.
- **Consistent NumPy idioms & pedagogical gallery:** Standardized docs and
  examples on `np.*` helpers (`np.minimum`, `np.add`, `np.add.at`, etc.)
  alongside `@par.jit` / `par.*` operators, replaced bare `Law 1..6` shorthand
  outside `docs/algebra.md` with plain-English optimization descriptions, and
  reordered the [Kernel gallery](https://rubinsteyn.com/parakeet/gallery/)
  progressively from pairwise NumPy fusion to attention and end-to-end training.

## 1.2.0 (November 25, 2026)

Major automatic differentiation upgrade: IR-level checkpointed reverse loops
for runtime-bound `for`/`while` loops, `par.fold`, and general `par.scan`,
plus `par.stop_gradient`, `@par.custom_vjp`, and `@par.checkpoint` (AD Upgrade
Wave 1, closing `#75` and advancing `#77`).

### Reverse-mode AD through runtime & long loops

- **IR-level $O(\sqrt{T})$ checkpointed reverse `for` loops:** Loops over
  runtime-length ranges or arrays (`for t in range(T):`, `for t in range(weights.shape[0]):`)
  and long static loops (`> 32` iterations) compile once to a two-level
  checkpointed reverse loop in IR without unrolling or recompiling per length,
  removing the `MAX_UNROLL = 4096` cliff.
- **IR-level $O(\log T)$ checkpointed reverse `while` loops:** Runtime-converging
  `while` loops (fixed-point solvers, adaptive ODE/relaxation steps) now
  differentiate through a fixed-slot (`64` slots) stride-doubling online
  checkpoint schedule with bounded memory.
- **Checkpointed $O(\sqrt{N})$ `par.fold` & differentiable general `par.scan`:**
  Arbitrary non-monoidal `par.reduce`/`par.fold` and 1-D `par.scan` now
  differentiate via checkpointed reverse segments without storing full $O(N)$
  histories.
- **Rewrite inspection:** `par.show_rewrite(par.grad(...), ...)` now reports
  `AD: Checkpointed Reverse Loop` when checkpointed reverse loops are emitted.

### Explicit AD control primitives

- **`par.stop_gradient(x)`:** Zero-copy gradient barrier across scalars, arrays,
  tuples, `RaggedArray`, and records (for straight-through estimators, target
  networks, and stop-gradient RL losses).
- **`@par.custom_vjp` (`.defvjp(fwd, bwd)`):** User-defined reverse-mode rules
  supporting multi-output pytree values, auxiliary residuals, and higher-order
  differentiation.
- **`@par.checkpoint` (rematerialization):** Explicit activation checkpointing
  that re-runs the forward block during backpropagation while composing
  cleanly with Law 2 FlashAttention / LayerNorm rematerialization.
- **Record static metadata leaves:** Non-float fields inside `NamedTuple` and
  frozen `@dataclass` records now pass through AD as zero-tangent static leaves.

## 1.1.0 (November 25, 2026)

Major expansion of Parakeet's GPU schedules, algebraic data structures,
scientific/bioinformatics primitives, and documentation; unified under one
top-level `import parakeet as par` (`@par.jit`, `par.*`) interface.

### Unified namespace, docs & benchmarks

- **Single unified `parakeet` namespace:** Exported `fold`, `topk`,
  `topk_fold`, `tanimoto_topk`, `ipa_attention`, `show_rewrite`,
  `RewriteReport`, `scatter`, `iscan`, `iscatter`, and `ifilter_reduce`
  directly on top-level `parakeet` (with `parakeet.dsl` retained as a
  backward-compatible alias). Standardized examples and docs on
  `import parakeet as par` (`@par.jit`, `par.*`).
- **Reproducible benchmark figures & overhaul:** Added
  `tools/gen_benchmark_figures.py` generating per-replicate scatter + IQR +
  median SVGs (`bench-pairwise-l4.svg`, `bench-scaling-1nn.svg`,
  `bench-workloads-l4.svg`) directly from `docs/assets/colab-l4-0dcb7d5.json`,
  reorganized the [Benchmarks](https://rubinsteyn.com/parakeet/release-benchmarks/)
  page by workload category, connected `README.md` directly to runnable
  implementations, and updated the NYU history links (Rubinsteyn 2014 thesis,
  Hielscher et al. 2013 `arXiv:1304.1835`, HotPar '12).
- **Bioinformatics & structural biology guide:** Added the
  [Bioinformatics & structural biology](https://rubinsteyn.com/parakeet/bioinformatics/)
  gallery (`examples/bio_gallery.py`) and reproducible benchmark suite
  (`benchmarks/bio_workloads.py`).

### GPU schedules & matrix units

- **Tiled GPU pairwise folds:** Broadcasting NumPy and nested-map/structured
  `par.fold` exact 1-NN and inverse-distance smoothing now stream shared-memory
  score tiles (`16 KiB`/block) on CUDA and Metal with exact first-index tie and
  NaN semantics; **4.8× faster** (`2.60 ms → 0.54 ms`) and **21× less global
  scratch** (`64 MiB → 3 MiB`) at `1,024 × 16,384 × 32` on NVIDIA L4.
- **Hardware matrix units in monoidal kernels:** Enabled Metal
  `simdgroup_matrix` and CUDA TF32 WMMA inside Law 1 rescaled-monoid kernels
  (FlashAttention, RBF, linear cross-entropy, nanoGPT training).
- **Device-resident GPU scans & reductions:** Kept GPU prefix scans, small GPU
  reductions, and extrema values device-resident without host round-trips.

### New algebraic primitives & data structures

- **First-class `RaggedArray` (CSR) & segmented operators:** `RaggedArray`,
  `segment_ids`, `segmented_map`, `segmented_reduce`, `segmented_scan`,
  `segmented_softmax`, and differentiable `varlen_flash_attention` with ragged
  reverse-mode AD (`par.grad` preserves offsets structurally).
- **1-to-K ragged compaction (`IndexExpand`):** Added `par.flatmap` and
  `par.filter_pairs` (two-pass count → prefix-sum → emit) with dead-output
  elimination and AD gather support.
- **Parallel sorting, search & lookup tables:** Parallel radix `sort` /
  `argsort` / `unique`, fusible `searchsorted`, `SortedIndex`, and static
  open-addressed `FrozenHashTable`.
- **3D spatial pruning & block-sparse attention (Law 5 generalized):**
  `morton_order`, 3D AABB `spatial_cutoff_sum`, and CSR `BlockMask`
  `block_sparse_attention` with forward + backward AD.
- **2D / banded semiring DP (`IndexWavefront`):** Added `par.wavefront`,
  `par.batched_wavefront`, and Law 2 checkpointed differentiable
  `par.soft_align` (Smith-Waterman / Needleman-Wunsch / soft-DTW).
- **Vector-valued Carry folds & pair-biased / IPA attention:** Multi-slot
  `Carry` folds powering exact `par.topk`, `par.topk_fold`, `par.tanimoto_topk`,
  and Law 1+2 `par.ipa_attention`.
- **Zero-copy bytes, bitops & Python ergonomics:** Zero-copy `uint8` bytes /
  `mmap` inputs, `popcount`/`clz`/`ctz`/`rotl`/`rotr`/`bswap`, `bf16_to_f32` /
  `f32_to_bf16`, `IntEnum`, scalar `match`/`case`, frozen `@dataclass`
  `dataclasses.replace`, and structured NumPy views.

## 1.0.0 (November 24, 2026)

A ground-up rewrite for modern Python.  The `@jit` decorator, the adverbs
(`map`, `reduce`, `scan`, `allpairs`, `imap`, ...) and the overall idea are
unchanged; everything underneath is new.

### Algebraic operators

- The same CPU tile schedule serves exact nearest-neighbor queries written as
  broadcasting NumPy or nested parallel maps. It merges a distance and index
  per query and preserves the first index on ties.
- Two weighted histograms over shared bin IDs can fuse into a single traversal.
  Structured folds accept tuples and named records with a user-defined merge.
- Algebraic rewrites cover online normalized sums, their gradients, reassociation
  of contractions, and chunked affine scans. The rewrite inspector shows which
  passes fired for a given function and input.
- The [gallery](https://rubinsteyn.com/parakeet/gallery/) includes runnable
  examples. The [release benchmarks](https://rubinsteyn.com/parakeet/release-benchmarks/)
  compare five workloads with general-purpose and specialist implementations on
  a four-vCPU worker and an NVIDIA A10.

### Highlights

- **`grad` and `value_and_grad`:** reverse-mode differentiation as an adverb.
  Element-wise operators, reductions, matrix products, views, `map` and
  `allpairs` with your own functions all have derivative rules; loops of known
  length are unrolled; anything without a rule raises instead of returning a
  wrong gradient. `examples/nanogpt.py` and `examples/convnet.py` train with it.
- **More of `grad`:** `cumsum`, `sort`, `einsum`, stacks of matrices, folds
  with any function, and `if` statements on run-time conditions (both branches
  run, results are selected). The derivative of a `map` runs in parallel.
- **A middle end:** elaboration now produces operator statements (`IndexMap`,
  `IndexReduce`, `IndexScan`, `IndexFilter`, printed by `spec.operators`) and a pipeline of passes rewrites
  them before they become loops. Tiling is one such pass, a pattern match on
  the IR instead of a trial elaboration; **vertical fusion** (`Map→Map`
  including matmul epilogues, `Scan→Map` into `emit`, `Map→Filter`,
  `Map→Reduce`), **filter fusion** (`Filter→Filter` and `Filter→Reduce`:
  `np.sum(x[x > 0])` is one loop and allocates nothing), **sibling fusion**
  (`x.min(), x.max()` and `np.sin(x), np.cos(x)` each share one loop),
  **reduction interchange** (`np.sum(X, axis=0)` becomes L1-tiled contiguous
  SIMD loops), and dead-operator elimination all run at this level;
  `find_broken_transform` bisects the pass list; `config.skip_transforms`
  switches passes off.
- **General operators:** `reduce`, `scan` and `filter` take several arrays
  with `map=` (and `emit=`); `np.einsum`; `@` on stacks; reductions over
  several axes in one pass; tiling for maps of any rank.
- **Parallel primitives:** prefix scans (`cumsum`), filters (`x[mask]`,
  `nonzero`) and `argmin`/`argmax` run in parallel, on CPUs and GPUs.
- **GPU:** arrays stay on the device between the loops of one call; a loop is
  sent to the device by the work in it, not its element count.
- **BLAS:** float matrix products call the BLAS NumPy uses.
- **Tiling:** two-dimensional maps of reductions (`@`, `allpairs` distances, ...)
  run a cache-sized tile at a time with their reductions jammed 2×4. A
  1024×1024 matrix product is 2× faster on one core and 3× on all of them.
- `np.sort` and `np.argsort` take `axis` and sort every lane in parallel
  (N-d input used to be flattened without saying so).
- The dimensions of an array argument can be compile-time constants when the
  code needs them to be (`range(w.shape[0])` in a loop that must be unrolled).
- `parakeet.macro` and `parakeet.find_broken_transform` are back, and the code
  is laid out in `frontend/`, `syntax/`, `transforms/` and `lib/` again.
- Faster compiles for large functions: identical loop bodies are compiled once.
- **Checked on real NVIDIA hardware.** The whole suite passes with every
  parallel loop forced onto a Tesla T4 (`tools/colab_job.py`, run on demand
  with `runplz colab`; not part of CI).
- **Reads like Parakeet again:** two-space indentation, `parakeet.prims` and
  `parakeet.ndtypes`, and the 0.x helpers `typed_repr`, `specialize`,
  `run_python_fn`, `openmp_available`, `rank`/`size`/`shape`, plus the old
  `config.opt_fusion` / `config.print_*` option names.
- Fixed: two arguments of one GPU kernel that view the same memory now share a
  device buffer (previously one of them could read a truncated copy).

- **Python 3.11+ and NumPy 2**, no dependencies besides NumPy and a C compiler
  (the `dsltools`, `appdirs`, `llvmpy` and PyCUDA dependencies are gone).
- **New frontend.** Functions are elaborated directly into a typed loop IR:
  calls are inlined, closures/tuples/slices are resolved at compile time, and
  array expressions are fused into their consumers.
- **Backends:** sequential C; multi-core through a built-in thread pool, OpenMP
  or Grand Central Dispatch; **Apple GPUs through Metal**; **NVIDIA GPUs through
  CUDA** (CuPy/NVRTC, no `nvcc` needed).  GPU backends fall back to the CPU per
  loop, with a warning that says why.
- **NumPy-exact typing** (NEP 50): result dtypes match NumPy, including the
  treatment of Python scalars.
- **Safety:** bounds checks, shape checks, division and domain checks raise the
  same exceptions NumPy/Python would; compile errors carry the source location
  and the chain of inlined calls; reference-counted temporaries are freed
  eagerly (no more leaks in loops).
- **Much larger language subset:** early `return`, `break`/`continue`, `while`,
  keyword/default/`*args` arguments, walrus, `assert`/`raise`, boolean-mask and
  integer-array indexing (reads and writes), in-place operators with NumPy
  aliasing semantics, 0-d and empty arrays, all integer/float dtypes.
- **Much larger library:** ~200 NumPy functions and methods, `np.random`
  (a counter-based generator that is reproducible in parallel and on GPUs),
  sorting, `math`, and most numeric builtins.
- Globals and closure variables are read at call time (no stale constants);
  arguments that must be compile-time constants (such as `axis`) are
  specialised on demand.
- Compiled modules are cached on disk and shared between processes.
- `parakeet.config`, `parakeet.options(...)`, `PARAKEET_*` environment
  variables, `f.specialize(...).ir/.source/.kernels`, `available_backends()`.

### Removed

- Python 2 support, the LLVM backend, the `interp` backend (use
  `backend="python"`), `typed_macro`/`staged_macro`, and most of the old
  `config.opt_*` flags.

---

# Releases before the rewrite

## 0.24 / March 24th, 2014 ###
- cache compiled code by hash of generated C source
- tuple slicing
- fixed (or disabled) unreliable optimizations
- casting between array types using map over elements

### 0.23 / November 21st, 2013 ###

- Generalized StrideSpecialization to ValueSpecialization 
- Significantly decreased overhead of calling into Parakeet (though still ~500x slower than a normal Python call)

### 0.22 / November 20th, 2013 ###

- Changed NumPy calls to use newer API, cuts down on number of compile warnings
- If compilation fails, retry using distutils, may help some people on Windows

### 0.21 / November 19th, 2013 ###

- Got rid of testing dependency on SciPy
- Deleted unused and unfinished optimizations
- Small misc. bugs 

### 0.20 / November 19th, 2013 ###

The last release added experimental CUDA support but the performance was terrible. This release includes lots of tweaks and optimizations necessary for getting beneficial speedups on the GPU. However, the default backend remains OpenMP since some program constructs will not work on the GPU and the nvcc compile times are unacceptably slow.

- Expanded and generalized fusion optimization
- Filled in missing methods from shape inference
- Using ShapeElimination on every function (repurposes the shape inference results as a symbolic execution optimization)
- Fixed lots of small bugs in other optimizations exposed by ShapeElimination
- Shaved off small amount of compile time by moving away from Node pseudo-ASTs to regular Python constructors
- Hackishly added int24 just as a sentinel for default values in reductions that need to cast up to int32 from bool, int8, int16.
- Eliminate redundant & constant array operator arguments with SpecializeFnArgs

### 0.19 / November 4th, 2013 ###

- Added experimental CUDA backend (use by passing _backend='cuda' to functions wrapped by @jit)

### 0.18 / October 30th, 2013 ###

- Added OpenMP backend (runs most map-like computations across multiple threads)
- Stack-allocate representations for all structured types in C
- Disabled Flattening -- tricky transform needs careful audit
- Debugged and enabled CopyElimination
- Fixed negative step in slices 
- Added RLock around AST translation to play nice with Python threads (thanks Russell Power)
- Fixed link argument order for building on cygwin in Windows (thanks Yves-Rémi Van Eycke)

### 0.17 / October 9th, 2013 ###

- Added support for binding multiple variables in a for loop (i.e. "for (x,(y,z)) in enumerate(zip(ys,zs)):")
- More array constructors support 'dtype' argument 
- macros can now generate wrapper functions with keyword arguments 
- lib_helpers._get_type can pull element types out of a wider range of values 
- Added more unit tests from Python benchmarks 
- Added option to manually specify compiler in c_backend.config.compiler_path
- Slightly better support for negative indexing but negative step sizes are still mostly broken 
 
### 0.16.2 / October 1st, 2013 ###

- Moved version info into submodule so setup.py can run without full dependencies (thanks rjpower). 
- Fixed support for references to global arrays.
- Make C backend respect runtime changes to config flags. 
- Got rid of unncessary linking against libpython. 

### 0.16.1 / September 30th, 2013 ###

- Fixed bugs in C backend and a several optimizatons.
- Only print stderr if C backend fails to compile.  

### 0.16 / September 27th, 2013 ###

- Added flattening transformation which gets rid of all structures except scalars and pointers.
- Created new C backend which compiles with either gcc or clang.
- Disabled some of the more powerful optimizations (i.e. scalar replacement) which might introduce bugs. 
- Revamped the optimization pipeline to avoid running duplicate transformations and allow for better logging. 
- "Indexify" all adverbs by turning them into ParFor/IndexReduce/IndexScan. 

