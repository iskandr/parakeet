# Configuration

Settings live on `parakeet.config`. Change them process-wide via assignment,
temporarily with `with parakeet.options(...):`, per function with `@par.jit(...)`,
or before import via `PARAKEET_*` environment variables. Misspelled attribute
names raise `AttributeError` listing valid settings.

| Setting | Default | Environment variable | Meaning |
|---|---|---|---|
| `backend` | `"auto"` | `PARAKEET_BACKEND` | Execution backend (`"auto"`, `"c"`, `"threads"`, `"openmp"`, `"gcd"`, `"metal"`, `"cuda"`, `"gpu"`, `"python"`; see [Backends](backends.md)) |
| `bounds_check` | `True` | `PARAKEET_BOUNDS_CHECK` | Raise `IndexError` on out-of-bounds array indexing |
| `fusion` | `True` | `PARAKEET_FUSION` | Fuse array expressions into single loops |
| `tiling` | `True` | `PARAKEET_TILING` | Tile matmul-shaped loops for cache/SRAM locality (see [Performance](performance.md#tiling)) |
| `tile_bytes` | half of L2 | `PARAKEET_TILE_BYTES` | Target operand bytes per CPU tile |
| `shape_specialization` | `True` | `PARAKEET_SHAPE_SPECIALIZATION` | Specialize tiled/GPU schedules on finite power-of-2 shape buckets and divisibility |
| `skip_transforms` | `()` | `PARAKEET_SKIP_TRANSFORMS` | Middle-end passes to omit (comma-separated in env; see `parakeet.transforms.pipeline.names()`: `"dead code"`, `"normalize"`, `"vertical fusion"`, `"filter fusion"`, `"sibling fusion"`, `"contraction reassociation"`, `"monoidal tiling"`, `"row pipeline fusion"`, `"chunkwise scan"`, `"pairwise fold tiling"`, `"reduction interchange"`, `"tiling"`) |
| `blas` | `True` | `PARAKEET_BLAS` | Call NumPy's BLAS for standalone CPU float `a @ b` (on Windows, `PARAKEET_BLAS=force` opts in) |
| `num_threads` | all cores | `PARAKEET_NUM_THREADS` | Worker threads for multi-core CPU backends |
| `parallel_grain` | `8192` | `PARAKEET_PARALLEL_GRAIN` | Minimum cheap iterations per chunk before spawning threads |
| `device_threshold` | `32768` | `PARAKEET_DEVICE_THRESHOLD` | Minimum work (`elements × inner_iters`) to dispatch a loop to GPU (also requires `≥ 1024` elements) |
| `device_reduce_chunks` | `8192` | – | Number of partial results for GPU reductions |
| `device_fallback` | `"warn"` | `PARAKEET_DEVICE_FALLBACK` | Action when a loop cannot run on GPU: `"warn"`, `"silent"`, or `"error"` |
| `tensor_cores` | `False` | `PARAKEET_TENSOR_CORES` | Enable TF32 Tensor Cores (inline PTX `mma.sync`) on CUDA `sm_80+` (Metal FP32 `simdgroup_matrix` is exact IEEE FP32 and always enabled) |
| `cuda_compiler` | `"nvrtc"` | `PARAKEET_CUDA_COMPILER` | CUDA kernel compiler: `"nvrtc"` (default, zero-header NVRTC + CUBIN cache), `"ptx"` (direct PTX 7.8 + NVRTC fallback), or `"nvcc"` (`nvcc --cubin -O3`) |
| `cc` | auto | `PARAKEET_CC`, `CC` | C compiler executable |
| `cflags` | `()` | – | Extra C compiler flags |
| `opt_level` | `3` | – | C compiler optimization level (`-O3`) |
| `fast_math` | `False` | – | Enable `-ffast-math` (may alter floating-point rounding) |
| `cache_dir` | per-user cache | `PARAKEET_CACHE_DIR` | Directory for cached compiled modules |
| `print_operators` | `False` | `PARAKEET_PRINT_OPERATORS` | Print operator IR before middle-end passes (0.x alias: `print_indexified_function`) |
| `print_ir` | `False` | `PARAKEET_PRINT_IR` | Print lowered loop IR as functions compile |
| `print_code` | `False` | `PARAKEET_PRINT_CODE` | Print generated C source as functions compile |
| `verify` | `True` | `PARAKEET_VERIFY` | Run `parakeet.verify` before passes, after each pass, and after lowering (0.x alias: `opt_verify`) |

`PARAKEET_CUDA_SIMULATE=1` routes `backend="cuda"` through the CPU simulator
(see [Backends](backends.md#nvidia-gpus-cuda)).

Default cache locations: `~/Library/Caches/parakeet` (macOS),
`$XDG_CACHE_HOME/parakeet` or `~/.cache/parakeet` (Linux),
`%LOCALAPPDATA%\parakeet\Cache` (Windows).
