# Configuration

Settings live on `parakeet.config`. Assign to change them for the process, use
`parakeet.options(...)` to change them temporarily, pass `backend`,
`bounds_check` or `fusion` to `@jit` for one function, or set the environment
variable before importing Parakeet.

| Setting | Default | Environment variable | Meaning |
|---|---|---|---|
| `backend` | `"auto"` | `PARAKEET_BACKEND` | see [Backends](backends.md) |
| `bounds_check` | `True` | `PARAKEET_BOUNDS_CHECK` | raise `IndexError` for bad indices |
| `fusion` | `True` | `PARAKEET_FUSION` | fuse array expressions into single loops |
| `tiling` | `True` | `PARAKEET_TILING` | tile matmul-shaped loops for cache locality (see [Performance](performance.md#tiling)) |
| `tile_bytes` | half of L2 | `PARAKEET_TILE_BYTES` | bytes of operand data in one tile |
| `blas` | `True` | `PARAKEET_BLAS` | hand float matrix products to the BLAS NumPy uses (CPU backends; not yet on Windows, where `PARAKEET_BLAS=force` opts in) |
| `num_threads` | all cores | `PARAKEET_NUM_THREADS` | threads used by multi-core backends |
| `parallel_grain` | `8192` | `PARAKEET_PARALLEL_GRAIN` | minimum iterations per chunk for cheap loops |
| `device_threshold` | `32768` | `PARAKEET_DEVICE_THRESHOLD` | minimum work (elements × estimated inner iterations each) for a loop to go to a GPU; the loop also needs 1024 elements (or this many, if smaller) |
| `device_reduce_chunks` | `8192` | – | partial results used by GPU reductions |
| `device_fallback` | `"warn"` | `PARAKEET_DEVICE_FALLBACK` | `"warn"`, `"silent"` or `"error"` when a loop cannot run on the GPU |
| `cc` | auto | `PARAKEET_CC`, `CC` | C compiler executable |
| `cflags` | `()` | – | extra compiler flags |
| `opt_level` | `3` | – | compiler optimisation level |
| `fast_math` | `False` | – | allow value-changing floating-point optimisations |
| `cache_dir` | per-user cache | `PARAKEET_CACHE_DIR` | where compiled modules are stored |
| `print_ir` | `False` | `PARAKEET_PRINT_IR` | print the IR of everything compiled |
| `print_code` | `False` | `PARAKEET_PRINT_CODE` | print the generated C |

`PARAKEET_CUDA_SIMULATE=1` makes `backend="cuda"` use the CPU simulator described
in [Backends](backends.md#nvidia-gpus-cuda).

Misspelt setting names raise `AttributeError` listing the valid ones.

The default cache directory is `~/Library/Caches/parakeet` on macOS,
`$XDG_CACHE_HOME/parakeet` (or `~/.cache/parakeet`) on Linux and
`%LOCALAPPDATA%\parakeet\Cache` on Windows.
