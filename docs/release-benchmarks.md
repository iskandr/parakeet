# Release benchmarks

Five forward workloads compare Parakeet with general-purpose array compilers and specialist libraries. The tables report warmed median latency. Smaller values are faster. [Download all timing samples and validation results (JSON)](assets/modal-alpha-comparison.json).

![CPU exact nearest-neighbor timings for three implementations. Faiss Flat leads at the small shape, while Parakeet with broadcasting NumPy leads at the large shape. Each panel has its own time scale.](assets/cpu-nearest-neighbor.png)

Four CPU threads, exact 1-NN, float32. Panels show queries × references × features, with independent time scales. At the small shape Faiss took **0.033 ms**. At the large shape Parakeet's NumPy function took **6.593 ms** and its nested-map DSL took **6.132 ms**.

On the NVIDIA A10, Faiss led large exact search and compiled PyTorch led Gaussian regression. Parakeet beat PyKeOps for both at the large shape. All GPU comparisons ran on the same physical card.

## Pairwise workloads

Warmed median latency in milliseconds. Small is 256 × 1,024 × 3 and large is 1,024 × 4,096 × 32, in query × reference × feature order. Gaussian regression returns four normalized weighted sums per query.

### Four-vCPU worker

| Four-vCPU implementation | Small 1-NN | Large 1-NN | Small Gaussian | Large Gaussian |
|---|---:|---:|---:|---:|
| Parakeet, broadcasting NumPy | 0.664 | 6.593 | 2.971 | 28.951 |
| Parakeet, nested-map DSL | 0.638 | 6.132 | — | — |
| PyKeOps Genred | 0.255 | 18.246 | 1.087 | 38.038 |
| Faiss IndexFlatL2 | 0.033 | 9.577 | — | — |
| SciPy cKDTree, exact query | 1.441 | 51.846 | — | — |
| scikit-learn brute-force neighbors | 1.042 | 6.647 | — | — |
| NumPy broadcasting | 9.899 | 326.690 | 10.915 | 359.939 |

### NVIDIA A10

| NVIDIA A10 implementation | Small 1-NN | Large 1-NN | Small Gaussian | Large Gaussian |
|---|---:|---:|---:|---:|
| Parakeet, broadcasting NumPy | 0.261 | 0.796 | 0.226 | 1.130 |
| PyKeOps Genred | 0.262 | 0.962 | 0.295 | 1.289 |
| Faiss GpuIndexFlatL2 | 0.079 | 0.154 | — | — |
| PyTorch cdist | 0.128 | 0.323 | 0.166 | 0.568 |
| torch.compile over cdist | 0.089 | 0.208 | 0.106 | 0.294 |
| torch.compile over broadcasting | 0.053 | 0.416 | 0.074 | 0.487 |
| JAX jit, broadcasting | 0.115 | 0.424 | 0.098 | 0.663 |
| CuPy, broadcasting | 0.483 | 12.359 | 0.527 | 12.736 |

### Memory and dispatch

The CPU IR allocates a 16 × 128 float32 score tile (8 KiB) and 192 bytes of row summaries per active query strip. The CUDA broadcast form allocates a float32 query-by-reference matrix (16 MiB at the large shape) and runs two kernels. The nested DSL form allocates one reference-length row inside the query loop. Strict CUDA execution rejected it at both sizes. On Metal we also verified the fallback diagnostic for the DSL and two launches for broadcasting. These are IR allocations and dispatch counts. Peak process memory was not measured.

## Attention, linear cross-entropy and split finding

All three workloads run forward on the A10. Attention has one batch, four heads, 512 tokens and head width 64. Cross-entropy includes projecting 512 vectors of width 128 to 2,048 logits and taking the mean target loss. Split finding includes two weighted histograms over 262,144 rows and 256 bins, prefix sums and the best-bin and gain return.

| NVIDIA A10 implementation | Causal attention | Linear cross-entropy | GBDT split |
|---|---:|---:|---:|
| Parakeet | 0.386 | 1.604 | 0.276 |
| PyTorch SDPA / cross_entropy / scatter_add | 0.090 | 0.069 | 0.225 |
| torch.compile over the same PyTorch functions | 0.116 | 0.088 | 0.175 |
| JAX jit | 0.217 | 0.133 | 0.188 |
| CuPy | 0.326 | 0.255 | 0.420 |

Parakeet issued one launch and zero transfers for attention and Gaussian regression. Linear cross-entropy issued two launches and one download per call. Split finding issued six launches, two uploads and six downloads. Two clear follow-ups are a GPU schedule for ordered pairwise folds and keeping the histogram-to-split computation resident. The cross-entropy result calls for a measured choice between materializing logits and recomputing them at this shape.

## How to reproduce and interpret the numbers

Run from a checkout with the Modal client configured.

```bash
modal run tools/modal_release_bench.py --output comparison.json
```

- The measured compiler revision was `0345559f4b2d3c59a07ce8c1a71bfb619f690063`. The subsequent NumPy scalar compatibility fix concerns `count_nonzero`, which none of these workloads calls. The GPU reported 23,028 MiB and driver 620.70.06. The CPU reservation was four vCPUs with four library threads, and Modal hid the CPU model.
- Inputs were generated once per provider from the same fixed seed, using float32 values and integer indices. The oracle used float64 NumPy, exact nearest-neighbor and split indices, and tolerances of `rtol=5e-4, atol=3e-5` for floating outputs. SciPy's tree uses double precision internally. All published timings passed their checks.
- GPU inputs were uploaded before timing. The comparison includes dispatch, output production, internal transfers and synchronization for every call. Faiss database insertion, tree construction, sklearn fitting, imports and initial input upload were outside query timing. This measures repeated queries against a prepared reference set.
- Each method ran once for first-call timing and validation, then three warmups, then 21 synchronized GPU samples or 11 CPU samples. Providers ran sequentially in separate processes on the same GPU. JAX used its own Python environment. PyTorch used highest float32 matmul precision with TF32 disabled, JAX used highest precision, and Parakeet used strict device fallback, a threshold of one and tensor cores disabled. PyTorch SDPA selected its own float32 backend.
- First-call time includes execution and any lazy compilation. The large shape follows the small shape, so first calls can reuse specializations. For causal attention, Parakeet's first call took 1.692 s and its warmed median was 0.386 ms. The raw file includes first-call times, every sample, versions, validation error and the two strict-dispatch rejections. [Download the measurements](assets/modal-alpha-comparison.json).
