---
title: Parakeet
description: Compile numerical Python and NumPy into fused native code for multi-core CPUs, Apple Metal, and NVIDIA CUDA.
hide:
  - navigation
  - toc
---

<div class="pk-intro" markdown>
<div class="pk-identity" markdown>

# ![Parakeet](assets/logo.png){ .pk-wordmark width="1746" height="520" }

<p class="pk-lead">A data-parallel compiler for numerical Python.</p>

Write the calculation with NumPy or compose maps, folds and scans directly.
Parakeet specializes the function for its inputs and fuses array operations
before generating native code for CPUs, Apple Metal or NVIDIA CUDA.

[Get started](quickstart.md){ .md-button .md-button--primary }
[Explore the gallery](gallery.md){ .md-button }

</div>
<div class="pk-example" markdown>

## Exact nearest neighbor

```python
import numpy as np
from parakeet import jit

@jit
def nearest(x, y):
    delta = x[:, None, :] - y[None, :, :]
    d2 = np.sum(delta ** np.float32(2), axis=2)
    return np.argmin(d2, axis=1)
```

On CPU, Parakeet computes distances a tile at a time and keeps the winning
reference index for each query. Ties return the first index.

[Compare timings and allocations](release-benchmarks.md#pairwise-workloads)

</div>
</div>

## Install Parakeet 1.0.0

```bash
python -m pip install "parakeet==1.0.0"
```

Use Python 3.11 or newer, NumPy 2 and a C compiler. The getting started guide
covers [your first compiled function](quickstart.md#your-first-compiled-function)
and [choosing a backend](quickstart.md#picking-a-backend).

## Inside the compiler

- **Fusion.** Two histogram expressions with the same bin IDs can accumulate
  gradient and Hessian together in one traversal. [See the tree-split example](gallery.md#histograms-and-tree-splits).
- **Tiling.** A normalized weighted sum can consume its scores in blocks.
  Gaussian regression and attention share that reduction. [See the algebraic laws](algebra.md).
- **Differentiation.** Differentiate a scalar loss over arrays with
  `grad` or `value_and_grad`. [See the differentiation guide](autodiff.md).
- **Inspection.** Examine the operator IR, the rewrites that fired and the
  generated code. [Inspect a rewrite](gallery.md#inspect-a-rewrite).

Parakeet began as a research compiler at New York University. The 1.0 rewrite
retains its algebra of array operators and extends it to modern Python and GPUs.
[History, papers and language boundaries](philosophy.md).
