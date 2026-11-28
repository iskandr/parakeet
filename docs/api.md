# API reference

## Compiling

::: parakeet.jit
    options:
      members: [specialize, specializations, inspect]

::: parakeet.run

::: parakeet.specialize

::: parakeet.typed_repr

::: parakeet.Specialization
    options:
      members: [operators, ir, source, kernels, signature]

## Automatic differentiation

::: parakeet.grad

::: parakeet.value_and_grad

::: parakeet.jvp

::: parakeet.vjp

::: parakeet.linearize

::: parakeet.linear_transpose

::: parakeet.hvp

::: parakeet.jacfwd

::: parakeet.jacrev

::: parakeet.hessian

::: parakeet.stop_gradient

::: parakeet.checkpoint

::: parakeet.jvp_for

::: parakeet.vjp_for

::: parakeet.custom_vjp

::: parakeet.custom_jvp

## Backends, devices, and settings

::: parakeet.available_backends

::: parakeet.target_profile

::: parakeet.to_device

::: parakeet.to_host

::: parakeet.DeviceArray

::: parakeet.options

::: parakeet.seed

::: parakeet.clear_cache

## Data-parallel operators

All top-level data-parallel operators are documented with signatures and
examples in [Data-parallel operators](operators.md#operator-reference):

[`par.map`](operators.md#map),
[`par.allpairs`](operators.md#allpairs),
[`par.outer_map`](operators.md#outer_map),
[`par.reduce`](operators.md#reduce),
[`par.fold`](operators.md#fold),
[`par.scan`](operators.md#scan),
[`par.imap`](operators.md#imap),
[`par.ireduce`](operators.md#ireduce),
[`par.parfor`](operators.md#parfor),
[`par.filter`](operators.md#filter),
[`par.filter_reduce`](operators.md#filter_reduce),
[`par.flatmap`](operators.md#flatmap),
[`par.filter_pairs`](operators.md#filter_pairs),
[`par.scatter_add`](operators.md#scatter_add),
[`par.wavefront`](operators.md#wavefront),
[`par.batched_wavefront`](operators.md#batched_wavefront),
[`par.soft_align`](operators.md#soft_align).

## Neural-network helpers, optimizers, trees, and sequences

* **Neural network primitives (`par.nn`):** `par.nn.Module`, `Linear`,
  `LayerNorm`, `Embedding`, `MLP`, `CausalSelfAttention`, `TransformerBlock`,
  plus functional `relu`, `gelu`, `silu`, `softmax`, `log_softmax`,
  `layer_norm`, `rms_norm`, `causal_mask`, `scaled_dot_product_attention`, and
  `cross_entropy`.
* **Optimizers (`par.optim`):** `par.optim.sgd_step`, `par.optim.adam_init`,
  `par.optim.adam_step`, and stateful `par.optim.Adam`.
* **Parameter trees:** `par.tree_map`, `par.tree_map_arrays`, `par.tree_leaves`,
  `par.tree_array_leaves`.
* **Categorical sequence & text encoders:** `par.Alphabet` (including
  `par.Alphabet.mask`) and `par.Vocab` (including `par.Vocab.from_text`).

## Structured folds and rewrite inspection

::: parakeet.show_rewrite

::: parakeet.RewriteReport

::: parakeet.fold

## Diagnostics and verification

::: parakeet.find_broken_transform

::: parakeet.verify

## Exceptions

::: parakeet.errors
    options:
      members:
        - ParakeetError
        - CompileError
        - UnsupportedError
        - TypeInferenceError
        - StaticValueError
        - BackendError
        - BackendUnavailable
        - CCompilerError
        - InternalError
        - FallbackWarning
        - SourceLocation

## Testing helpers

::: parakeet.testing
    options:
      members: [check, same, expect, eq]
