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
