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
