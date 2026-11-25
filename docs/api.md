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

## Gradients

::: parakeet.grad

::: parakeet.value_and_grad

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
