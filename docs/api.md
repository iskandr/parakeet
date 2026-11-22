# API reference

## Compiling

::: parakeet.jit
    options:
      members: [specialize, specializations, inspect]

::: parakeet.run

::: parakeet.Specialization
    options:
      members: [ir, source, kernels, signature]

## Backends and settings

::: parakeet.available_backends

::: parakeet.options

::: parakeet.seed

::: parakeet.clear_cache

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
