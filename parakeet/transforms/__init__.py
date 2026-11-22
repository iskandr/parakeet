"""Program transformations that are more than a property of the frontend.

Fusion is not here: array expressions are fused as they are elaborated (see
:mod:`parakeet.frontend.arrays`).  What is here rewrites the loops or the
program itself: tiling for cache locality and reverse-mode differentiation.
"""
