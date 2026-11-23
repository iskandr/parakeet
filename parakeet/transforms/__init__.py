"""Program transformations that are more than a property of the frontend.

The frontend hands over operator statements (``IndexMap``, ``IndexReduce``);
:mod:`.pipeline` runs the passes over them (:mod:`.fusion`, :mod:`.tiling`)
and :mod:`.lowering` turns what is left into loops.  Vertical fusion is not
here: array expressions are fused as they are elaborated (see
:mod:`parakeet.frontend.arrays`).  Reverse-mode differentiation
(:mod:`.autodiff`) also still runs during elaboration.
"""
