
from ..config import set_opt_level
from closure_specializations import print_specializations
# from decorators import jit, macro, staged_macro 

import type_conv_decls as _decls 
from decorators import jit, macro, staged_macro
from ast_conversion import translate_function_value, translate_function_ast

from run_function import run_untyped_fn, run_typed_fn, run_python_fn, specialize
from diagnose import find_broken_transform 

def typed_repr(python_fn, args, optimize = True):
  typed_fn, _ = specialize(python_fn, args)
  if optimize:
    from ..transforms import pipeline
    return pipeline.high_level_optimizations.apply(typed_fn)
  else:
    return typed_fn 
