import numpy as np


import config
import type_conv_decls
from decorators import jit, macro
from lib import *
from run_function import run, specialize_and_compile
import syntax 

def typed_repr(fn, args):
  _, typed, _, _ = specialize_and_compile(fn, args)
  return typed

def clear_specializations():
  import closure_type
  for clos_t in closure_type._closure_type_cache.itervalues():
    clos_t.specializations.clear()

