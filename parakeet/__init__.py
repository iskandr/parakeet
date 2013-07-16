import config 

from frontend import jit, macro
from lib import *
from run_function import run, specialize_and_compile

def typed_repr(fn, args):
  _, typed, _, _ = specialize_and_compile(fn, args)
  return typed

def clear_specializations():
  import closure_type
  for clos_t in closure_type._closure_type_cache.itervalues():
    clos_t.specializations.clear()

from prims import * 
from lib import * 


from frontend.run_function import run 
from frontend.build_function import build_fn 