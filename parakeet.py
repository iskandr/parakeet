from lib_simple import *
from adverb_api import each, reduce, scan, allpairs, par_each
from lib_adverbs import * 
from run_function import run, specialize_and_compile

def typed_repr(fn, args):
  _, typed, _, _ = specialize_and_compile(fn, args)
  return typed
