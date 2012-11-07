from run_function import run, specialize_and_compile
from adverb_api import each, seq_reduce, seq_scan, allpairs, par_each 
from prelude import *


def typed_repr(fn, args):
  _, typed, _, _ = specialize_and_compile(fn, args)
  return typed 