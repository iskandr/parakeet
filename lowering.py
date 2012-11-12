from transform import apply_pipeline
from lower_adverbs import LowerAdverbs
from lower_structs import LowerStructs
from lower_indexing import LowerIndexing
from simplify import Simplify 

pipeline = [
  LowerAdverbs, LowerIndexing, 
  Simplify, 
  LowerStructs, 
  Simplify,
]


_lowered_functions = {}
def lower(fundef):
  if fundef.name in _lowered_functions:
    return _lowered_functions[fundef.name]
  else:
    # print "BEFORE LOWERING", fundef
    lowered_fn = apply_pipeline(fundef, pipeline, copy = True)
    # print "AFTER LOWERING", lowered_fn
    _lowered_functions[fundef.name] = lowered_fn
    _lowered_functions[lowered_fn.name] = lowered_fn
    return lowered_fn