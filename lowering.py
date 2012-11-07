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

def lower(fundef):
  
  print "BEFORE LOWERING", fundef
  fundef2 = apply_pipeline(fundef, pipeline, copy = True)
  print "AFTER LOWERING", fundef2
  return fundef2 