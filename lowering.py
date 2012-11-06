from transform import apply_pipeline
from lower_adverbs import LowerAdverbs
from lower_structs import LowerStructs
from lower_indexing import LowerIndexing
from simplify import Simplify
import optimize 

pipeline = [
  LowerAdverbs, LowerIndexing, 
  Simplify, 
  LowerStructs, 
  Simplify,
]

def lower(fundef):
  fundef = optimize.optimize(fundef)
  print "BEFORE LOWERING", fundef
  fundef2 = apply_pipeline(fundef, pipeline)
  print "AFTER LOWERING", fundef2
  return fundef2 