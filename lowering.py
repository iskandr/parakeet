from transform import apply_pipeline
from lower_adverbs import LowerAdverbs
from lower_structs import LowerStructs
from lower_indexing import LowerIndexing
from simplify import Simplify
from constant_propagation import ConstantPropagation 
from inline import Inliner
pipeline = [
  Inliner,
  LowerAdverbs, LowerIndexing, 
  ConstantPropagation, Simplify, 
  LowerStructs, 
]

def lower(fundef):
  
  print "BEFORE LOWERING", fundef
  fundef2 = apply_pipeline(fundef, pipeline)
  print "AFTER LOWERING", fundef2
  return fundef2 