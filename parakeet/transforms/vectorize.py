from ..ndtypes import Int32, Int64, Float32, Float64

from loop_transform import LoopTransform

class Vectorize(LoopTransform):
  
  def pre_apply(self, _):
    # skip the may-alias analysis from LoopTransform
    pass 
  
  vector_elt_types = (Int32, Int64, Float32, Float64) 
  def vectorize_loop(self, stmt):
     
    candidate_names = set([])
    before_values = {}
    after_values = {}
    for (k, (before, after)) in stmt.merge.iteritems():
      t = before.type
      if t in self.vector_elt_types:
        candidate_names.add(k)
        before_values[k] = before
        after_values[k] = after
      
  
  def transform_ForLoop(self, stmt):
    # for now we only vectorize if 
    # something is being accumulated, 
    # otherwise you need more sophisticated
    # cost-base heuristic to avoid lots of 
    # redundant SIMD packing and unpacking
    if self.is_simple_block(stmt.body) and len(stmt.merge) > 1:
      return self.vectorize_loop(stmt)
    else:
      return stmt 
      