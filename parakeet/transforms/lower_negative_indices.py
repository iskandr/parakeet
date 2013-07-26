from ..ndtypes import ArrayT, TupleT, SliceT 
from ..syntax import Tuple, Slice, Var 
from range_transform import RangeTransform

class LowerNegativeIndices(RangeTransform):
  
  """
  This doesn't entirely obviate the need for runtime checks on indexing 
  but does handle the simple case where a negative index can be inferred from the source 
  """
  
  def pre_apply(self, fn):
    RangeTransform.pre_apply(self, fn)
    #self.known_tuples = {}
    #self.known_slices = {}
   
  def get_tuple_elts(self, expr):
    if expr.__class__ is Tuple:
      # get the range of each elt expr 
      return [self.get(elt) for elt in expr.elts]
    
    elif expr.__class__ is Var and expr.name in self.known_tuples:
      return self.known_tuples[expr.name]
    
    else:
      return [None for _ in expr.type.elt_types]

  def get_slice_bounds(self, expr):
    if expr.__class__ is Slice:
      return (self.get(expr.start), self.get(expr.stop))
    
    elif expr.__class__ is Var and expr.name in self.known_slices:
      return self.known_slices[expr.name]
    else:
      return None
  
  
  
  def transform_Index(self, expr):
    arr = self.transform_expr(expr.value)
    idx = self.transform_expr(expr.index)
    arr_t = arr.type 
    assert arr_t.__class__ is ArrayT, "Expected array, got %s : %s" % (arr, arr.type)
    if idx.type.__class__ is TupleT:
      index_ranges = self.get_tuple_elts(idx)
    else:
      index_ranges = [self.get(idx)]
    
    #for i, idx in enumerate(indices):
    #  if idx.type.__class__ is SliceT:
  
  def transform_Assign(self, stmt):
    if stmt.lhs.__class__ is Var:
      if stmt.rhs.__class__ is Tuple:
        