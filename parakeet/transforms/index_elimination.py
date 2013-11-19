

from .. analysis.index_elim_analysis import IndexElimAnalysis
from .. analysis.index_elim_analysis import \
  (unknown, ConstValue, ConstElts, RangeArray, IndexMapResult)
from .. ndtypes import ScalarT, TupleT 
from .. syntax import Const, Var 
from transform import Transform   
 

        
class IndexElim(Transform):
  def pre_apply(self, fn):
    analysis = IndexElimAnalysis() 
    analysis.visit_fn(fn)
    self.array_values = analysis.array_values
     
  def transform_Index(self, expr):
    if expr.value.__class__ is not Var: return expr 
    x = expr.value.name 
    if x not in self.array_values: return expr
    v = self.array_values[x]
    if v is unknown: return expr 
    if self.is_tuple(expr.index):
      indices = self.tuple_elts(expr.index)
    else:
      indices = [expr.index]
    if not all(isinstance(idx.type, ScalarT) for idx in indices): return expr
     
    n_indices = len(indices)
    if v.__class__ is ConstValue:
      if n_indices != v.type.rank: return expr
      return Const(v.value, v.type)
    elif v.__class__ is ConstElts  and all(idx.__class__ is Const for idx in indices):
      if n_indices != v.type.rank: return expr
      idx_values = [idx.value for idx in indices]
      return Const(v.array[tuple(idx_values)], type = v.type.elt_type)
    elif v.__class__ is IndexMapResult: 
      fn = self.get_fn(v.fn)
      closure_args = self.get_closure_args(v.fn)
      input_types = fn.input_types
      last_input_type =  input_types[-1]
      if last_input_type.__class__ is TupleT:
        n_expected_indices = len(last_input_type.elt_types)
        indices_are_tuple = True 
      else:
        indices_are_tuple = False
        n_expected_indices = len(input_types) - len(closure_args)
      if n_indices != n_expected_indices:
        return expr 
      if indices_are_tuple:
        index_exprs = [self.tuple(indices)]
      else:
        index_exprs = list(indices)
      return self.call(v.fn, index_exprs)
        
    assert v.__class__ is RangeArray
    assert len(indices) == 1
    idx = indices[0]
    if idx.__class__ is Const and v.step.__class__ is Const and v.start.__class__ is Const:
      return Const(idx.value * v.step.value + v.start.value, type = v.type) 

    return self.add(v.start, self.mul(idx, v.step))
      