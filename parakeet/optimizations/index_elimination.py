from core_types import Int32, Int64, Float32, Float64, ScalarT
from syntax import Var 
from transform import Transform 
from syntax_visitor import SyntaxVisitor  
from node import Node 
from syntax import Range, Const, ConstArray, ConstArrayLike, Array, Index

import syntax
import collect_vars
class AbstractValue(Node):
  _members = []

class ConstValue(AbstractValue):
  """
  All elements are the same constant 
  """
  _members = ['value', 'type']

zeros_float32 = ConstValue(0.0, Float32)
zeros_float64 = ConstValue(0.0, Float64)
zeros_int32 = ConstValue(0, Int32)
zeros_int64 = ConstValue(0, Int64)

class ConstElts(AbstractValue):
  """
  All elements known, represented as a numpy array 
  """ 
  _members = ['array']

class RangeArray(AbstractValue):
  """
  Result of a range expression 
  """
  _members = ['start', 'step', 'type']

class Unknown(AbstractValue):
  _members = []
  
unknown = Unknown()

class IndexElimAnalysis(SyntaxVisitor):
    def __init__(self):
      # map variables to (start, stop, step)  
      self.index_vars = {}
      
      # map array names to abstract values 
      self.array_values = {}
      
      # map each array name to a set of names which share the same 
      # memory 
      self.linked_arrays = {}
      
    def visit_Assign(self, stmt):
      lhs = stmt.lhs
      rhs = stmt.rhs 
      
      if lhs.__class__ is Var:
        lhs_name = stmt.lhs.name
        
        rhs_class = rhs.__class__ 
        if rhs_class is Var:
          linked_set = self.linked_arrays.get(lhs_name, set([lhs_name]))
          rhs_name = stmt.rhs.name
          linked_set.add(rhs_name)
          self.linked_arrays[lhs_name] = linked_set 
          self.linked_arrays[rhs_name] = linked_set
          if lhs_name in self.array_values:
            self.array_values[rhs_name] = self.array_values[lhs_name]
        elif rhs_class is Range:
          v = RangeArray(rhs.start, rhs.step, rhs.type.elt_type)
          self.array_values[lhs_name] = v 
      else:
        # if not a var, might be an index expression 
        for tainted_lhs_name in collect_vars.collect_var_names(lhs):
          for linked_name in self.linked_arrays.get(tainted_lhs_name, set([tainted_lhs_name])):
            self.array_values[linked_name] = unknown 
        
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
      return syntax.Const(v.value, v.type)
    elif v.__class__ is ConstElts  and all(idx.__class__ is syntax.Const for idx in indices):
      if n_indices != v.type.rank: return expr
      idx_values = [idx.value for idx in indices]
      return syntax.Const(v.array[tuple(idx_values)], type = v.type.elt_type)
    
    assert v.__class__ is RangeArray
    assert len(indices) == 1
    idx = indices[0]
    if idx.__class__ is syntax.Const and v.step.__class__ is Const and v.start.__class__ is Const:
      return syntax.Const(idx.value * v.step.value + v.start.value, type = v.type) 

    return self.add(v.start, self.mul(idx, v.step))
      