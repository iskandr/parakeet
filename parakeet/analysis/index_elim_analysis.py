from dsltools import Node 

from .. ndtypes import Float32, Float64, Int32, Int64
from .. syntax import Var, Range, ConstArray, ConstArrayLike, IndexMap
from collect_vars import collect_var_names 
from syntax_visitor import SyntaxVisitor 


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

class IndexMapResult(AbstractValue):
  _members = ['fn']

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
          self.array_values[lhs_name] = RangeArray(rhs.start, rhs.step, rhs.type.elt_type)
        elif rhs_class in (ConstArray, ConstArrayLike):
          self.array_values[lhs_name] = ConstValue(value = rhs.value, type = rhs.type) 
        elif rhs_class is IndexMap:
          self.array_values[lhs_name] = IndexMapResult(fn = rhs.fn)
      else:
        # if not a var, might be an index expression 
        for tainted_lhs_name in collect_var_names(lhs):
          for linked_name in self.linked_arrays.get(tainted_lhs_name, set([tainted_lhs_name])):
            self.array_values[linked_name] = unknown 