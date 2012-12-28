import core_types 
import array_type 
import tuple_type 
import closure_type 
from shape import Shape, Tuple, Closure, Var, Slice 


def from_types(types):
  return Converter().from_types(types)

class Converter(object):
  """
  Turn a list of input types into a list of abstract values, 
  numbering the input arrays and scalars but preserving the 
  structure of tuples, closures, and slices
  """
  def __init__(self):
    self.counter = 0
  
  def fresh_var(self):
    n = self.counter 
    self.counter += 1
    return Var(n)
    
  def from_type(self, t):
    if isinstance(t, core_types.ScalarT):
      return self.fresh_var()
    elif isinstance(t, array_type.ArrayT):
      dim_vars = [self.fresh_var() for _ in range(t.rank)]
      return Shape(dim_vars) 
    elif isinstance(t, tuple_type.TupleT):
      elt_values = self.from_types(t.elt_types)
      return Tuple(elt_values) 
    elif isinstance(t, array_type.SliceT):
      start = self.from_type(t.start_type)
      stop = self.from_type(t.stop_type)
      step = self.from_type(t.step_type)
      return Slice(start, stop, step)
    elif isinstance(t, closure_type.ClosureT):
      arg_vals = self.from_types(t.arg_types)
      return Closure(t.fn, arg_vals) 
    else:
      assert False, "Unsupported type: %s" % t
    
  def from_types(self, arg_types):
    values = []
    for t in arg_types:
      v = self.from_type(t)
      values.append(v)
    return values
