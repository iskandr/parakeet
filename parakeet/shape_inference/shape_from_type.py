from ..ndtypes import (ArrayT, SliceT, ClosureT, NoneT, ScalarT, StructT, 
                       TypeValueT, TupleT, PtrT, FnT)
from shape import Shape, Tuple, Closure, Var, Slice, Struct, Ptr, any_scalar




def shapes_from_types(types):
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
    if isinstance(t, ScalarT):
      return self.fresh_var()
    
    elif t.__class__ is ArrayT:
      dim_vars = [self.fresh_var() for _ in range(t.rank)]
      return Shape(dim_vars)
    
    elif t.__class__ is TupleT:
      elt_values = self.from_types(t.elt_types)
      return Tuple(elt_values)
    
    elif t.__class__ is SliceT:
      start = self.from_type(t.start_type)
      stop = self.from_type(t.stop_type)
      step = self.from_type(t.step_type)
      return Slice(start, stop, step)
    
    elif t.__class__ is ClosureT:
      arg_vals = self.from_types(t.arg_types)
      return Closure(t.fn, arg_vals)
   
    elif t.__class__ is FnT:
      return Closure(t.fn, ())
    
    elif isinstance(t, StructT):
      field_names = [fn for (fn,_) in t._fields_]
      field_types = [ft for (_,ft) in t._fields_]
      field_vals = self.from_types(field_types)
      return Struct(field_names, field_vals)
    
    elif isinstance(t, (TypeValueT, NoneT)):
      return Tuple(())
    
    elif isinstance(t, PtrT): 
      return Ptr(any_scalar) 
    
    else:
      assert False, "Unsupported type: %s" % t
    
  def from_types(self, arg_types):
    values = []
    for t in arg_types:
      v = self.from_type(t)
      values.append(v)
    return values
