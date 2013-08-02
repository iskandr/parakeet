from .. import names 
from ..builder import build_fn
from ..ndtypes import ScalarT, NoneT, ArrayT, SliceT, TupleT, Int64, PtrT, ptr_type
from transform import Transform


def flatten_type(t):
  """
  Turn a structured type into a list of primitive types 
  """
  if isinstance(t, (ScalarT, NoneT)):
    return (t,)
  elif isinstance(t, TupleT):
    return flatten_types(t.elt_types)
  elif isinstance(t, SliceT):
    return (t.start_type, t.stop_type, t.step_type)
  elif isinstance(t, ArrayT):
    rank = Int64 
    offset = Int64
    shape_elts = flatten_type(t.shapes_t)
    stride_elts = flatten_type(t.strides_t)
    return (rank,) + flatten_type(t.ptr_t) + (offset,) + shape_elts + stride_elts    
  elif isinstance(t, PtrT):
    if isinstance(t.elt_type, ScalarT):
      return (t,)
    else:
      # split ptr(Tuple(int,float)) into ptr(int), ptr(float) 
      return tuple(ptr_type(elt_t) for elt_t in flatten_type(t.elt_type)) 
  else:
    assert False, "Unsupported type %s" % (t,)

def flatten_types(ts):
  result = []
  for t in ts:
    if isinstance(t, (ScalarT, NoneT)):
      result.append(t)
    else:
      result.extend(flatten_type(t))
  return tuple(result) 



# map from a type signature to a helper which pulls apart all the inputs and returns
# their scalar/ptr elements     
_unwrappers = {}

def build_unwrapper(fn):
  input_types = fn.input_types 
  return_type = fn.return_type
  base_name = names.original(fn.name)
  key = (base_name, tuple(input_types), return_type)
  if key in _unwrappers:
    return _unwrappers[key]
  flattened_input_types = flatten_types(input_types)
  unwrap_name = names.fresh("unwrap_" + base_name)
  input_names = [names.refresh(name) for name in fn.arg_names]
  f, builder, input_vars  = build_fn(input_types, 
                                     flattened_input_types, 
                                     name = unwrap_name, 
                                     input_names = input_names)
  result_vars = []
  # need to return a mapping which tells us which variable is 
  # tuple_obj[0].tuple_field[1]
  
  
# map from a return type to a function which takes all the elements of a struct and constructs it
_wrappers = {}

class AccessPath(object):
  def __eq__(self, other):
    return False 
  
  def __neq__(self, other):
    return not self == other 

class Attr(AccessPath):
  def __init__(self, base, attr):
    self.base = base
    self.attr = attr
    
  def __eq__(self, other):
    return other.__class__ is Attr and self.base == other.base and self.attr == other.attr  
    
class Idx(AccessPath):
  def __init__(self, base, idx):
    self.base = base 
    self.idx = idx
    
  def __eq__(self, other):
    return other.__class__ is Idx and self.base == other.base and self.idx == other.idx 

class Flatten(Transform):
  """
  Split a function into:
    wrap(flattened_fn(unwrap(args)))
  such that all tuple indexing & attribute access happens in 'unwrap' and
  all tuple/struct construction happens in 'wrap'. Change all internal function calls
  to happen only between flattened versions of functions 
  """
  
  def pre_apply(self, fn):
    self.paths = {}
    
  def flatten_input(self, var, path = None):
    if path is None: path = var.name
    t = var.type
    if isinstance(t, (NoneT, ScalarT)):
      self.paths[path] = var
      return var 
    elif isinstance(t, ArrayT):
      data = self.attr(var, 'data', name = "array_data")
      self.flatten_input(var, path)
      self.paths[Attr(path, 'data')] = data
      
      offset = self.attr(var, 'offset', name = "array_offset")
      self.paths[Attr(path, 'offset')] = offset 
      
      


