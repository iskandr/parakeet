from .. import names 
from ..builder import build_fn
from ..ndtypes import ScalarT, NoneT, ArrayT, SliceT, TupleT, Int64, PtrT, ptr_type
from ..syntax import Var, Struct, Attribute, Tuple, TupleProj, Closure, ClosureElt, Const

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


  
def path_name(path):
  return "_".join(path)


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
    return fn 

  def flat_values(self, v, path):
    if path in self.paths:
      return self.paths[path]
    
    t = v.type 
    if isinstance(t, (NoneT, ScalarT)):
      result = (v,)
      
    elif isinstance(t, ArrayT):
      data_path = path + ("data",)
      data_values = \
        self.flat_values(self.attr(v, 'data', name = path_name(data_path)), data_path)
      
      offset_path = path + ("offset",)
      offset_values = \
        self.flat_values(self.attr(v, 'offset', name = path_name(offset_path)), offset_path)
      
      shape_values = \
        self.flat_values(self.shape(v), path + ("shape",))
      
      stride_values = \
        self.flat_values(self.strides(v), path + ("strides",))
      
      result = data_values + offset_values + shape_values + stride_values  
    
    elif isinstance(t, SliceT):
      start_path = path + ("start",)
      start_values = self.flat_values(self.attr(v, name = path_name(start_path)), start_path)
      
      stop_path = path + ("stop",)
      stop_values = self.flat_values(self.attr(v, name = path_name(stop_path)), stop_path)
      
      step_path = path + ("step",)
      step_values = self.flat_values(self.attr(v, name = path_name(step_path)), step_path)
      
      result = start_values + stop_values + step_values 
    
    elif isinstance(t, TupleT):
      result = []
      for i, elt in enumerate(self.tuple_elts(v)):
        field = "elt%d" % i 
        elt_path = path + (field,)
        if elt.__class__ is not Var:
          elt = self.assign_name(elt, path_name(elt_path))
        result.extend(self.flat_values(elt, elt_path))
      result = tuple(result)
    self.paths[path] = result
    return result 
    
  def flatten_var(self, v):
    assert v.__class__ is Var
    path = (v.name,)  
    return self.flat_values(v, path)
  
  def get_path(self, expr):
    c = expr.__class__  
    if c is Const:
      return ()
    elif c is Var:
      return (expr.name,)
    elif c is Attribute:
      return self.get_path(expr.value) + (expr.name,)
    elif c is TupleProj:
      return self.get_path(expr.tuple) + ("elt%d" % expr.index)
    elif c is ClosureElt:
      return self.get_path(expr.closure) + ("closure_elt%d" % expr.index)
    else:
      assert False, "Can't get path of expression %s" % expr 
    
    
  def flatten_expr(self, expr):
    c = expr.__class__
    if c is Var:
      return self.flatten_var(expr)
    elif c is Const: 
      return expr
    path = self.get_path(expr)  
    if c is Attribute:
      field_name = expr.name 
      idx = expr.type.field_index(field_name)
      vs = self.flat_values(v, path)
  
  
  
  def transform_Tuple(self, expr):
    pass
  
  def transform_TupleProj(self, expr):
    pass  
  
  def transform_Assign(self, stmt):
    pass 
    """
    c = stmt.rhs.__class__
    if c is Tuple:
       
    elif c is TupleProj: 
    elif c is Closure:
      lhs_vars = 
      # enumerate function and fixed args
      # and assign then to lhs fixed args 
      
    elif c is ClosureElt:
    elif
    """ 
      
    
  #def flatten_vars(self, vars):
  #  result = []
  #  for var in vars:
  #    result.extend(self.flatten_value(var, path=(var.name,)))
  #  return tuple(result)
  
  def flatten_inputs(self):
    input_vars = self.input_vars(self.fn)
    
    for var in input_vars:
      name = var.name 
      self.flatten_input(var, (name,))
    
      


