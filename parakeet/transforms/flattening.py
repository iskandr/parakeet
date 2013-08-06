from .. import names 
from ..builder import build_fn
from ..ndtypes import ScalarT, NoneT, ArrayT, SliceT, TupleT, Int64, PtrT, ptr_type
from ..syntax import (Var, Attribute, Tuple, TupleProj, Closure, ClosureElt, Const,
                      Struct, Index) 

from transform import Transform
from parakeet.syntax.expr import TupleProj


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


class FlatRepr(object):
  pass 

class FlatTuple(FlattenedRepr):
  def __init__(self, elts):
    self.elts = elts
    
  def __getattr_(self, index):
    assert isinstance(index, (int, long))
    assert index < len(self.elts)
    return self.elts[index]

  def __iter__(self):
    for i, elt in enumerate(self.elts):
      if isinstance(elt, FlattenedRepr):
        for sub_path, sub_elt in elt:
          yield concat_path(i, sub_path), sub_elt
      else:
        yield i, elt  
        
def FlatStruct(FlattenedRepr):
  def __init__(self, fields):
    self.fields = fields
  
  def __iter__(self):
    for (k,v) in self.fields:
      if isinstance(v, FlattenedRepr):
        for sub_path, sub_elt in v:
          yield concat_path(k, sub_path), sub_elt 
      else:
        yield k,v 
      
  def __getattr__(self, k):
    for k2,v in self.fields:
      if k == k2:
        return v
    assert False, "Key not found %s" % k

def concat_path(p1, p2):
  if not isinstance(p1, tuple):
    p1 = (p1,)
    
  if not isinstance(p2, tuple):
    p2 = (p2,)
  return p1 + p2 

def path_name(path):
  return "_".join(path)


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
  
  
  
  
  def flatten_expr_tuple(self, exprs):
    result = []
    for expr in exprs:
      result.extend(self.transform_expr(expr))
    return tuple(result)
  
  def transform_expr(self, expr):
    result = Transform.transform_expr(self, expr)
    if isinstance(result, tuple):
      return result
    elif isinstance(result, list):
      return tuple(result)
    else:
      return (result,) 
  

  def build_path(self, expr):
    c = expr.__class__ 
    if c is Var:
      return (expr.name,)
    elif c is Attribute:
      base_path = self.build_path(expr.value)
      return base_path + (expr.name,)
    elif c is TupleProj:
      base_path = self.build_path(expr.tuple)
      return base_path + (expr.index,)
    else:
      assert False, "Can't build path for expression %s" % expr 

  def transform_Tuple(self, expr):
    return self.flatten_expr_tuple(expr.elts)
    
  def transform_TupleProj(self, expr):
    elts = self.transform_expr(expr.tuple)
    assert len(elts) >= expr.index, "Insufficient elements %s for tuple %s" (elts, expr)
    return elts[expr.index]
  
  def transform_Var(self, expr):
    
    #if expr.name in self.flattened_vars:
    #  return self.flattened_vars[expr.name]
    path = (expr.name,)
    if path in self.paths:
      return self.paths[path]
    return expr 
  
      
  def transform_Attribute(self, expr):
    fields = self.transform_expr(expr.value)
    field_index = expr.type.field_index(expr.name)
    # WHAT IF THE ATTRIBUTE IS ITSELF A STRUCTURED OBJECT?
    #  ie.e. {a : tuple(int, int), b : array } ??? 
    # then the fields return are going to need slicing or sub-indexing 
    return fields[field_index]
    
    
  
  def lhs_vars(self, expr):
    if isinstance(expr, Var):
      path = (expr.name,)
      if path in self.paths:
        return self.paths[path]
      self.paths[path] = expr
    elif isinstance(expr, Index):
      
    elif isinstance(expr, Attribute):
      assert expr.value.__class__ is Var, \
        "Flattening expects all attribute access to go through Var, not %s" % expr 
      path = (expr.value.name, expr.name)
      fields = self.transform_expr(expr.value)
      field_idx = expr.value.type.field_index(expr.name)
      return (fields[field_idx],)
    elif isinstance(expr, Tuple):
      result = []
      for elt in expr.elts:
        result.append(self.lhs_vars(elt))
      return tuple(result)
  
  def bind(self, lhs, rhs, path = ()):
    t = lhs.type 
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
    
    if isinstance(t, ScalarT):
      return  
    if isinstance(t, ArrayT):
      
    if isinstance(t, TupleT):
       result = []
      for i, elt in enumerate(self.tuple_elts(v)):
        field = "elt%d" % i 
        elt_path = path + (field,)
        if elt.__class__ is not Var:
          elt = self.assign_name(elt, path_name(elt_path))
        result.extend(self.flat_values(elt, elt_path))
      result = tuple(result)
    self.paths[path] = result
    
    assert path not self.paths, "Already have binding for access path %s : %s" % (path, t)
    
  def bind(self, lhs, rhs):
    c = lhs.__class__
    if c is Var:
      name = lhs.name 
      for rhs_path, rhs_var in rhs:
        lhs_path = (name,) + path
        lhs_name = path_name(lhs_path)
        lhs_var = self.fresh_var(lhs_name)
        self.assign(lhs_var, rhs_var) 
  
  def transform_lhs(self, expr, path = ()):
    t = expr.type 
    if isinstance(t, (NoneT, ScalarT)):
      result = (expr,)
      
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
    
    if isinstance(t, ScalarT):
      return  
    if isinstance(t, ArrayT):
      
    if isinstance(t, TupleT):
       result = []
      for i, elt in enumerate(self.tuple_elts(v)):
        field = "elt%d" % i 
        elt_path = path + (field,)
        if elt.__class__ is not Var:
          elt = self.assign_name(elt, path_name(elt_path))
        result.extend(self.flat_values(elt, elt_path))
      result = tuple(result)
    self.paths[path] = result
    
    assert path not self.paths, "Already have binding for access path %s : %s" % (path, t)
    
  def transform_Assign(self, stmt):
    lhs = tuple(self.transform_lhs(stmt.lhs))
    rhs = tuple(self.transform_expr(stmt.rhs))
    assert len(lhs) == len(rhs), "Mismatching in LHS terms %s and RHS terms %s" % (lhs, rhs)
    for lhs_elt, rhs_elt in zip(lhs,rhs):
      self.assign(lhs_elt, rhs_elt)
    return None 
   
    """  
    #lhs = stmt.lhs 
    #rhs_values = tuple(self.transform_expr(stmt.rhs)) 
    c = lhs.__class__
    assert c is not Tuple, \
      "Unexpected tuple LHS encountered while flattening: %s" % stmt
    if c is Index:
      # TODO: Make this work for arrays of structures/tuples
      assert len(rhs_values) == 1, \
        "Wrong number of flattened values on RHS of setidx: %s ==> %s" % (stmt, rhs_values) 
      assert lhs.value.type.__class__ is PtrT, \
        "Flattening expects all indexing to go directly through pointers, not %s" % lhs.value.type  
      self.rhs = rhs_values[0]
      return stmt 
    if lhs.__class__ is Var:
      base_path = (lhs.name,)
      self.bind(base_path, lhs.type, rhs_values)
      paths = self.expand(lhs.type, path = base_path)
      assert len(paths)
      
    lhs_vars = self.lhs_vars(stmt.lhs)
    rhs_vars = self.transform_expr(stmt.rhs)
    nvars = len(lhs_vars)
    assert nvars == len(rhs_vars)
    for i in xrange(nvars):
      lhs = lhs_vars[i]
      rhs = rhs_vars[i]
      self.assign(lhs, rhs)
    return None 
    
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
    
      


