from .. import names 
from ..builder import build_fn
from ..ndtypes import ScalarT, NoneT, ArrayT, SliceT, TupleT, Int64, PtrT, ptr_type, ClosureT, FnT
from ..syntax import (Var, Attribute, Tuple, TupleProj, Closure, ClosureElt, Const,
                      Struct, Index, TypedFn) 

from transform import Transform
from parakeet.syntax.expr import TupleProj









def flatten_seqs(*seqs):
  result = []
  for seq in seqs:
    for elt in seq:
      result.append(elt)
  return tuple(result)


def flatten_type(t):
  if isinstance(t, (ScalarT, PtrT)):
    return (t,)
  elif isinstance(t, TupleT):
    return flatten_seqs(flatten_type(elt_t) for elt_t in t.elt_types)
  elif isinstance(t, (NoneT, FnT)):
    return ()
  elif isinstance(t, ClosureT):
    return flatten_seqs(flatten_type(elt_t) for elt_t in t.arg_types)
  elif isinstance(t, ArrayT):
    return flatten_seqs(
      [t.ptr_t, Int64], # base pointer and elt offset
      flatten_type(t.shape_t),
      flatten_type(t.strides_t)           
    )
  elif isinstance(t, SliceT):
    return (t.start_type, t.stop_type, t.step_type)
  else:
    assert False, "Unsupported type %s" % (t,)
    
def field_pos(t, field, _cache = {}):
  key = (t, field)
  if key in _cache:
    return _cache[key]
  assert isinstance(t, StrucT)
  offset = 0
  for (field_name, field_t) in t._fields_:
    n = len(flatten_type(field_t))
    if field_name == field:
      result = (offset, offset+n)
      _cache[key] = offset
      return result
  assert False, "Field %s not found on type %s" % (field, t)

def build_unwrap_inputs(input_types, arg_names):
  pass 

def build_wrap_result(f):
  pass






















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

class FlatTuple(FlatRepr):
  def __init__(self, elts):
    self.elts = elts
    
  def __getattr_(self, index):
    assert isinstance(index, (int, long))
    assert index < len(self.elts)
    return self.elts[index]

  def __iter__(self):
    for i, elt in enumerate(self.elts):
      if isinstance(elt, FlatRepr):
        for sub_path, sub_elt in elt:
          yield concat_path(i, sub_path), sub_elt
      else:
        yield i, elt  
        
class FlatStruct(FlatRepr):
  def __init__(self, fields):
    self.fields = fields
  
  def __iter__(self):
    for (k,v) in self.fields:
      if isinstance(v, FlatRepr):
        for sub_path, sub_elt in v:
          yield concat_path(k, sub_path), sub_elt 
      else:
        yield k,v 
      
  def __getitem__(self, k):
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
    self.env = {}
    
    old_input_vars = self.input_vars(fn)
    
    new_input_vars = []
    for var in old_input_vars:
      name = var.name
      t = var.type  
      if isinstance(t, ScalarT):
        new_input_vars.append(var)
      else:
        new_input_vars.extend(self.transform_lhs_type(t, path= (name,)))
    
    return fn 
    #fn_name = names.add_prefix("flat_", fn.name)
    #arg_names = [var.name for var in new_input_vars]
    #arg_types = [var.type for var in new_input_vars] 
    #type_env = dict( (var.name, var.type) for var in new_input_vars)
    #return TypedFn(name = fn_name,
    #          arg_names = arg_names,
    #          body = fn.body, 
    #          input_types = arg_types, 
    #          return_type = flatten_type(fn.return_type),
    #          type_env = type_env)
    
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
    

  
  def flatten(self, *exprs):
    result = []
    for expr in exprs:
      c = expr.__class__ 
      if c is Tuple:
        result.extend(expr.elts)
      else:
        result.append(expr)
    if len(result) == 1:
      return result[0]
    else:
      return self.tuple(result)
  
  #def flatten_expr_tuple(self, exprs):
  #  result = []
  #  for expr in exprs:
  #    result.extend(self.transform_expr(expr))
  #  return tuple(result)
  
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

  #def transform_Tuple(self, expr):
  #  return self.flatten_expr_tuple(expr.elts)
   
  def transform_expr(self, expr):
    return self.flatten(Transform.transform_expr(self, expr))
   
  def transform_TupleProj(self, expr):
    elts = self.transform_expr(expr.tuple)
    assert len(elts) >= expr.index, "Insufficient elements %s for tuple %s" (elts, expr)
    return elts[expr.index]
  
  def transform_Var(self, expr):
    if expr.name in self.env:
      return self.env[expr.name]
    return expr 
  
      
  def transform_Attribute(self, expr):
    
    fields = self.transform_expr(expr.value)
    print expr, expr.value.type, fields 
    return fields[expr.name]
    
  
  def transform_lhs_type(self, t, path):
    name = "_".join(path)
    if isinstance(t, (NoneT, ScalarT)):
      result = (self.fresh_var(name),)
      print "transform_lhs_type %s" % t, result    
    elif isinstance(t, ArrayT):
      data = self.fresh_var(name + "_data")
      offset = self.fresh_var(name + "_offset")
      shape = self.transform_lhs_type(t.shape_t, path = (name,'shape'))
      strides = self.transform_lhs_type(t.strides_t, path = (name, 'strides'))
             
      result = FlatStruct([('data', data), 
                           ('offset',offset), 
                           ('shape', shape),
                           ('strides',strides)])
      print "transform_lhs_type ArrayT", result 
    elif isinstance(t, SliceT):
      start_var = self.fresh_var(name + "_start")
      stop_var = self.fresh_var(name + "_stop")
      step_var = self.fresh_var(name + "_step")
      result = FlatStruct([("start", start_var), ("stop", stop_var), ("step",step_var)])
      print "transform_lhs_type SliceT", result 
    elif isinstance(t, TupleT):
      result = []
      for i, elt_t in enumerate(t.elt_types):
        field = "elt%d" % i 
        elt_path = path + (field,)
        result.extend(self.transform_lhs_type(elt_t, elt_path))
      result = FlatTuple(result)
      print "transform_lhs_type TupleT", result
    elif isinstance(t, ClosureT):
      result = []
      for i, elt_t in enumerate(t.arg_types):
        field = "closure_arg%d" % i
        elt_path = path + (field,)
        result.extend(self.transform_lhs_type(elt_t, elt_path))
      print "transform_lhs_type %s" % t, result
    else:
      assert False, "Unsupported type %s" % t  
  
    self.env[name] = result 
    assert result is not None, (t, path)
    return result
 
  def transform_lhs(self, expr, path = ()):
    c = expr.__class__ 
    if c is Tuple:
      return FlatTuple([self.transform_lhs(elt, path = path + (("elt%d" % i) ,)) 
                        for i, elt in enumerate(expr.elts)])
    
    elif c is Index:
      assert isinstance(expr.array.type, PtrT)
      assert isinstance(expr.index, (Const, Var))
      return (expr,)
   
    elif c is Attribute:
      
      assert isinstance(expr.value, Var)
      fields = self.transform_Var(expr.value)
      assert isinstance(fields, FlatStruct)
      return (fields[expr.name],)
    
    assert c is Var
    assert len(path) == 0
    return self.transform_lhs_type(expr.type, path = (expr.name,))

    
  def transform_Assign(self, stmt):
    lhs = tuple(self.transform_lhs(stmt.lhs))
    rhs = tuple(self.transform_expr(stmt.rhs))
    assert len(lhs) == len(rhs), "Mismatching in LHS terms %s and RHS terms %s" % (lhs, rhs)
    for lhs_elt, rhs_elt in zip(lhs,rhs):
      self.assign(lhs_elt, rhs_elt)
    return None 
    
  def transform_Return(self, stmt):
    t = stmt.value.type
    if isinstance(t, ScalarT) and self.is_simple(stmt.value):
      return stmt 
    stmt.value = self.transform_expr(stmt.value)
    return stmt  