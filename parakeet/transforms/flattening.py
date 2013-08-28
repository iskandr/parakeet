from .. import names 
from ..builder import build_fn
from ..ndtypes import (ScalarT, NoneT, ArrayT, SliceT, TupleT, make_tuple_type, 
                       Int64, PtrT, ptr_type, ClosureT, FnT)
from ..syntax import (Var, Attribute, Tuple, TupleProj, Closure, ClosureElt, Const,
                      Struct, Index, TypedFn) 
from ..syntax.helpers import none 

from transform import Transform

def concat(seqs):
  result = []
  for seq in seqs:
    for elt in seq:
      result.append(elt)
  return tuple(result)

def concat_args(*seqs):
  return concat(seqs)

def map_concat(f, seq):
  return concat(f(elt) for elt in seq)

def flatten_type(t):
  if isinstance(t, (ScalarT, PtrT)):
    return (t,)
  elif isinstance(t, TupleT):
    return concat(flatten_type(elt_t) for elt_t in t.elt_types)
  elif isinstance(t, (NoneT, FnT)):
    return ()
  elif isinstance(t, ClosureT):
    return concat(flatten_type(elt_t) for elt_t in t.arg_types)
  elif isinstance(t, ArrayT):
    return concat_args(
      [t.ptr_t, Int64], # base pointer and elt offset
      flatten_type(t.shape_t),
      flatten_type(t.strides_t)           
    )
  elif isinstance(t, SliceT):
    return (t.start_type, t.stop_type, t.step_type)
  else:
    assert False, "Unsupported type %s" % (t,)

def flatten_types(ts):
  return flatten_seq(*[flatten_type(t) for t in ts])
  
def field_pos_range(t, field, _cache = {}):
  key = (t, field)
  if key in _cache:
    return _cache[key]
  assert isinstance(t, StrucT)
  offset = 0
  
  for i, (field_name, field_t) in enumerate(t._fields_):
    n = len(flatten_type(field_t))
    if field_name == field or (isinstance(field, (int, long)) and i == field):
      result = (offset, offset+n)
      _cache[key] = offset
      return result
  assert False, "Field %s not found on type %s" % (field, t)

def single_type(ts):
  """
  Turn a sequence of types into a single type object
  """
  if len(ts) == 0:
    return NoneT
  elif len(ts) == 1:
    return ts[0]
  else:
    return make_tuple_type(ts)
  
def single_value(values):
  if len(values) == 0:
    return none 
  elif len(values) == 1:
    return values[0]
  else:
    t = make_tuple_type(tuple(v.t for v in values))
    return Tuple(values, type = t)
  
def mk_vars(names, types):
  """
  Combine a list of names and a list of types into a single list of vars
  """
  return [Var(name = name, type = t) for name, t in zip(names, types)]

class FlattenBody(object):
  """
  Take an ordinary function and make an equivalent function 
  where all variables have scalar or pointer types. This requires
  expanding structures into multiple components (thus changing
  both the input and output types of the fn)
  """
  
  def __init__(self, old_fn):
    self.old_fn = old_fn
    # maps name of var in old_fn to a collection of variables
    self.flat_types = {}
      
    self.blocks = NestedBlocks()
    
    # a var of struct type from the old fn 
    # maps to multiple variables in the new
    # function 
    self.var_expansions = {}
  
  def transform_TypedFn(self, expr):
    return build_flat_fn(expr)

class BuildFlatFn(object):
  def __init__(self, old_fn):
    self.old_fn = old_fn 
    base_name = names.original(old_fn.name)
    flat_fn_name = names.fresh("flat_" + base_name)

    self.type_env = {}
    for (name, t) in old_fn.type_env.iteritems():
      old_var = Var(name = name, type = type)
      for new_var in self.flatten_lhs_var(old_var):
        self.type_env[new_var.name] = new_var.type
    old_input_vars = mk_vars(old_fn.arg_names, old_fn.input_types)
    new_input_vars = self.flatten_lhs_var(old_input_vars)
    new_input_names = tuple(var.name for var in new_input_vars)
    new_input_types = tuple(var.type for var in new_input_var)
    new_return_type =  single_type(flatten_type(old_fn.return_type))
    
    self.flat_fn = \
      TypedFn(name = flat_fn_name, 
        arg_names = new_input_names, 
        body = [], 
        type_env = self.type_env,
        input_types = new_input_types, 
        return_type = new_return_type) 
    self.blocks = NestedBlocks()
    
  def run(self):
    self.flat_fn.body = self.flatten_block(self.old_fn.body)
    return self.flat_fn
  
  
  #######################
  #
  #     Helpers
  #
  #######################
  
  def extract_fn(self, fn):
    t = fn.type 
    if isinstance(t, FnT):
      return fn 
    else:
      assert isinstance(t, ClosureT)
      return t.fn 
    
  def make_closure(self, fn, args):
    if len(args) == 0:
      return fn 
    else:
      arg_types = tuple(arg.type for arg in args)
      t = make_closure_type(fn, arg_types)
      return Closure(fn, args, type = t)
    
  
  def flatten_block(self, stmts):
    self.blocks.push()
    for stmt in stmts:
      result = self.flatten_stmt(stmt)
      if result is None:
        continue
      if not isinstance(result, (list,tuple)):
        result = [result]
      for new_stmt in result:
        assert isinstance(new_stmt, Stmt), "Expected statement, got %s" % (new_stmt,)
        self.blocks.append(new_stmt)
    return self.blocks.pop()
  
  
  
  #######################
  #
  #     Statements
  #
  #######################
  
  def flatten_Assign(self, stmt):
    c = stmt.lhs.__class__
    
    if c is Var:
      result = []
      vars = self.flatten_lhs_var(stmt.lhs)
      values = self.flatten_expr(stmt.rhs)
      assert len(vars) == len(values)
      self.var_expansions(stmt.lhs.name, values)
      
      for var, value in zip(vars, values):
        result.append(Assign(var, value))
      return result 
    elif c is Index:
      assert False, "LHS indexing not implemented"  
  
  def flatten_ForLoop(self, stmt):
    var = self.flatten_scalar_lhs_var(stmt.var)
    start = self.flatten_scalar_expr(stmt.start)
    stop = self.flatten_scalar_expr(stmt.stop)
    step = self.flatten_scalar_expr(stmt.step)
    body = self.flatten_block(stmt.body)
    merge = self.flatten_merge(stmt.merge)
    return ForLoop(var, start, stop, step, body, merge)
  
  def flatten_While(self, stmt):
    cond = self.flatten_expr(stmt.cond)
    body = self.flatten_block(stmt.body)
    merge = self.flatten_merge(stmt.merge)
    return While(cond, body, merge)
     
   
  def flatten_ExprStmt(self, stmt):
    return ExprStmt(value = self.flatten_expr(stmt.value))
   
  def flatten_ParFor(self, stmt):
    old_fn = elf.extract_fn(stmt.fn)
    assert isinstance(old_fn, TypedFn)
    new_fn = self.transform_TypedFn(old_fn)
    
    closure_elts  = self.flatten_expr(stmt.fn) 
    closure = self.make_closure(fn, closure_elts)
    bounds = single_value(self.flatten_expr(stmt.bounds))
  
  def flatten_Comment(self, stmt):
    return stmt 
  
  def flatten_stmt(self, stmt):
    method_name = "flatten_%s" % stmt.__class__.__name__
    return getattr(self, method_name)(stmt)
  
  #######################
  #
  #     Expressions 
  #
  #######################
  
  def flatten_expr(self, expr):
    method_name = "flatten_%s" % expr.__class__.__name__
    return getattr(self, method_name)(stmt)
  
  def flatten_expr_list(self, exprs):
    return map_concat(self.flatten_expr, exprs)
  
  def flatten_Const(self, expr):
    return [expr]
  
  def flatten_Cast(self, expr):
    return [expr]
  
  def flatten_Var(self, expr):
    if isinstance(expr.type, (ScalarT, PtrT)):
      return (expr,)
    elif isinstance(expr.type, NoneT):
      return ()
    else:
      name = expr.name
      assert name in self.var_expansions 
      return self.var_expansions[name]
    
  def flatten_Tuple(self, expr):
    return self.flatten_expr_list(expr.elts)
  
  def flatten_field(self, struct, field):
    elts = self.flatten_expr(struct)
    start, stop = field_pos_range(struct.type, field)
    return elts[start:stop]
   
  def flatten_TupleProj(self, expr):
    return self.flatten_field(expr.tuple, expr.index)
  
  def flatten_Closure(self, expr):
    pass 
  
  def flatten_ClosureElt(self, expr):
    return self.flatten_field(expr.closure, expr.index)
  
  def flatten_Attribute(self, expr):
    return self.flatten_field(expr.value, expr.name)
  
  def flatten_Alloc(self, expr):
    pass 
  
  def flatten_Array(self, expr):
    assert False, "Array node should be an explicit allocation by now"
    # or, if we flatten structured elts, maybe we should handle it here?
  def flatten_Index(self, expr):
    
    #expr.value
    #expr.index 
    #expr.check_negative 
    pass 
  
  def flatten_Slice(self, expr):
    return self.flatten_expr_list([expr.start, expr.stop, expr.step])
  
  def flatten_Len(self, expr):
    pass 
  
  def flatten_ConstArray(self, expr):
    pass 
  
  def flatten_ConstArrayLike(self, expr):
    pass 
  
  def flatten_Range(self, expr):
    pass 
  
  def flatten_AllocArray(self, expr):
    pass 
  
  def flatten_ArrayView(self, expr):
    pass 
  
  def flatten_Ravel(self, expr):
    pass 
  
  def flatten_Reshape(self, expr):
    pass 
  
  def flatten_Shape(self, expr):
    pass 
  
  def flatten_Strides(self, expr):
    pass 
  
  def flatten_Transpose(self, expr):
    pass 
  
  def flatten_Where(self, expr):
    pass 
     
   
    
  #######################
  #
  # Adverbs
  #
  #######################
  def flatten_Map(self, expr):
    pass 
  
  def flatten_Reduce(self, expr):
    pass 
  
  def flatten_Scan(self, expr):
    pass
  
  def flatten_IndexMap(self, expr):
    pass
  
  def flatten_IndexReduce(self, expr):
    pass 
  
  def flatten_IndexScan(self, expr):
    pass 
  
  def flatten_OuterMap(self, expr):
    pass 
  
  
  
  def flatten_scalar_expr(self, expr):
    """
    Give me back a single expression instead of a list 
    """
    flat_exprs = self.flatten_expr(expr)
    assert len(flat_exprs) == 1
    return flat_exprs[0]
    
  def flatten_lhs_var(self, old_var):
    name = old_var.name 
    t = old_var.type 
    if isinstance(t, (ScalarT, PtrT)):
      return [old_var]
    elif isinstance(t, (FnT, NoneT)):
      return []
    elif isinstance(t, SliceT):
      start = Var(name = "%s_start" % name, type = t.start_type)
      stop = Var(name = "%s_stop" % name, type = t.stop_type)
      step = Var(name = "%s_step" % name, type = t.step_type)
      field_vars = [start, stop, step]
    elif isinstance(t, ClosureT):
      field_vars = [Var(name = "%s_closure_elt%d" % (name,i) , type = t) 
                    for i,t in enumerate(t.arg_types)]
    elif isinstance(t, TupleT):
      field_vars = [Var(name = "%s_elt%d" % (name, i), type = t) 
                    for i,t in enumerate(t.elt_types)]
    elif isinstance(t, ArrayT):
      data = Var(name = "%s_data" % name, type = t.ptr_t)
      offset = Var(name = "%s_offset" % name, type = Int64)
      shape = Var(name = "%s_shape" % name, type = t.shape_t)
      strides = Var(name = "%s_strides" % name, type = t.strides_t)
      field_vars = [data, offset, shape, strides]
    else:
      assert Fasle, "Unsupport type %s" % (t,)
    return self.flatten_lhs_vars(field_vars)
  
  def flatten_lhs_vars(self, old_vars):
    return map_concat(self.flatten_lhs_var, old_vars)
  
  def flatten_scalar_lhs_var(self, old_var):
    vars = self.flatten_lhs_var(old_var)
    assert len(vars) == 1
    return vars[0]
  
def build_flat_fn(old_fn, _cache = {}):
  key = (old_fn.name, old_fn.copied_by)
  if key in _cache:
    return _cache[key]
  
  flat_fn = BuildFlatFn(old_fn).run()
  _cache[key] = flat_fn
  return flat_fn
  













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