from .. import names 
from treelike import NestedBlocks
from ..builder import build_fn
from ..ndtypes import (ScalarT, NoneT, ArrayT, SliceT, TupleT, make_tuple_type, 
                       Int64, PtrT, ptr_type, ClosureT, FnT, StructT)
from ..syntax import (Var, Attribute, Tuple, TupleProj, Closure, ClosureElt, Const,
                      Struct, Index, TypedFn, Return, Stmt, Assign) 
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

def concat_map(f, seq):
  print "concat_map", f, seq
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
      flatten_type(t.ptr_t), 
      [Int64], # base pointer and elt offset
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
  assert isinstance(t, StructT)
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


class BuildFlatFn(object):
  def __init__(self, old_fn):
    self.old_fn = old_fn 
    base_name = names.original(old_fn.name)
    flat_fn_name = names.fresh("flat_" + base_name)

    self.type_env = {}
    for (name, t) in old_fn.type_env.iteritems():
      old_var = Var(name = name, type = t)
      for new_var in self.flatten_lhs_var(old_var):
        self.type_env[new_var.name] = new_var.type
    old_input_vars = mk_vars(old_fn.arg_names, old_fn.input_types)
        
    # a var of struct type from the old fn 
    # maps to multiple variables in the new
    # function 
    self.var_expansions = {}
    
    new_input_vars = []
    for var in old_input_vars:
      flat_vars = self.flatten_lhs_var(var)
      self.var_expansions[var.name]  = flat_vars 
      new_input_vars.extend(flat_vars)
    
    new_input_names = tuple(var.name for var in new_input_vars)
    new_input_types = tuple(var.type for var in new_input_vars)
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
      self.var_expansions[stmt.lhs.name] = values
      
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
  
  def flatten_Return(self, stmt):
    return Return(single_value(self.flatten_expr(stmt.value)))
  
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
    return getattr(self, method_name)(expr)
  
  def flatten_expr_list(self, exprs):
    return concat_map(self.flatten_expr, exprs)
  
  def flatten_scalar_expr(self, expr):
    """
    Give me back a single expression instead of a list 
    """
    flat_exprs = self.flatten_expr(expr)
    assert len(flat_exprs) == 1
    return flat_exprs[0]
    
  def flatten_scalar_expr_list(self, exprs):
    assert isinstance(exprs, (list, tuple)), "Expected list, got %s" % (exprs,)
    return [self.flatten_scalar_expr(e) for e in  exprs]
  
  def flatten_Const(self, expr):
    return [expr]
  
  def flatten_Cast(self, expr):
    return [expr]
  
  def transform_TypedFn(self, expr):
    return build_flat_fn(expr)

    
  def flatten_Var(self, expr):
    if isinstance(expr.type, (ScalarT, PtrT)):
      return (expr,)
    elif isinstance(expr.type, NoneT):
      return ()
    else:
      name = expr.name
      assert name in self.var_expansions, "No fields known for %s : %s" % (expr, expr.type) 
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
    assert False, "Not implemented" 
  
  def flatten_ClosureElt(self, expr):
    return self.flatten_field(expr.closure, expr.index)
  
  def flatten_Attribute(self, expr):
    return self.flatten_field(expr.value, expr.name)
  
  def flatten_Alloc(self, expr):
    assert False, "Not implemented" 
  
  def flatten_Array(self, expr):
    assert False, "Array node should be an explicit allocation by now"
    # or, if we flatten structured elts, maybe we should handle it here?
  def flatten_Index(self, expr):
    
    #expr.value
    #expr.index 
    #expr.check_negative 
    assert False 
  
  def flatten_PrimCall(self, expr):
    args = self.flatten_scalar_expr_list(expr.args)
    expr.args = args
    return [expr] 
  
  def flatten_Slice(self, expr):
    return self.flatten_expr_list([expr.start, expr.stop, expr.step])
  
  def flatten_Len(self, expr):
    assert False, "Not implemented" 
  
  def flatten_ConstArray(self, expr):
    assert False, "Not implemented" 
  
  def flatten_ConstArrayLike(self, expr):
    assert False, "Not implemented" 
  
  def flatten_Range(self, expr):
    assert False, "Not implemented" 
  
  def flatten_AllocArray(self, expr):
    assert False, "Not implemented" 
  
  def flatten_ArrayView(self, expr):
    assert False, "Not implemented" 
  
  def flatten_Ravel(self, expr):
    assert False, "Not implemented" 
  
  def flatten_Reshape(self, expr):
    assert False, "Not implemented" 
  
  def flatten_Shape(self, expr):
    assert False, "Not implemented" 
  
  def flatten_Strides(self, expr):
    assert False, "Not implemented" 
  
  def flatten_Transpose(self, expr):
    assert False, "Not implemented" 
  
  def flatten_Where(self, expr):
    assert False, "Not implemented" 
     
   
    
  #######################
  #
  # Adverbs
  #
  #######################
  def flatten_Map(self, expr):
    assert False, "Not implemented" 
  
  def flatten_Reduce(self, expr):
    assert False, "Not implemented" 
  
  def flatten_Scan(self, expr):
    assert False, "Not implemented"
  
  def flatten_IndexMap(self, expr):
    assert False, "Not implemented"
  
  def flatten_IndexReduce(self, expr):
    assert False, "Not implemented" 
  
  def flatten_IndexScan(self, expr):
    assert False, "Not implemented" 
  
  def flatten_OuterMap(self, expr):
    assert False, "Not implemented" 
  
  
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
      assert False, "Unsupport type %s" % (t,)
    print field_vars, self.flatten_lhs_vars(field_vars)
    return self.flatten_lhs_vars(field_vars)
  
  def flatten_lhs_vars(self, old_vars):
    return concat_map(self.flatten_lhs_var, old_vars)
  
  def flatten_scalar_lhs_var(self, old_var):
    vars = self.flatten_lhs_var(old_var)
    assert len(vars) == 1
    return vars[0]
  
def build_flat_fn(old_fn, _cache = {}):
  key = (old_fn.name, old_fn.copied_by)
  print "BUILD FLAT FN", key 
  if key in _cache:
    return _cache[key]
  print "MISS"
  flat_fn = BuildFlatFn(old_fn).run()
  _cache[key] = flat_fn
  _cache[(flat_fn.name, flat_fn.copied_by)] = flat_fn
  return flat_fn
  


class Flatten(Transform):
  
  def unbox_var(self, var):
    t = var.type 
    if isinstance(t, (FnT, NoneT)):
      return []
    elif isinstance(t, (PtrT, ScalarT)):
      return [var]
    elif isinstance(t, ArrayT):
      # for structured arrays, should this be a tuple? 
      data = self.attr(var, 'data', name = var.name + "_data")
      offset = self.attr(var, 'offset', name = var.name + "_offset")
      shape = self.attr(var, 'shape', name = var.name + "_shape")
      strides = self.attr(var, 'strides', name = var.name + "_strides")
      return self.unbox_vars([data, offset, shape, strides])
    elif isinstance(t, SliceT):
      start = self.attr(var, 'start')
      stop = self.attr(var, 'stop')
      step = self.attr(var, 'step')
      return self.unbox_vars([start, stop, step])
    elif isinstance(var, ClosureT):
      closure_elts = [self.closure_elt(var, i, name = var.name + "_closure_elt%d" % i)
                      for i in xrange(len(t.arg_types))]
      return self.unbox_vars(closure_elts)
    elif isinstance(var, TupleT):
      tuple_elts = [self.tuple_proj(var, i, name = var.name + "_elt%d" % i)
                      for i in xrange(len(t.elt_types))]
      return self.unbox_vars(tuple_elts)
    else:
      assert False, "Unsupported type %s" % (t,)
  
  def unbox_vars(self, vars):
    return concat_map(self.unbox_var, vars)
  
  def to_seq(self, expr):
    if isinstance(expr.type, TupleT):
      return self.tuple_elts(expr)
    else:
      return [expr]
      
  
  def box(self, t, elts):
    if isinstance(t, NoneT):
      assert len(elts) == 0
      return none 
    elif isinstance(t, ScalarT):
      assert len(elts) == 1
      return elts[0]
    elif isinstance(t, SliceT):
      assert len(elts) == 3
      start, stop, step = elts
      return self.slice_value(start, stop, step)
    elif isinstance(t, ArrayT):
      assert False, "Not implemented"
    elif isinstance(t, TupleT):
      assert False, "Not implemented" 
    elif isinstance(t, ClosureT):
      assert False, "Not implemented" 
    elif isinstance(t, FnT):
      assert False, "Not implemented" 
  
  def transform_block(self, stmts):
    return stmts
   
  
  def pre_apply(self, old_fn, _cache = {}):
   
     
    if old_fn.name in _cache:
      return _cache[old_fn.name]
    
    flat_fn = build_flat_fn(old_fn)
    input_vars = mk_vars(old_fn.arg_names, old_fn.input_types)
    self.blocks.push()
    unboxed_inputs = self.unbox_vars(input_vars)
    assert len(unboxed_inputs) == len(flat_fn.input_types)
    unboxed_result = self.call(flat_fn, unboxed_inputs, name = "unboxed_result")
    unboxed_elts = self.to_seq(unboxed_result)
    boxed_result = self.box(old_fn.return_type, unboxed_elts)
    self.return_(boxed_result)
    old_fn.body = self.blocks.pop()
    _cache[old_fn.name] = old_fn
    _cache[flat_fn.name] = flat_fn 
    return old_fn
  