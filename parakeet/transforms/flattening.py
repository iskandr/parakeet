from .. import names 
from treelike import NestedBlocks
from ..builder import build_fn, Builder
from ..ndtypes import (ScalarT, NoneT, NoneType, ArrayT, SliceT, TupleT, make_tuple_type, 
                       Int64, PtrT, ptr_type, ClosureT, make_closure_type, FnT, StructT)
from ..syntax import (Var, Attribute, Tuple, TupleProj, Closure, ClosureElt, Const,
                      Struct, Index, TypedFn, Return, Stmt, Assign, Alloc, AllocArray, 
                      ParFor, PrimCall, If) 
from ..syntax.helpers import none, const_int 

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
      flatten_type(t.shape_t),
      flatten_type(t.strides_t),          
      (Int64, Int64), # offset and size
      
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
  assert isinstance(t, StructT), "Expected struct got %s.%s" % (t, field)
  offset = 0
  for i, (field_name, field_t) in enumerate(t._fields_):
    n = len(flatten_type(field_t))
    if field_name == field or (isinstance(field, (int, long)) and i == field):
      result = (offset, offset+n)
      
      _cache[key] = result
      return result
    offset += n 
  assert False, "Field %s not found on type %s" % (field, t)

def get_field_elts(t, values, field):
  start, stop = field_pos_range(t, field)
  return values[start:stop]

def single_type(ts):
  """
  Turn a sequence of types into a single type object
  """
  if len(ts) == 0:
    return NoneType
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
    t = make_tuple_type(tuple(v.type for v in values))
    return Tuple(values, type = t)
  
def mk_vars(names, types):
  """
  Combine a list of names and a list of types into a single list of vars
  """
  return [Var(name = name, type = t) for name, t in zip(names, types)]


class BuildFlatFn(Builder):
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
    values = self.flatten_expr(stmt.rhs)
    if c is Var:
      result = []
      vars = self.flatten_lhs_var(stmt.lhs)
      assert len(vars) == len(values), "Mismatch between LHS %s and RHS %s" % (vars, values)
      self.var_expansions[stmt.lhs.name] = values 
      for var, value in zip(vars, values):
        result.append(Assign(var, value))
      return result 
    elif c is Index:
      
      idx = self.flatten_scalar_expr(stmt.lhs.index)
      values = self.flatten_expr(stmt.lhs.value)
      array_t = stmt.lhs.value.type 
      if isinstance(array_t, PtrT):
        return stmt  
      data = get_field_elts(array_t, values, 'data')[0]
      shape = get_field_elts(array_t, values, 'shape')
      strides = get_field_elts(array_t, values, 'strides')
      offset = get_field_elts(array_t, values, 'offset')[0]
      offset = self.add(offset, self.mul(idx, strides[0]))
      stmt.lhs = Index(data, offset)
      stmt.rhs = values[0]
      return [stmt]
  
  def flatten_merge(self, phi_nodes):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      t = left.type
      assert right.type == t 
      if isinstance(expr.type, (ScalarT, PtrT)):
        result[k] = (self.flatten_scalar_expr(left), self.flatten_scalar_expr(right))
      elif isinstance(expr.type, (FnT, NoneT)):
        continue 
      else:
        vars = self.var_expansions[k]
        flat_left = self.flatten_expr(left)
        flat_right = self.flatten_expr(right)
        assert len(vars) == len(flat_left)
        assert len(vars) == len(flat_right)
        for i, var in enumerate(vars):
          result[var.name] = (flat_left[i], flat_right[i])
    return result 
   
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
     
  
  def flatten_If(self, stmt):
    cond = self.flatten_scalar_expr(stmt.cond)
    true = self.flatten_block(stmt.true)
    false = self.flatten_block(stmt.false)
    merge = self.flatten_merge(stmt.merge)
    assert merge is not None
    return If(cond, true, false, merge = merge)
  
  def flatten_ExprStmt(self, stmt):
    return ExprStmt(value = self.flatten_expr(stmt.value))
   
  def flatten_ParFor(self, stmt):
    old_fn = self.get_fn(stmt.fn)
    assert isinstance(old_fn, TypedFn)
    new_fn = self.flatten_TypedFn(old_fn)
    
    closure_elts  = self.flatten_expr_list(self.closure_elts(stmt.fn)) 
    closure = self.closure(new_fn, closure_elts)
    bounds = single_value(self.flatten_expr(stmt.bounds))
    return ParFor(fn = closure, bounds = bounds)
    
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
    if isinstance(expr.type, NoneT):
      return []
    return [expr]
  
  def flatten_Cast(self, expr):
    return [expr]
  
  def flatten_TypedFn(self, expr):
    result = build_flat_fn(expr)
    return result
    
  def flatten_Var(self, expr):
    if isinstance(expr.type, (ScalarT, PtrT)):
      return (expr,)
    elif isinstance(expr.type, (FnT, NoneT)):
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
    fn = self.flatten_TypedFn(expr.fn)
    args = self.flatten_expr_list(expr.args)
    return [self.closure(fn, args)]
  
  def flatten_ClosureElt(self, expr):
    return self.flatten_field(expr.closure, expr.index)
  
  def flatten_Attribute(self, expr):
    return self.flatten_field(expr.value, expr.name)
  
  def flatten_Select(self, expr):
    cond = self.flatten_scalar_expr(expr.cond)
    true_values = self.flatten_expr(expr.true_value)
    false_values = self.flatten_expr(expr.false_value)
    assert len(true_values) == len(false_values)
    return [self.select(cond,t,f) for t,f in zip(true_values, false_values)]
    
  def flatten_Alloc(self, expr):
    assert False, "Not implemented" 
  
  def flatten_Array(self, expr):
    assert False, "Array node should be an explicit allocation by now"
    # or, if we flatten structured elts, maybe we should handle it here?
  def flatten_Index(self, expr):
    
    t = expr.value.type 
    if isinstance(t, PtrT):
      return [expr]   
    assert isinstance(t, ArrayT), "Expected Index to take array, got %s" % (expr.type,)
    array_fields = self.flatten_expr(expr.value)
    data_fields = get_field_elts(t, array_fields, 'data')
    shape = get_field_elts(t, array_fields, 'shape')
    strides = get_field_elts(t, array_fields, 'strides')
    offset = get_field_elts(t, array_fields, 'offset')[0]
    assert isinstance(expr.index.type, ScalarT)
    return [self.index(data_fields[0], 
                       self.add(offset, self.mul(expr.index, strides[0])))]
     
  
  def flatten_PrimCall(self, expr):
    args = self.flatten_scalar_expr_list(expr.args)
    return [PrimCall(prim = expr.prim, args = args, type = expr.type)]
  
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
  
  def strides_from_shape_elts(self, shape_elts):
    strides = [const_int(1)]
    for dim in reversed(shape_elts[1:]):
      strides = [self.mul(strides[0], dim)] + strides
    return strides
  
  def flatten_AllocArray(self, expr):
    nelts = const_int(1)
    shape_elts = self.flatten_expr(expr.shape)
    for dim in shape_elts:
      nelts = self.mul(nelts, dim)
    ptr = Alloc(elt_type = expr.elt_type, count = nelts, type = ptr_type(expr.elt_type))
    stride_elts = self.strides_from_shape_elts(shape_elts)
    return (ptr,) + tuple(shape_elts) + tuple(stride_elts) + (self.int(0), nelts)
  
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
      shape = Var(name = "%s_shape" % name, type = t.shape_t)
      strides = Var(name = "%s_strides" % name, type = t.strides_t)
      offset = Var(name = "%s_offset" % name, type = Int64)
      nelts = Var(name = "%s_nelts" % name, type = Int64)
      field_vars = [data, shape, strides, offset, nelts]
    
    else:
      assert False, "Unsupport type %s" % (t,)
    return self.flatten_lhs_vars(field_vars)
  
  def flatten_lhs_vars(self, old_vars):
    return concat_map(self.flatten_lhs_var, old_vars)
  
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
      shape = self.attr(var, 'shape', name = var.name + "_shape")
      strides = self.attr(var, 'strides', name = var.name + "_strides")
      offset = self.attr(var, 'offset', name = var.name + "_offset")
      size = self.attr(var, 'size', name = var.name + "_size")
      return self.unbox_vars([data, shape, strides, offset, size])
    elif isinstance(t, SliceT):
      start = self.attr(var, 'start')
      stop = self.attr(var, 'stop')
      step = self.attr(var, 'step')
      return self.unbox_vars([start, stop, step])
    elif isinstance(t, ClosureT):
      closure_elts = [self.closure_elt(var, i, name = var.name + "_closure_elt%d" % i)
                      for i in xrange(len(t.arg_types))]
      return self.unbox_vars(closure_elts)
    elif isinstance(t, TupleT):
      tuple_elts = [self.assign_name(self.tuple_proj(var, i), name = var.name + "_elt%d" % i)
                      for i in xrange(len(t.elt_types))]
      return self.unbox_vars(tuple_elts)
    else:
      assert False, "Unsupported type %s" % (t,)
  
  def unbox_vars(self, vars):
    return concat_map(self.unbox_var, vars)
  
  def to_seq(self, expr):
    if isinstance(expr.type, TupleT):
      return self.tuple_elts(expr)
    elif isinstance(expr.type, (FnT, NoneT)):
      return []
    else:
      return [expr]
      
  
  def box(self, t, elts):
    if isinstance(t, NoneT):
      assert len(elts) == 0, "Expected 0 values for None, got %s" % (elts,)
      return none 
    elif isinstance(t, ScalarT):
      assert len(elts) == 1
      return elts[0]
    elif isinstance(t, SliceT):
      assert len(elts) == 3
      start, stop, step = elts
      return self.slice_value(start, stop, step)
    elif isinstance(t, ArrayT):
      data = get_field_elts(t, elts, 'data')[0]
      shape = self.tuple(get_field_elts(t, elts, 'shape'))
      strides = self.tuple(get_field_elts(t, elts, 'strides'))
      offset = get_field_elts(t, elts, 'offset')[0]
      nelts = get_field_elts(t, elts, 'size')[0]
      return self.array_view(data, shape, strides, offset, nelts)
    elif isinstance(t, TupleT):
      assert False, "Not implemented: TupleT" 
    elif isinstance(t, ClosureT):
      assert False, "Not implemented: ClosureT" 
    elif isinstance(t, FnT):
      assert False, "Not implemented: FnT" 
  
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
  