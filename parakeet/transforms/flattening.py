from .. import names 

from ..builder import Builder
from ..ndtypes import (ScalarT, NoneT, NoneType, ArrayT, SliceT, TupleT, make_tuple_type, 
                       Int64, PtrT, ptr_type, ClosureT, FnT, StructT, 
                       TypeValueT)
from ..syntax import (Var, Tuple, 
                      Index, TypedFn, Return, Stmt, Assign, Alloc,  
                      ParFor, PrimCall, If, While, ForLoop, Call, Expr, 
                      IndexReduce, ExprStmt) 
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
  elif isinstance(t, (NoneT, FnT, TypeValueT)):
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
    return flatten_types( (t.start_type, t.stop_type, t.step_type) )
  else:
    assert False, "Unsupported type %s" % (t,)

def flatten_types(ts):
  return concat([flatten_type(t) for t in ts])
  
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
  assert stop <= len(values), \
    "Insufficient number of flattened fields %s for %s.%s" % (values, t, field)
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
    Builder.__init__(self)
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
        return_type = new_return_type, 
        created_by = self) 
      
    
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
  
  
  #def bind_name(self):
  #def bind_var(self, lhs, rhs):
  
  def flatten_Assign(self, stmt):
    c = stmt.lhs.__class__
    rhs = self.flatten_expr(stmt.rhs)

    if c is Var:
      
      lhs_vars = self.flatten_lhs_var(stmt.lhs)
      self.var_expansions[stmt.lhs.name] = lhs_vars 
      if isinstance(rhs, Expr):
        if len(lhs_vars) == 1:
          return [Assign(lhs_vars[0], rhs)]
        else:
          # the IR doesn't allow for multiple assignment 
          # so we fake it with tuple literals
          return [Assign(self.tuple(lhs_vars), rhs)]
      assert isinstance(rhs, (list,tuple))
      assert len(lhs_vars) == len(rhs), \
        "Mismatch between LHS %s and RHS %s : %s => %s in stmt %s" % \
        (lhs_vars, stmt.rhs, stmt.rhs.type, rhs, stmt)
      result = []
      for var, value in zip(lhs_vars, rhs):
        result.append(Assign(var, value))
      return result 

    elif c is Index:
      array_t = stmt.lhs.value.type 
      if isinstance(array_t, PtrT):
        return stmt  
      indices = self.flatten_expr(stmt.lhs.index)
      values = self.flatten_expr(stmt.lhs.value)
      data = get_field_elts(array_t, values, 'data')[0]
      shape = get_field_elts(array_t, values, 'shape')
      strides = get_field_elts(array_t, values, 'strides')
      offset = get_field_elts(array_t, values, 'offset')[0]
      n_dims = len(strides)
      n_indices = len(indices)
      assert n_dims == n_indices, \
        "Expected %d indices but only got %d in %s" % (n_dims, n_indices, stmt)
      for idx, stride in zip(indices, strides):
        offset = self.add(offset, self.mul(idx, stride))

      stmt.lhs = self.index(data, offset, temp=False)
      stmt.rhs = rhs[0]
      return [stmt]
    else:
      assert False, "LHS not supported in flattening: %s" % stmt 
  
  def enter_branch(self, phi_nodes):
    for (k, (left, _)) in phi_nodes.iteritems():
      self.var_expansions[k] = self.flatten_lhs_name(k, left.type)
  
  def flatten_merge(self, phi_nodes):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      t = left.type
      assert right.type == t 
      if isinstance(t, (ScalarT, PtrT)):
        result[k] = (self.flatten_scalar_expr(left), self.flatten_scalar_expr(right))
      elif isinstance(t, (FnT, NoneT)):
        continue 
      else:
        fields = self.var_expansions[k]
        flat_left = self.flatten_expr(left)
        flat_right = self.flatten_expr(right)
        assert len(fields) == len(flat_left)
        assert len(fields) == len(flat_right)
        for i, var in enumerate(fields):
          result[var.name] = (flat_left[i], flat_right[i])
    return result 
   
  def flatten_ForLoop(self, stmt):
    self.enter_branch(stmt.merge)
    var = self.flatten_scalar_lhs_var(stmt.var)
    start = self.flatten_scalar_expr(stmt.start)
    stop = self.flatten_scalar_expr(stmt.stop)
    step = self.flatten_scalar_expr(stmt.step)
    body = self.flatten_block(stmt.body)
    merge = self.flatten_merge(stmt.merge)
    return ForLoop(var, start, stop, step, body, merge)
  
  def flatten_While(self, stmt):
    self.enter_branch(stmt.merge)
    cond = self.flatten_scalar_expr(stmt.cond)
    body = self.flatten_block(stmt.body)
    merge = self.flatten_merge(stmt.merge)
    return While(cond, body, merge)
     
  
  def flatten_If(self, stmt):
    self.enter_branch(stmt.merge)
    cond = self.flatten_scalar_expr(stmt.cond)
    true = self.flatten_block(stmt.true)
    false = self.flatten_block(stmt.false)
    merge = self.flatten_merge(stmt.merge)
    assert merge is not None
    return If(cond, true, false, merge = merge)
  
  def flatten_ExprStmt(self, stmt):
    return ExprStmt(value = self.flatten_expr(stmt.value))
   
  def flatten_ParFor(self, stmt):
    new_fn, closure_elts = self.flatten_fn(stmt.fn)
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
      return ()
    return (expr,)
  
  def flatten_fn(self, closure):
    fn = self.get_fn(closure)
    import pipeline
    fn = pipeline.indexify.apply(fn)
    flat_fn = build_flat_fn(fn)
    flat_closure_args = self.flatten_expr(closure)
    return flat_fn, flat_closure_args
  
  
  def flatten_Call(self, expr):
    flat_fn, flat_closure_args = self.flatten_fn(expr.fn)
    flat_args = self.flatten_expr_list(expr.args)
    args = tuple(flat_closure_args) + tuple(flat_args)
    return Call(flat_fn, args, type = flat_fn.return_type)
  
  def flatten_Cast(self, expr):
    return [expr]
  
  def flatten_UntypedFn(self, expr):
    return []
  
  def flatten_TypedFn(self, expr):
    return []
    
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
    result = self.flatten_field(expr.tuple, expr.index)
    return result
    
  def flatten_Closure(self, expr):
    return self.flatten_expr_list(expr.args)
    
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
    count_exprs = self.flatten_expr(expr.count)
    assert len(count_exprs) == 1
    return Alloc(count = count_exprs[0], elt_type = expr.elt_type, type = expr.type)
    
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
    
    index = expr.index 
    if isinstance(index.type, (NoneT, SliceT, ScalarT)):
      indices = [index]
    elif isinstance(index, Tuple):
      indices = index.elts 
    else:
      assert isinstance(index.type, TupleT), "Expected index to scalar, slice, or tuple"
      indices = self.tuple_elts(index)


      
    #indices = self.flatten_expr(expr.index)
   
    n_indices = len(indices)
    n_strides = len(strides)
    assert n_indices == n_strides, \
      "Not supported:  indices vs. dimensions: %d != %d in %s" % (n_indices, n_strides, expr)


    # fast-path for the common case when we're indexing
    # by all scalars to retrieve a scalar result
    #if syntax.helpers.all_scalars(indices):
    for i, idx in enumerate(indices):
      offset = self.add(offset, self.mul(idx, strides[i]))
    return [self.index(data_fields[0], offset)]
     
  
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
    data = self.flatten_expr(expr.data)
    shape = self.flatten_expr(expr.shape)
    strides = self.flatten_expr(expr.strides)
    offset = self.flatten_expr(expr.offset)
    size = self.flatten_expr(expr.size)
    return data + shape + strides + offset + size
    
  
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
    assert False, "Unexpected Map encountered during flattening, should be IndexScan"
  
  def flatten_Reduce(self, expr):
    assert False, "Unexpected Reduce encountered during flattening, should be IndexScan"
  
  def flatten_Scan(self, expr):
    assert False, "Unexpected Scan encountered during flattening, should be IndexScan"
  
  def flatten_IndexMap(self, expr):
    #fn, closure_args = self.flatten_fn(expr.fn)
    #shape_elts = self.flatten_expr(expr.shape)
    # import pdb; pdb.set_trace()
    #return IndexMap(fn = self.closure(fn, closure_args), shape = self.tuple(shape_elts), type = None)
    assert False, "Unexpected IndexMap, should have been turned into ParFor before flattening"
    
  
  def flatten_OuterMap(self, expr):
    assert False, "Unexpected OuterMap, should have been turned into ParFor before flattening"
  
  def flatten_IndexReduce(self, expr):
    # assert isinstance(expr.type, ScalarT), "Non-scalar reductions not yet implemented"
    fn, fn_args = self.flatten_fn(expr.fn)
    fn = self.closure(fn, fn_args)
    combine, combine_args = self.flatten_fn(expr.combine)
    combine = self.closure(combine, combine_args)
    shape = self.tuple(self.flatten_expr(expr.shape))
    
    init = self.flatten_expr(expr.init)
    t = flatten_type(expr.type)
    if len(t) == 1:
      init = init[0]
      t = t[0]
    else:
      init = self.tuple(init)
      t = make_tuple_type(t)
    
    result =  IndexReduce(fn = fn, combine = combine, shape = shape, type = t, init = init)
    return [result]
     
      
  def flatten_IndexScan(self, expr):
    assert False, "IndexScan Not implemented" 
  
  def flatten_lhs_name(self, name, t):
    if isinstance(t, (ScalarT, PtrT)):
      return [Var(name = name, type = t)]
    elif isinstance(t, (NoneT, FnT, TypeValueT)):
      return []
    elif isinstance(t, SliceT):
      base = name.replace(".", "_")
      start = Var(name = "%s_start" % base, type = t.start_type)
      stop = Var(name = "%s_stop" % base, type = t.stop_type)
      step = Var(name = "%s_step" % base, type = t.step_type)
      field_vars = [start, stop, step]
    elif isinstance(t, ClosureT):
      base = name.replace(".", "_")
      field_vars = [Var(name = "%s_closure_elt%d" % (base,i) , type = t) 
                    for i,t in enumerate(t.arg_types)]
    elif isinstance(t, TupleT):
      base = name.replace(".", "_")
      field_vars = [Var(name = "%s_elt%d" % (base, i), type = t) 
                    for i,t in enumerate(t.elt_types)]
    elif isinstance(t, ArrayT):
      base = name.replace(".", "_")
      data = Var(name = "%s_data" % base, type = t.ptr_t)
      shape = Var(name = "%s_shape" % base, type = t.shape_t)
      strides = Var(name = "%s_strides" % base, type = t.strides_t)
      offset = Var(name = "%s_offset" % base, type = Int64)
      nelts = Var(name = "%s_nelts" % base, type = Int64)
      field_vars = [data, shape, strides, offset, nelts]
    else:
      assert False, "Unsupport type %s" % (t,)
    return self.flatten_lhs_vars(field_vars)
  
  def flatten_lhs_var(self, old_var):
    t = old_var.type 
    if isinstance(t, (PtrT, ScalarT)):
      return [old_var]
    name = old_var.name 
    return self.flatten_lhs_name(name, t)
  
  def flatten_lhs_vars(self, old_vars):
    return concat_map(self.flatten_lhs_var, old_vars)
  
  def flatten_scalar_lhs_var(self, old_var):
    lhs_vars = self.flatten_lhs_var(old_var)
    assert len(lhs_vars) == 1
    return lhs_vars[0]
  
def build_flat_fn(old_fn, _cache = {}):
  key = old_fn.cache_key
  if key in _cache:
    return _cache[key]
  flat_fn = BuildFlatFn(old_fn).run()
  _cache[key] = flat_fn
  _cache[flat_fn.cache_key] = flat_fn
  return flat_fn
  
class Flatten(Transform):
  
  def unbox_var(self, var):
    t = var.type 

    if isinstance(t, (FnT, NoneT, TypeValueT)):
      return []
    elif isinstance(t, (PtrT, ScalarT)):
      return [var]
    elif isinstance(t, ArrayT):
      # for structured arrays, should this be a tuple? 
      base = var.name.replace(".", "_")
      data = self.attr(var, 'data', name = base + "_data")
      shape = self.attr(var, 'shape', name = base + "_shape")
      strides = self.attr(var, 'strides', name = base + "_strides")
      offset = self.attr(var, 'offset', name = base + "_offset")
      size = self.attr(var, 'size', name = base + "_size")
      return self.unbox_vars([data, shape, strides, offset, size])
    elif isinstance(t, SliceT):
      start = self.attr(var, 'start')
      stop = self.attr(var, 'stop')
      step = self.attr(var, 'step')
      return self.unbox_vars([start, stop, step])
    elif isinstance(t, ClosureT):
      base = var.name.replace(".", "_")
      closure_elts = [self.closure_elt(var, i, name = base + "_closure_elt%d" % i)
                      for i in xrange(len(t.arg_types))]
      return self.unbox_vars(closure_elts)
    elif isinstance(t, TupleT):
      base = var.name.replace(".", "_")
      tuple_elts = [self.assign_name(self.tuple_proj(var, i), name = base + "_elt%d" % i)
                      for i in xrange(len(t.elt_types))]
      return self.unbox_vars(tuple_elts)
    else:
      assert False, "Unsupported type %s" % (t,)
  
  def unbox_vars(self, exprs):
    return concat_map(self.unbox_var, exprs)
  
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
      boxed_elts = []
      for i, elt_t in enumerate(t.elt_types):
        elt = self.box(elt_t, get_field_elts(t, elts, i))
        boxed_elts.append(elt)
      return self.tuple(boxed_elts)
    elif isinstance(t, ClosureT):
      assert False, "Not implemented: ClosureT" 
    elif isinstance(t, FnT):
      assert False, "Not implemented: FnT" 
  
  def transform_block(self, stmts):
    return stmts
    
  def pre_apply(self, old_fn, _cache = {}):

    key = old_fn.cache_key
    if key  in _cache:
      return _cache[key]
    
    flat_fn = build_flat_fn(old_fn)
    
    flat_fn.created_by = old_fn.created_by
    flat_fn.transform_history = old_fn.transform_history.copy()
    
    input_vars = mk_vars(old_fn.arg_names, old_fn.input_types)
    self.blocks.push()
    unboxed_inputs = self.unbox_vars(input_vars)
    assert len(unboxed_inputs) == len(flat_fn.input_types)
    unboxed_result = self.call(flat_fn, unboxed_inputs, name = "unboxed_result")
    unboxed_elts = self.to_seq(unboxed_result)
    boxed_result = self.box(old_fn.return_type, unboxed_elts)
    self.return_(boxed_result)
    old_fn.body = self.blocks.pop()
    _cache[key] = old_fn
    _cache[flat_fn.cache_key] = flat_fn 
    return old_fn
  