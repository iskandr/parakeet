import syntax 
import names 
import core_types 
import tuple_type
import array_type  
import prims 

from syntax_helpers import get_type, get_types, wrap_if_constant, \
                           wrap_constants, zero

import function_registry
import syntax_helpers

from nested_blocks import NestedBlocks

class Transform(object):
  def __init__(self, fn):
    self.blocks = NestedBlocks()
    self.type_env = None 
    self.fn = fn

  def lookup_type(self, name):
    assert self.type_env is not None
  
  def fresh_var(self, t, prefix = "temp"):
    assert t is not None, "Type required for new variable %s" % prefix
    ssa_id = names.fresh(prefix)
    self.type_env[ssa_id] = t
    return syntax.Var(ssa_id, type = t)
  
  def fresh_i32(self, prefix = "temp"):
    return self.fresh_var(core_types.Int32, prefix)
  
  def fresh_i64(self, prefix = "temp"):
    return self.fresh_var(core_types.Int64, prefix)
  
  def insert_stmt(self, stmt):
    self.blocks.append_to_current(stmt)
  
  def assign(self, lhs, rhs, recursive = False):
    if recursive:
      rhs = self.transform_expr(rhs)
      lhs = self.transform_lhs(lhs)
    stmt = syntax.Assign(lhs, rhs)
    if recursive and \
       self.transform_Assign.im_func != Transform.transform_Assign.im_func:
      stmt = self.transform_Assign(stmt)
    self.insert_stmt(stmt)
  
  def assign_temp(self, expr, name = "temp"):
    if isinstance(expr, syntax.Var):
      return expr
    else:
      var = self.fresh_var(expr.type, name)
      self.assign(var, expr)
      return var 
    
  def zero(self, t = core_types.Int32, name = "counter"):
    return self.assign_temp(zero(t), name)
  
  def zero_i32(self, name = "counter"):
    return self.zero(t = core_types.Int32, name = name)
  
  def zero_i64(self, name = "counter"):
    return self.zero(t = core_types.Int64, name = name)
  
  def cast(self, expr, t):
    assert isinstance(t, core_types.ScalarT), "Casts not yet implemented for non-scalar types"
    if expr.type == t:
      return expr
    else:
      return self.assign_temp(syntax.Cast(expr, type = t), "cast_%s" % t)
  
  def index(self, arr, idx, temp = True):
    """
    Index into array or tuple differently depending on the type
    """
    n_required = arr.type.rank 
    n_indices = len(idx) if hasattr(idx, '__len__') else 1
  
    if n_indices < n_required:
      # all unspecified dimensions are considered fully sliced 
      extra = (None,) * (n_required - n_indices)
      first_indices = tuple(idx) if hasattr(idx, '__iter__') else (idx,)
      idx = first_indices + extra
      
    if isinstance(idx, tuple):
      idx = self.tuple(map(wrap_if_constant,idx), "index_tuple")
    else:
      idx = wrap_if_constant(idx)
    arr_t = arr.type 
    if isinstance(arr_t, core_types.ScalarT):
      # even though it's not correct externally, it's 
      # often more convenient to treat indexing 
      # into scalars as the identity function. 
      # Just be sure to catch this as an error in 
      # the user's code earlier in the pipeline. 
      return arr
    elif isinstance(arr_t, tuple_type.TupleT):
      if isinstance(idx, syntax.Const):
        idx = idx.value
      proj = self.tuple_proj(arr, idx)
      if temp:
        return self.assign_temp(proj, "tuple_elt%d" % idx)
      else:
        return proj 
    else:
      t = arr_t.index_type(idx.type)
      idx_expr = syntax.Index(arr, idx, type = t)
      if temp:
        return self.assign_temp(idx_expr, "array_elt")
      else:
        return idx_expr
      
  def tuple_proj(self, tup, idx):
    assert isinstance(idx, (int, long))
    t = tup.type.elt_types[idx]
    return syntax.TupleProj(tup, idx, type = t)
        
  def prim(self, prim_fn, args, name = None):
    args = wrap_constants(args)
    arg_types = get_types(args)
    upcast_types = prim_fn.expected_input_types(arg_types)
    result_type = prim_fn.result_type(upcast_types)
    upcast_args = [self.cast(x, t) for (x,t) in zip(args, upcast_types)]
    prim_call = syntax.PrimCall(prim_fn, upcast_args, type = result_type)
    if name:
      return self.assign_temp(prim_call, name)
    else:
      return prim_call
    
  def add(self, x, y, name = None):
    return self.prim(prims.add, [x,y], name)
  
  def sub(self, x, y, name = None):
    return self.prim(prims.subtract, [x,y], name)
 
  def mul(self, x, y, name = None):
    return self.prim(prims.multiply, [x,y], name)
  
  def div(self, x, y, name = None):
    return self.prim(prims.divide, [x,y], name)
    
  def lt(self, x, y, name = None):
    return self.prim(prims.less, [x,y], name)

  def lte(self, x, y, name = None):
    return self.prim(prims.less_equal, [x,y], name)

  def gt(self, x, y, name = None):
    return self.prim(prims.greater, [x,y], name)

  def gte(self, x, y, name = None):
    return self.prim(prims.greater_equal, [x,y], name)

  def eq(self, x, y, name = None):
    return self.prim(prims.equal, [x,y], name)
  
  def neq(self, x, y, name = None):  
    return self.prim(prims.not_equal, [x,y], name)
  
  def attr(self, obj, field, name = None):
    if name is None:
      name = field
    obj_t = obj.type 
    assert isinstance(obj_t, core_types.StructT), \
      "Can't get attribute '%s' from type %s" % (field, obj_t)
    field_t = obj.type.field_type(field)
    attr_expr = syntax.Attribute(obj, field, type = field_t)
    if name:
      return self.assign_temp(attr_expr, name)
    else:
      return attr_expr
  
  def shape(self, array, dim = None):
    assert isinstance(array.type, array_type.ArrayT)
    shape = self.attr(array, "shape")
    if dim is None:
      return shape
    else:
      dim_t = shape.type.elt_types[dim]
      dim_value = syntax.TupleProj(shape, dim, type = dim_t)
      return self.assign_temp(dim_value, "dim%d" % dim) 
  
  def strides(self, array, dim = None):
    assert isinstance(array.type, array_type.ArrayT)
    strides = self.attr(array, "strides")
    if dim is None:
      return strides
    else:
      elt_t = strides.type.elt_types[dim]
      elt_value = syntax.TupleProj(strides, dim, type = elt_t)
      return self.assign_temp(elt_value, "stride%d" % dim) 
  
  def tuple(self, elts, name = "tuple"):
    tuple_t = tuple_type.make_tuple_type(get_types(elts))
    tuple_expr = syntax.Tuple(elts, type = tuple_t)
    if name:
      return self.assign_temp(tuple_expr, name)
    else:
      return tuple_expr
  
  def alloc_array(self, elt_t, dims, name = "temp_array"):
    if not isinstance(dims, (list, tuple)):
      dims = [dims]
    rank = len(dims)  
    nelts = dims[0]
    for d in dims[1:]:
      nelts = self.mul(nelts, d, "nelts")
        
    array_t = array_type.make_array_type(elt_t, rank)
    ptr_t = core_types.ptr_type(elt_t)

    ptr_var = self.assign_temp(syntax.Alloc(elt_t, nelts, type = ptr_t), "data_ptr")
    shape = self.tuple( dims, "shape")
    stride_elts = [syntax_helpers.const(1)]

    # assume row-major for now!
    for d in reversed(dims[1:]):
      next_stride = self.mul(stride_elts[0], d, "dim")
      stride_elts = [next_stride] + stride_elts
      
    strides = self.tuple(stride_elts, "strides")
    array = syntax.Struct([ptr_var, shape, strides], type = array_t) 
    return self.assign_temp(array, name) 
    
  def transform_if_expr(self, maybe_expr):
    if isinstance(maybe_expr, syntax.Expr):
      return self.transform_expr(maybe_expr)
    elif isinstance(maybe_expr, tuple):
      return tuple([self.transform_if_expr(x) for x in maybe_expr])
    elif isinstance(maybe_expr, list):
      return [self.transform_if_expr(x) for x in maybe_expr]
    else:
      return maybe_expr
  
  def transform_generic_expr(self, expr):
    args = {}
    for member_name in expr.members:
      member_value = getattr(expr, member_name)
      args[member_name] = self.transform_if_expr(member_value)
    return expr.__class__(**args)
  
  def find_method(self, expr, prefix = "transform_"):
    method_name = prefix + expr.node_type()
    if hasattr(self, method_name):
      return getattr(self, method_name)
    else:
      return None
  
  
  def transform_expr(self, expr):
    """
    Dispatch on the node type and call the appropriate transform method
    """
    method = self.find_method(expr, "transform_")
    if method:
      result = method(expr)
    else:
      result = self.transform_generic_expr(expr)
    assert result.type is not None, "Missing type for %s" % result 
    return result 
  
  def transform_lhs(self, lhs):
    """
    Overload this is you want different behavior
    for transformation of left-hand side of assignments
    """ 
    lhs_method = self.find_method(lhs, prefix = "transform_lhs_")
    if lhs_method:
      return lhs_method(lhs)
    
    method = self.find_method(lhs, prefix = "transform_")
    if method:
      return method(lhs)
    
    return self.transform_expr(lhs)
  
  def transform_expr_list(self, exprs):
    return [self.transform_expr(e) for e in exprs]
  
  def transform_phi_nodes(self, phi_nodes):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      new_left = self.transform_expr(left)
      new_right = self.transform_expr(right)
      result[k] = new_left, new_right 
    return result 
  
 
  
  def transform_Assign(self, stmt):
    rhs = self.transform_expr(stmt.rhs)
    lhs = self.transform_lhs(stmt.lhs) 
    return syntax.Assign(lhs, rhs)
    
  def transform_Return(self, stmt):
    return syntax.Return(self.transform_expr(stmt.value))
    
  def transform_If(self, stmt):
    cond = self.transform_expr(stmt.cond)
    true = self.transform_block(stmt.true)
    false = self.transform_block(stmt.false)
    merge = self.transform_phi_nodes(stmt.merge)      
    return syntax.If(cond, true, false, merge) 
    
  def transform_While(self, stmt):
    cond = self.transform_expr(stmt.cond)
    body = self.transform_block(stmt.body)
    merge = self.transform_phi_nodes(stmt.merge)
    return syntax.While(cond, body, merge)
  
  def transform_stmt(self, stmt):
    method_name = "transform_" + stmt.node_type()
    if hasattr(self, method_name):
      result = getattr(self, method_name)(stmt)
    import types
    assert isinstance(result, (syntax.Stmt, types.NoneType)), \
      "Expected statement: %s" % result 
    return result 
  
  def transform_block(self, stmts):
    self.blocks.push()
    for old_stmt in stmts:
      new_stmt = self.transform_stmt(old_stmt)
      if new_stmt:
        self.blocks.append_to_current(new_stmt)
    return self.blocks.pop() 
  
  def apply(self):
    
    if isinstance(self.fn, syntax.TypedFn): 
      self.type_env = self.fn.type_env.copy()
    else:
      self.type_env = {}
    body = self.transform_block(self.fn.body)
    new_fundef_args = dict([ (m, getattr(self.fn, m)) for m in self.fn._members])
    # create a fresh function with a distinct name and the 
    # transformed body and type environment 
    new_fundef_args['name'] = names.refresh(self.fn.name)
    new_fundef_args['body'] = body
    new_fundef_args['type_env'] = self.type_env 
    new_fundef = syntax.TypedFn(**new_fundef_args)
    # register this function so if anyone tries to call it they'll be
    # able to find its definition later 
    function_registry.typed_functions[new_fundef.name] = new_fundef 
    return new_fundef 


_transform_cache = {}
def cached_apply(T, fn):
  """
  Applies the transformation, caches the result,
  and registers the new function in the global registry  
  """
  key = (T, fn.name)
  if key in _transform_cache:
    return _transform_cache[key]
  else:
    new_fn = T(fn).apply()
    _transform_cache[key] = new_fn
    return new_fn
  
def apply_pipeline(fn, transforms):
  for T in transforms:
    fn = cached_apply(T, fn) 
  return fn 
