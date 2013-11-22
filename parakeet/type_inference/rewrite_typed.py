
from .. import names, syntax 
from ..ndtypes import (IncompatibleTypes, ScalarT, 
                       Bool, Type,  ArrayT, Int64, TupleT, 
                       make_tuple_type, make_array_type)
from ..syntax import (Assign, Tuple, TupleProj, Var, Cast, Return, Index, Map, 
                      ConstArrayLike, Const)
from ..syntax.helpers import get_types, zero_i64, one_i64, none, const 
from ..transforms import Transform 


class RewriteTyped(Transform):
  def __init__(self, return_type = None):
    Transform.__init__(self, verify = False)
    self.forced_return_type = return_type 
    
    
  def pre_apply(self, fn):
    
    if self.forced_return_type is None:
      self.fn_return_type = self.fn.type_env["$return"]
    else:
      self.fn_return_type = self.forced_return_type
  
  def post_apply(self, fn):
    fn.return_type = self.fn_return_type  
  
  def coerce_expr(self, expr, t):
    assert t is not None
    expr = self.transform_expr(expr)
    if expr.type == t:
      return expr
    elif expr.__class__ is Tuple:
      if t.__class__ is not TupleT or \
          len(expr.type.elt_types) != t.elt_types:
        raise IncompatibleTypes(expr.type, t)
      else:
        new_elts = []
        for elt, elt_t in zip(expr.elts, t.elt_types):
          new_elts.append(self.coerce_expr(elt, elt_t))
        return Tuple(new_elts, type = t)
    else:
      assert \
          isinstance(expr.type, ScalarT) and \
          isinstance(t, ScalarT), \
          "Can't cast type %s into %s" % (expr.type, t)
      return syntax.Cast(expr, type=t)

  def transform_merge(self, merge):
    new_merge = {}
    for (var, (left, right)) in merge.iteritems():
      t = self.type_env[var]
      new_left = self.coerce_expr(left, t)
      new_right = self.coerce_expr(right, t)
      new_merge[var] = (new_left, new_right)
    return new_merge

  def transform_lhs(self, lhs):
    if isinstance(lhs, Var):
      t = self.lookup_type(lhs.name)
      if t == lhs.type:
        return lhs
      else:
        return Var(lhs.name, type = t)
    elif isinstance(lhs, syntax.Tuple):
      elts = map(self.transform_lhs, lhs.elts)
      elt_types = get_types(elts)
      if elt_types != lhs.type.elt_types:
        return syntax.Tuple(elts, type = make_tuple_type(elt_types))
      else:
        return lhs
    else:
      return lhs

  def transform_Var(self, expr):
    expr.type = self.fn.type_env[expr.name]
    return expr

  def transform_PrimCall(self, expr):
      arg_types = get_types(expr.args)
      upcast_types = expr.prim.expected_input_types(arg_types)
      result_type = expr.prim.result_type(upcast_types)
      upcast_args = [self.coerce_expr(x, t)
                     for (x,t) in zip(expr.args, upcast_types)]
      return syntax.PrimCall(expr.prim, upcast_args, type = result_type)

  def transform_Array(self, expr):

    array_t = expr.type
    elt_t = array_t.elt_type
    assert array_t.rank > 0
    if array_t.rank == 1:
      new_elts = [self.coerce_expr(elt, elt_t) for elt in expr.elts]
      return syntax.Array(new_elts, type = array_t)
    else:
      # need to allocate an output array and copy the elements in
      first_elt = self.assign_name(expr.elts[0], "first_elt")
      elt_dims = [self.shape(first_elt, i) for i in xrange(array_t.rank - 1)]
      n = len(expr.elts)
      outer_dim = const(n)
      all_dims = (outer_dim,) + tuple(elt_dims)
      array = self.alloc_array(elt_t, all_dims, "array_literal")
      for i, elt in enumerate(expr.elts):
        idx_expr = self.index(array, i, temp = False)
        # transform indexing to make missing indices explicit
        self.assign(idx_expr, expr.elts[i])
      return array

  def transform_Reduce(self, expr):
    acc_type = self.return_type(expr.combine)
    if expr.init and \
        not self.is_none(expr.init) and \
        expr.init.type != acc_type:
      assert len(expr.args) == 1
      expr.init = self.coerce_expr(expr.init, acc_type)
    return expr
    
  def transform_Scan(self, expr):
    acc_type = self.return_type(expr.combine)
    if expr.init and not self.is_none(expr.init) and expr.init.type != acc_type:
      if isinstance(acc_type, ScalarT):
        expr.init = self.coerce_expr(expr.init, acc_type)
      elif isinstance(expr.init.type, ScalarT) and \
           isinstance(expr.axis, Const) and \
           len(expr.args) == 1:
        arr_slice = self.slice_along_axis(expr.args[0], expr.axis, zero_i64)
        init_type = make_array_type(elt_type = expr.init.type, rank = arr_slice.type.rank)
        expr.init = ConstArrayLike(array = arr_slice, 
                                   value = expr.init, 
                                   type = init_type)
      else:
        assert False, \
          "Scan with scalar init of type %s and accumulator of type %s not yet supported" % \
                    (expr.init.type, acc_type)
    return expr

  def transform_Slice(self, expr):
    """
    # None step defaults to 1
    if isinstance(expr.step.type, core_types.NoneT):
      start_t = expr.start.type
      stop_t = expr.stop.type
      step = syntax_helpers.one_i64
      step_t = step.type
      slice_t = array_type.make_slice_type(start_t, stop_t, step_t)
      expr.step = step
      expr.type = slice_t
    """
    return expr
  

  def get_index_fn(self, array_t, idx_t,  _index_function_cache = {}):
    key = (array_t, idx_t) 
    if key in _index_function_cache:
      return _index_function_cache[key]
    array_name = names.fresh("array")
    array_var = Var(array_name, type = array_t)
    idx_name = names.fresh("idx")
    idx_var = Var(idx_name, type = idx_t)
    if idx_t is not Int64:
      idx_var = Cast(value = idx_var,  type = Int64)
    elt_t = array_t.index_type(idx_t)

    fn = syntax.TypedFn(
        name = names.fresh("fancy_indexing_helper"), 
        arg_names = (array_name, idx_name),
        input_types = (array_t, idx_t),
        return_type = elt_t,
        type_env = {array_name:array_t, idx_name:idx_t}, 
        body = [Return (Index(array_var, idx_var, type = elt_t))]) 
    _index_function_cache[key] = fn 
    return fn 
    
  def transform_Index(self, expr):
    # TODO: Make fancy indexing work 
    # with multiple indices, boolean index elements, 
    # and multi-dimensional indexing
    
    index = expr.index
    #if index.type.__class__ is TupleT:
    #  indices = index.elts 
    #else:
    #  indices = [index]
    
    if index.type.__class__ is ArrayT:
      assert index.type.rank == 1, \
        "Don't yet support indexing by %s" % index.type 
      
      index_elt_t = index.type.elt_type
      
      if index_elt_t == Bool:
        assert False, "Indexing by boolean vector not yet implemented"
        #index_array = Where(expr.index)
        #index_elt_t = Int64
      else:
        index_array = expr.index 
      index_array = expr.index 
      index_fn = self.get_index_fn(expr.value.type, index_elt_t)
      index_closure = self.closure(index_fn, [expr.value])
      return Map(fn = index_closure, 
                 args = [index_array], 
                 type = expr.value.type, 
                 axis = zero_i64)
    else:
      return expr 
  
  def transform_Assign(self, stmt):
    new_lhs = self.transform_lhs(stmt.lhs)
    lhs_t = new_lhs.type
    rhs = self.transform_expr(stmt.rhs)
    assert lhs_t is not None, "Expected a type for %s!" % stmt.lhs
    if new_lhs.__class__ is Tuple: 
      rhs = self.assign_name(rhs)
      for (i, lhs_elt) in enumerate(new_lhs.elts):
        if lhs_elt.__class__ is Var:
          name = names.original(lhs_elt.name) 
        else:
          name = None 
        idx = self.index(rhs, i, name = name )
        elt_stmt = self.transform_Assign(Assign(lhs_elt, idx))
        self.blocks.append_to_current(elt_stmt)
      return None  
    elif new_lhs.__class__ is Index and \
         isinstance(new_lhs.value.type, ArrayT) and \
         isinstance(rhs.type, ScalarT):
      new_rhs = self.coerce_expr(rhs, new_lhs.value.type.elt_type)
      assert new_rhs.type and isinstance(new_rhs.type, Type), \
          "Expected type annotation on %s, but got %s" % (new_rhs, new_rhs.type)
      
      stmt.lhs = new_lhs
      stmt.rhs = new_rhs
      return stmt 

    else:     
      new_rhs = self.coerce_expr(rhs, lhs_t)
      assert new_rhs.type and isinstance(new_rhs.type, Type), \
          "Expected type annotation on %s, but got %s" % (new_rhs, new_rhs.type)
      stmt.lhs = new_lhs
      stmt.rhs = new_rhs
      return stmt
    
  def transform_If(self, stmt):
    stmt.cond = self.coerce_expr(stmt.cond, Bool)
    stmt.true = self.transform_block(stmt.true)
    stmt.false = self.transform_block(stmt.false)
    stmt.merge = self.transform_merge(stmt.merge)
    return stmt

  def transform_Return(self, stmt):
    stmt.value = self.coerce_expr(stmt.value, self.fn_return_type)
    return stmt

  def transform_ForLoop(self, stmt):
    var_t = stmt.var.type
    stmt.start = self.coerce_expr(stmt.start, var_t)
    stmt.stop = self.coerce_expr(stmt.stop, var_t)
    stmt.step = self.coerce_expr(stmt.step, var_t)
    stmt.body = self.transform_block(stmt.body)
    stmt.merge = self.transform_merge(stmt.merge)
    return stmt 
  
  def transform_While(self, stmt):
    stmt.cond = self.coerce_expr(stmt.cond, Bool)
    stmt.body = self.transform_block(stmt.body)
    stmt.merge = self.transform_merge(stmt.merge)
    return stmt

def rewrite_typed(typed_fundef, return_type = None):
  return RewriteTyped(return_type).apply(typed_fundef)
