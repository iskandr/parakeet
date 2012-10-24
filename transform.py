import syntax 
import names 
import core_types 
import tuple_type 
import prims 

from typed_syntax_helpers import get_type, get_types, wrap_constant, wrap_constants


class NestedBlocks:
  def __init__(self):
    self.blocks = []
     
  
  def push(self):
    self.blocks.append([])
  
  def pop(self):
    return self.blocks.pop()
  
  def current(self):
    return self.blocks[-1]
  
  def append_to_current(self, stmt):
    self.current().append(stmt)
  
  def extend_current(self, stmts):
    self.current().extend(stmts)


class Transform:
  def __init__(self, fn):
    self.blocks = NestedBlocks()
    self.fn = fn
    self.type_env = None 
  
  
  def lookup_type(self, name):
    assert self.type_env is not None
  
  def fresh_var(self, t, prefix = "temp"):
    assert t is not None, "Type required for new variable %s" % prefix
    ssa_id = names.fresh(prefix)
    self.type_env[ssa_id] = t
    return syntax.Var(ssa_id, type = t)
  
  
  
  def insert_stmt(self, stmt):
    self.blocks.append_to_current(stmt)
  
  def insert_assign(self, lhs, rhs):
    self.insert_stmt(syntax.Assign(lhs, rhs))
  
  def assign_temp(self, expr, name = "temp"):
    var = self.fresh_var(expr.type, name)
    self.insert_assign(var, expr)
    return var 
  
  def cast(self, expr, t):
    assert isinstance(t, core_types.ScalarT), "Casts not yet implemented for non-scalar types"
    return self.assign_temp(syntax.Cast(expr, type = t), "cast_%s" % t)
     
  
  
  def get_struct_field(self, struct, attr_name):
    assert isinstance(struct.type, core_types.StructT)
    field_type = struct.type.field_type(attr_name)
    return self.assign_temp(syntax.Attribute(struct, attr_name, type = field_type), attr_name)
  
  def get_index(self, arr, idx):
    print "get_index", arr, idx
    idx = wrap_constant(idx)
    arr_t = arr.type 
    if isinstance(arr_t, tuple_type.TupleT):
      if isinstance(idx, syntax.Const):
        idx = idx.value
      assert isinstance(idx, (int, long))
      t = arr_t.elt_types[idx]
      return self.assign_temp(syntax.TupleProj(arr, idx, type = t), "tuple_elt%d" % idx)
    else:
      t = arr_t.index_type(idx.type)
      return self.assign_temp(syntax.Index(arr, idx, type = t), "array_elt")
      
  
  def prim(self, prim_fn, args, name = "temp"):
    args = wrap_constants(args)
    arg_types = get_types(args)
    upcast_types = prim_fn.expected_input_types(arg_types)
    result_type = prim_fn.result_type(upcast_types)
    upcast_args = [self.cast(x, t) for (x,t) in zip(args, upcast_types)]
    
    prim_call = syntax.PrimCall(prim_fn, upcast_args, type = result_type)
    print 
    print "prim_fn", prim_fn 
    print "original_args", args 
    print "upcast_args", upcast_args 
    return self.assign_temp(prim_call, name)
    
  def add(self, x, y, name = "add"):
    return self.prim(prims.add, [x,y], name)
  
  def sub(self, x, y, name = "sub"):
    return self.prim(prims.subtract, [x,y], name)
 
  def mul(self, x, y, name = "mul"):
    return self.prim(prims.multiply, [x,y], name)
  
  def div(self, x, y, name = "div"):
    return self.prim(prims.divide, [x,y], name)
    
  #def cast(expr, t):
  #  assert isinstance(t, core_types.ScalarT), "Casts not yet implemented for non-scalar types"
  #  return typed_ast.Cast(expr, type = t)    
  
  def transform_generic_expr(self, expr):
    args = {}
    for member_name in expr.members:
      member_value = getattr(expr, member_name)
      if isinstance(member_value, syntax.Expr):
        member_value = self.transform_expr(member_value)
      args[member_name] = member_value
    return expr.__class__(**args)
  
  def transform_expr(self, expr):
    """
    Dispatch on the node type and call the appropriate transform method
    """
    method_name = "transform_" + expr.node_type()
    if hasattr(self, method_name):
      result = getattr(self, method_name)(expr)
    else:
      result = self.transform_generic_expr(expr)
    assert result.type is not None, "Missing type for %s" % result 
    return result 
  
  def transform_expr_list(self, exprs):
    return [self.transform_expr(e) for e in exprs]
  
  def transform_phi_nodes(self, phi_nodes):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      new_left = self.transform_expr(left)
      new_right = self.transform_expr(right)
      result[k] = new_left, new_right 
    return result 
  
  def transform_lhs(self, lhs):
    """
    Overload this is you want different behavior
    for transformation of left-hand side of assignments
    """ 
    
    return self.transform_expr(lhs)
  
  def transform_Assign(self, stmt):
    # TODO: flatten tuple assignment ptype
    #assert isinstance(stmt.lhs, (str, syntax.Var)), \
    #  "Pattern-matching assignment not implemented" 
    rhs = self.transform_expr(stmt.rhs)
    lhs = self.transform_lhs(stmt.lhs) 
    # if isinstance(lhs, syntax.Var):
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
    merge_before = self.transform_phi_nodes(stmt.merge_before)
    merge_after = self.transform_phi_nodes(stmt.merge_after)
    return syntax.While(cond, body, merge_before, merge_after)
  
  def transform_stmt(self, stmt):
    method_name = "transform_" + stmt.node_type()
    if hasattr(self, method_name):
      result = getattr(self, method_name)(stmt)
    assert isinstance(result, syntax.Stmt), \
      "Expected statement: %s" % result 
    return result 
  
  def transform_block(self, stmts):
    self.blocks.push()
    for old_stmt in stmts:
      new_stmt = self.transform_stmt(old_stmt)
      self.blocks.append_to_current(new_stmt)
    return self.blocks.pop() 
  
  def apply(self):
    
    self.type_env = self.fn.type_env.copy()
    body = self.transform_block(self.fn.body)
    new_fundef_args = dict([ (m, getattr(self.fn, m)) for m in self.fn._members])
    new_fundef_args['body'] = body
    new_fundef_args['type_env'] = self.type_env 
    return syntax.TypedFn(**new_fundef_args)
  

def apply_pipeline(fn, transforms):
  for T in transforms:
    fn = T(fn).apply()
  return fn 
