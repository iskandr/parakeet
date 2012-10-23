import syntax as typed_ast 
from common import dispatch 
import closure_signatures
import core_types 

def create_simple_transform(transform_expr):
  def transform_phi_nodes(phi_nodes):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      result[k] = (transform_expr(left), transform_expr(right))
    return result 
  
  def transform_stmt(stmt):
    
    def transform_Assign():
      # TODO: flatten tuple assignment ptype
      assert isinstance(stmt.lhs, (str, typed_ast.Var)), "Pattern-matching assignment not implemented" 
      rhs = transform_expr(stmt.rhs)
      return typed_ast.Assign(stmt.lhs, rhs)
    
    def transform_Return():
      return typed_ast.Return(transform_expr(stmt.value))
    
    def transform_If():
      cond = transform_expr(stmt.cond)
      true = transform_block(stmt.true)
      false = transform_block(stmt.false)
      merge = transform_phi_nodes(stmt.merge)      
      return typed_ast.If(cond, true, false, merge) 
    
    def transform_While():
      cond = transform_expr(stmt.cond)
      body = transform_block(stmt.body)
      merge_before = transform_phi_nodes(stmt.merge_before)
      merge_after = transform_phi_nodes(stmt.merge_after)
      return typed_ast.While(cond, body, merge_before, merge_after)
    
    return dispatch(stmt, "transform")
    
  def transform_block(block):
    return [transform_stmt(s) for s in block]
  
  return transform_block 
  

def make_structs_explicit(fundef):
  
  def transform_expr(expr):
    def transform_Tuple():
      struct_args = map(transform_expr, expr.elts)
      return typed_ast.Struct(struct_args, type = expr.type)
    
    def transform_Closure():
      closure_args = map(transform_expr, expr.args)
      closure_id = closure_signatures.get_id(expr.type)
      closure_id_node = typed_ast.Const(closure_id, type = core_types.Int64)
      return typed_ast.Struct([closure_id_node] + closure_args, type = expr.type)
    
    
    def transform_Invoke():
      new_closure = transform_expr(expr.closure)
      new_args =  map(transform_expr, expr.args) 
      return typed_ast.Invoke(new_closure, new_args, type = expr.type) 
    
    def transform_TupleProj():
      new_tuple = transform_expr(expr.tuple)
      assert isinstance(expr.index, int)
      tuple_t = expr.tuple.type

      field_name, field_type  = tuple_t._fields_[expr.index]
      return typed_ast.Attribute(new_tuple, field_name, type = field_type)
       
    return dispatch(expr, 'transform', default = lambda expr: expr)
  
  transform_block = create_simple_transform(transform_expr)
  body = transform_block(fundef.body)
  new_fundef_args = dict([ (m, getattr(fundef,m)) for m in fundef._members])
  new_fundef_args['body'] = body
  return typed_ast.TypedFn(**new_fundef_args)
  