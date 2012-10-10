import syntax
from common import dispatch 
import closure_signatures
import ptype 

def create_simple_transform(transform_expr):
  def transform_phi_nodes(phi_nodes):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      result[k] = (transform_expr(left), transform_expr(right))
    return result 
  
  def transform_stmt(stmt):
    
    def transform_Assign():
      # TODO: flatten tuple assignment
      assert isinstance(stmt.lhs, (str, syntax.Var)), "Pattern-matching assignment not implemented" 
      rhs = transform_expr(stmt.rhs)
      return syntax.Assign(stmt.lhs, rhs)
    
    def transform_Return():
      return syntax.Return(transform_expr(stmt.value))
    
    def transform_If():
      cond = transform_expr(stmt.cond)
      true = transform_block(stmt.true)
      false = transform_block(stmt.false)
      merge = transform_phi_nodes(stmt.merge)      
      return syntax.If(cond, true, false, merge) 
    
    def transform_While():
      cond = transform_expr(stmt.cond)
      body = transform_block(stmt.body)
      merge_before = transform_phi_nodes(stmt.merge_before)
      merge_after = transform_phi_nodes(stmt.merge_after)
      return syntax.While(cond, body, merge_before, merge_after)
    
    return dispatch(stmt, "transform")
    
  def transform_block(block):
    return [transform_stmt(s) for s in block]
  
  return transform_block 
  

def make_structs_explicit(fundef):
  
  def transform_expr(expr):
    def transform_Tuple():
      struct_args = map(transform_expr, expr.elts)
      return syntax.Struct(struct_args, type = expr.type)
    
    def transform_Closure():
      closure_args = map(transform_expr, expr.args)
      closure_id = closure_signatures.get_id(expr.type)
      closure_id_node = syntax.Const(closure_id, type = ptype.Int64)
      return syntax.Struct([closure_id_node] + closure_args, type = expr.type)
    
    return dispatch(expr, 'transform', default = lambda expr: expr)
  
  transform_block = create_simple_transform(transform_expr)
  body = transform_block(fundef.body)
  import copy 
  fundef2 = copy.deepcopy(fundef)
  fundef2.body = body 
  return fundef2 