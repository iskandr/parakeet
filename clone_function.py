import names 
import syntax 
from syntax import TypedFn, Var, Const, Tuple, Attribute, Index 
from syntax import Assign, Return, If, While  
from syntax import Attribute
from transform import Transform 

class CloneFunction(Transform):
  """
  Copy all the objects in the AST of a function
  """
  
  def transform_expr(self, expr):
    c = expr.__class__
    if c is Var:
      return syntax.Var(expr.name, type = expr.type)
    elif c is Const:
      return syntax.Const(expr.value, type = expr.type)
    elif c is Tuple:
      new_elts = tuple(self.transform_expr(elt) for elt in expr.elts)
      return syntax.Tuple(elts = new_elts, type = expr.type)
    elif c is Attribute:
      value = self.transform_expr(expr.value)
      return Attribute(value, expr.name, type = expr.type)
    elif c is Index:
      value = self.transform_expr(expr.value)
      index = self.transform_expr(expr.index)
      return syntax.Index(value, index, type = expr.type)
    elif c is TypedFn:
      return expr 
    else:
      args = {}
      for member_name in expr.members():
        old_value = getattr(expr, member_name)
        new_value = self.transform_if_expr(old_value)
        args[member_name] = new_value
      return expr.__class__(**args)
  
  
  def transform_Assign(self, stmt):
    new_lhs = self.transform_expr(stmt.lhs)
    new_rhs = self.transform_expr(stmt.rhs)
    return syntax.Assign(new_lhs, new_rhs)

  def tranfsorm_Return(self, stmt):
    return syntax.Return(self.transform_expr(stmt.value))
  
  def transform_If(self, stmt):
    new_true = self.transform_block(stmt.true)
    new_false = self.transform_block(stmt.false)
    new_merge = self.transform_merge(stmt.merge)
    new_cond = self.transform_expr(stmt.cond)
    return syntax.If(new_cond, new_true, new_false, new_merge)
  
  def transform_While(self, stmt):
    new_body = self.transform_block(stmt.body)
    new_merge = self.transform_merge(stmt.merge)
    new_cond = self.transform_expr(stmt.cond)
    return syntax.While(new_cond, new_body, new_merge)
  
  
  def pre_apply(self, old_fn):
    new_fundef_args = dict([(m, getattr(old_fn, m)) for m in old_fn._members])
    # create a fresh function with a distinct name and the
    # transformed body and type environment
    new_fundef_args['name'] = names.refresh(self.fn.name)
    new_fundef_args['type_env'] = old_fn.type_env.copy()
    # don't need to set a new body block since we're assuming 
    # that transform_block will at least allocate a new list 
    new_fundef = syntax.TypedFn(**new_fundef_args)
    return new_fundef 

