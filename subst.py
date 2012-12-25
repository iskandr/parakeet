from syntax import Expr 
from transform import Transform 
from clone_function import CloneFunction

class RewriteVars(Transform):
  def __init__(self, rename_dict):
    Transform.__init__(self, require_types = False)
    self.rename_dict = rename_dict 
  
  def transform_merge(self, old_merge):
    new_merge = {}
    for (k,(l,r)) in old_merge.iteritems():
      new_name = self.rename_dict.get(k,k)
      new_left = self.transform_expr(l)
      new_right = self.transform_expr(r)
      new_merge[new_name] = new_left, new_right 
    return new_merge 
  
  def transform_Var(self, expr):
    new_value = self.rename_dict.get(expr.name, expr.name)
    if isinstance(new_value, str):
      if new_value != expr.name:
        expr.name = new_value  
      return expr 
    else:
      assert isinstance(expr, Expr)    
      assert new_value.type is not None, \
        "Type of replacement value %s can't be None" % new_value 
      assert new_value.type == expr.type, \
        "Can't replace %s with %s since it changes type %s intp %s" % \
        (expr, new_value, expr.type, new_value.type)
      return new_value

def subst_expr(expr, rename_dict):
  fresh_expr = CloneFunction().transform_expr(expr)
  return RewriteVars(rename_dict).transform_expr(fresh_expr)

def subst_expr_list(nodes, rename_dict):
  return [subst_expr(node, rename_dict) for node in nodes]

def subst_expr_tuple(elts, rename_dict):
  return tuple(subst_expr_list(elts, rename_dict))

def subst_stmt_list(stmts, rename_dict):
  fresh_stmts = CloneFunction().transform_block(stmts)
  return RewriteVars(rename_dict).transform_block(fresh_stmts)

