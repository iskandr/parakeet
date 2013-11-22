from .. syntax import Expr, Var
from clone_function import CloneFunction
from transform import Transform

class RewriteVars(Transform):
  def __init__(self, rename_dict):
    Transform.__init__(self, require_types = False)
    self.rename_dict = rename_dict

  def transform_merge(self, old_merge):
    new_merge = {}
    for (k,(l,r)) in old_merge.iteritems():
      new_name = self.rename_dict.get(k,k)
      if type(new_name) != str:
        assert new_name.__class__ is Var, \
            "Unexpected substitution %s for %s" % (new_name, k)
        new_name = new_name.name
      new_left = self.transform_expr(l)
      new_right = self.transform_expr(r)
      new_merge[new_name] = new_left, new_right
    return new_merge
  
  def transform_Var(self, expr):
    new_value = self.rename_dict.get(expr.name, expr.name)
    if new_value.__class__ is str:
      if new_value != expr.name:
        expr.name = new_value
      return expr
    else:
      assert isinstance(expr, Expr)
      assert new_value.type is not None, \
          "Type of replacement value %s can't be None" % new_value
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
