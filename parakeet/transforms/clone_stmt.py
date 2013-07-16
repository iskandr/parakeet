from .. import names 
from .. analysis.collect_vars import  collect_binding_names
from .. syntax import Var, Assign, ForLoop
from clone_function import CloneFunction
from transform import Transform

class CloneStmt(CloneFunction):
  def __init__(self, outer_type_env):
    Transform.__init__(self)
    self.recursive = False
    self.type_env = outer_type_env
    self.rename_dict = {}

  def rename(self, old_name):
    old_type = self.type_env[old_name]
    new_name = names.refresh(old_name)
    new_var = Var(new_name, old_type)
    self.rename_dict[old_name] = new_var
    self.type_env[new_name] = old_type
    return new_name

  def rename_var(self, old_var):
    new_name = names.refresh(old_var.name)
    new_var = Var(new_name, old_var.type)
    self.rename_dict[old_var.name] = new_var
    self.type_env[new_name] = old_var.type
    return new_var

  def transform_merge(self, merge):
    new_merge = {}
    for (old_name, (l,r)) in merge.iteritems():
      new_name = self.rename(old_name)
      new_left = self.transform_expr(l)
      new_right = self.transform_expr(r)
      new_merge[new_name] = (new_left, new_right)
    return new_merge

  def transform_merge_before_loop(self, merge):
    new_merge = {}
    for (old_name, (l,r)) in merge.iteritems():
      new_name = self.rename(old_name)

      new_left = self.transform_expr(l)
      new_merge[new_name] = (new_left, r)
    return new_merge

  def transform_merge_after_loop(self, merge):
    for (new_name, (new_left, old_right)) in merge.items():
      merge[new_name] = (new_left, self.transform_expr(old_right))
    return merge

  def transform_Assign(self, expr):
    for name in collect_binding_names(expr.lhs):
      self.rename(name)
    new_lhs = self.transform_expr(expr.lhs)
    new_rhs = self.transform_expr(expr.rhs)
    return Assign(new_lhs, new_rhs)

  def transform_Var(self, expr):
    return self.rename_dict.get(expr.name, expr)

  def transform_ForLoop(self, stmt):
    new_var = self.rename_var(stmt.var)

    merge = self.transform_merge_before_loop(stmt.merge)
    new_start = self.transform_expr(stmt.start)
    new_stop = self.transform_expr(stmt.stop)
    new_step = self.transform_expr(stmt.step)
    new_body = self.transform_block(stmt.body)
    merge = self.transform_merge_after_loop(merge)
    return ForLoop(new_var, new_start, new_stop, new_step, new_body, merge)
