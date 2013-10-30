
from ..analysis.collect_vars import collect_binding_names, collect_var_names
from ..analysis.escape_analysis import may_alias

from ..syntax import Return, While, ForLoop, If, Assign, Var, Index, ExprStmt
from transform import Transform

class LoopTransform(Transform):
  def is_simple_block(self, stmts, allow_branches = True):
    for stmt in stmts:
      if stmt.__class__ is If:
        if not allow_branches or \
           not self.is_simple_block(stmt.true) or \
           not self.is_simple_block(stmt.false):
          return False
      elif stmt.__class__ not in (Assign, ExprStmt):
        return False 
    return True

  def pre_apply(self, fn):
    self.may_alias = may_alias(fn)

  def collect_loop_vars(self, loop_vars, loop_body):
    """Gather the variables whose values change between loop iterations"""
    for stmt in loop_body:
      assert stmt.__class__ in (Assign, If, ExprStmt), \
        "Unexpected statement in simple block: %s" % stmt 
      if stmt.__class__ is Assign:
        lhs_names = collect_binding_names(stmt.lhs)
        rhs_names = collect_var_names(stmt.rhs)
        if any(name in loop_vars for name in rhs_names):
          loop_vars.update(lhs_names)

  def collect_memory_accesses(self, loop_body):
      # Assume code is normalized so a read will
      # directly on rhs of an assignment and
      # a write will be directly on the LHS

      # map from variables to index sets
      reads = {}
      writes = {}
      for stmt in loop_body:
        if stmt.__class__ is Assign:
          if stmt.lhs.__class__ is Index:
            assert stmt.lhs.value.__class__ is Var, "Unexpected LHS %s" % stmt.lhs 
            writes.setdefault(stmt.lhs.value.name, set([])).add(stmt.lhs.index)
          if stmt.rhs.__class__ is Index:
            assert stmt.rhs.value.__class__ is Var, "Unexpected LHS %s" % stmt.lhs 
            reads.setdefault(stmt.rhs.value.name, set([])).add(stmt.rhs.index)
      return reads, writes

  def is_loop_var(self, loop_vars, expr):
    #if expr.__class__ is Tuple:
    #  return all(self.is_loop_var(loop_vars, elt) for elt in expr.elts)
    #else:
    return expr.__class__ is Var and expr.name in loop_vars

  def any_loop_vars(self, loop_vars, expr_set):
    return any(self.is_loop_var(loop_vars, expr) for expr in expr_set)
