from .. syntax import Assign, Index, Var

from loop_transform import LoopTransform

class RedundantLoadElimination(LoopTransform):
  def transform_block(self, stmts):
    stmts = LoopTransform.transform_block(self, stmts) 
    if self.is_simple_block(stmts, allow_branches = False):
      reads, writes = self.collect_memory_accesses(stmts)
      safe_arrays = set([])
      for name in reads:
        # if any alias of this array gets written to, consider it unsafe 
        aliases = self.may_alias.get(name, set([]))
        aliases.add(name)
        unsafe = any(alias in writes for alias in aliases) 
        if not unsafe:
          safe_arrays.add(name)
      available_expressions = {}
      new_stmts = []
      for stmt in stmts:
        if stmt.__class__ is Assign:
          if stmt.rhs.__class__ is Index and \
             stmt.rhs.value.__class__ is Var and \
             stmt.rhs.value.name in safe_arrays:
            key = (stmt.rhs.value.name, stmt.rhs.index)
            if key in available_expressions:
              stmt.rhs = available_expressions[key]
            elif stmt.lhs.__class__ is Var:
              available_expressions[key] = stmt.lhs
            else:
              temp = self.fresh_var(stmt.rhs.type,  "load")
              new_stmts.append(Assign(temp, stmt.rhs))
              stmt.rhs = temp
              available_expressions[key] = temp
          new_stmts.append(stmt)
      return new_stmts
    else:
      return stmts
        
        