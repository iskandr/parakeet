import syntax
from common import dispatch

def is_live_lhs(lhs, live_vars):
  if isinstance(lhs, syntax.Var):
    return lhs.name in live_vars
  elif isinstance(lhs, syntax.Tuple):
    return any(is_live_lhs(elt, live_vars) for elt in lhs.elts)
  elif isinstance(lhs, str):
    return lhs in live_vars
  elif isinstance(lhs, tuple):
    return any(is_live_lhs(elt, live_vars) for elt in lhs)
  else:
    return True

def elim_merge(merge, live_vars):
  new_merge = {}
  for var in merge:
    if var in live_vars:
      new_merge[var] = merge[var]
  return new_merge

import syntax_helpers

def elim_stmt(stmt, live_vars):
  def elim_Assign():
    if is_live_lhs(stmt.lhs, live_vars):
      return stmt
    else:

      return None

  def elim_While():
    new_body = elim_block(stmt.body, live_vars)
    new_merge = elim_merge(stmt.merge, live_vars)
    if len(new_merge) == 0 and \
       (len(new_body) == 0 or syntax_helpers.is_false(stmt.cond)):
      return None
    return syntax.While(stmt.cond, new_body, new_merge)

  def elim_If():
    new_true = elim_block(stmt.true, live_vars)
    new_false = elim_block(stmt.false, live_vars)
    new_merge = elim_merge(stmt.merge, live_vars)
    if len(new_true) == 0 and len(new_false) == 0 and \
       (len(new_merge) == 0 or syntax_helpers.is_false(stmt.cond)):
      return None
    else:
      return syntax.If(stmt.cond, new_true, new_false, new_merge)

  def elim_Return():
    return stmt

  return dispatch(stmt, 'elim')

def elim_block(stmts, live_vars):
  new_block = []
  for stmt in stmts:
    new_stmt = elim_stmt(stmt, live_vars)
    if new_stmt:
      new_block.append(new_stmt)
  return new_block
