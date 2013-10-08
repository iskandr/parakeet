
from ..syntax import Assign, ExprStmt, If, While, ForLoop, Comment, Return, ParFor  

def all_branches_return(stmt):
  if isinstance(stmt, Return):
    return True 
  elif isinstance(stmt, If):
    return len(stmt.true) > 0 and \
      all_branches_return(stmt.true[-1]) and \
      len(stmt.false) > 0 and \
      all_branches_return(stmt.false[-1])

def can_inline_block(stmts, outer = False):
  for stmt in stmts:
    stmt_class = stmt.__class__
    if stmt_class in (Assign, ExprStmt, ParFor):
      pass
    elif stmt_class is If:
      # if every branch leading from here ends up 
      # returning a value, then it's OK to replace those
      # with variable assignments  
      # ...but only once the Inliner knows how to deal with 
      # control flow...      
      #if outer and all_branches_return(stmt):
      #  return True
      if not can_inline_block(stmt.true, outer = False):
        return False
      if not can_inline_block(stmt.false, outer=False):
        return False 
          
    elif stmt_class in (While, ForLoop):
      if not can_inline_block(stmt.body):
        return False
    elif stmt_class is Comment:
      continue
    
    else:
      assert stmt_class is Return, "Unexpected statement: %s" % stmt
      if not outer:
        return False 
  return True

def can_inline(fundef):
  return can_inline_block(fundef.body, outer = True)