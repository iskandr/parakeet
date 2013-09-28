
from ..syntax import Assign, ExprStmt, If, While, ForLoop, Comment, Return, ParFor  

def can_inline_block(stmts, outer = False):
  for stmt in stmts:
    stmt_class = stmt.__class__
    if stmt_class in (Assign, ExprStmt, ParFor):
      pass
    elif stmt_class is If:
      
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