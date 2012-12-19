import syntax 

from simplify import Simplify
from inline import Inliner

def simple_expr(expr):
  if expr.__class__ is syntax.Call:
    return False 
  elif expr.__class__ in (syntax.Const, syntax.Var):
    return True 
  else:
    return all(not isinstance(child, syntax.Expr) or simple_expr(child)
               for child in expr.itervalues())
    
def simple_stmt(stmt):
  c = stmt.__class__ 
  if c is syntax.Return:
    return simple_expr(stmt.value)
  elif c is syntax.Assign:
    return simple_expr(stmt.rhs)
  elif c is syntax.If:
    return simple_expr(stmt.cond) and simple_block(stmt.true) and simple_block(stmt.false)
  else:
    assert c is syntax.While, "Unexpected expr class %s" % c 
    return simple_expr(stmt.cond) and simple_block(stmt.body)

def simple_block(stmts):
  return all(simple_stmt(stmt) for stmt in stmts)      

def simple_fn(fn):
  return simple_block(fn.body)
  

# map names of unoptimized typed functions to 
# names of optimized 
_optimized_cache = {}
def optimize(fn, copy = False):
  if isinstance(fn, syntax.Fn):
    raise RuntimeError("Can't optimize untyped functions")
  elif isinstance(fn, str):
    assert fn in syntax.TypedFn.registry, \
      "Unknown typed function: " + str(fn)
      
    fn = syntax.TypedFn.registry[fn]
  else:
    assert isinstance(fn, syntax.TypedFn)
      
  if fn.name in _optimized_cache:
    return _optimized_cache[fn.name]
  else:
    opt = Simplify(fn).apply(copy = True)
    if not simple_fn(fn):
      inliner = Inliner(opt)
      opt = inliner.apply(copy=False)
    opt = Simplify(opt).apply(copy=False)
    _optimized_cache[fn.name] = opt
    return opt 