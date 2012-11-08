import parakeet 
import syntax
import testing_helpers
 
def A(x):
  return x + 1

def B(x):
  return A(x)

def C(x):
  return B(x) 

def simple_args(exprs):
  return all(simple_expr(e) for e in exprs)

def simple_primcall(e):
  return isinstance(e, syntax.PrimCall) and simple_args(e.args)

def simple_expr(expr):
  return isinstance(expr, (syntax.Const, syntax.Var)) or \
    simple_primcall(expr)

def simple_stmt(stmt):
  """
  Is this a statement from a simple
  straightline function?
  
  (can only contain scalar computations and 
   no control flow)
  """
  if isinstance(stmt, syntax.Return):
    return simple_expr(stmt.value)
  elif isinstance(stmt, syntax.Assign):
    return simple_expr(stmt.rhs)
  else:
    return False 
  
def simple_block(stmts):
  return all(simple_stmt(s) for s in stmts)
  
def simple_fn(fn):
  return simple_block(fn.body)

def test_inline():
  typed_fn = parakeet.typed_repr(C, [1])
  assert len(typed_fn.body) in [1,2], \
    "Expected body to be 1 or 2 statements, got %s" % typed_fn
  assert simple_fn(typed_fn)
  testing_helpers.expect(C, [1], 2)
  
def lots_of_arith(x):
  y = 4 * 1
  z = y + 1
  a = z / 5 
  b = x * a
  return b

def test_simple_constant_folding():
  testing_helpers.expect(lots_of_arith, [1], 1)
  typed_fn = parakeet.typed_repr(lots_of_arith, [1])  
  assert len(typed_fn.body) == 1, \
    "Insufficiently simplified body: %s" % typed_fn


def const_across_control_flow(b):
  if b:
    x = 1
  else:
    x = 1
  return x  

def test_constants_across_control_flow():
  testing_helpers.expect(const_across_control_flow, [True], 1) 
  typed_fn = parakeet.typed_repr(const_across_control_flow, [True]) 
  assert len(typed_fn.body) == 1, "Fn body too long: " + str(typed_fn.body)
  stmt = typed_fn.body[0]
  assert isinstance(stmt, syntax.Return)
  assert isinstance(stmt.value, syntax.Const)

if __name__ == '__main__':
  testing_helpers.run_local_tests()