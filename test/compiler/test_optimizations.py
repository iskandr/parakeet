import numpy as np

import parakeet
from parakeet import config, each, syntax
from parakeet.transforms.pipeline import lowering
from parakeet.analysis.syntax_visitor import SyntaxVisitor
from parakeet.testing_helpers import expect, run_local_tests


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
  Is this a statement from a simple straightline function?

  (can only contain scalar computations and no control flow)
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
  expect(C, [1], 2)

def lots_of_arith(x):
  y = 4 * 1
  z = y + 1
  a = z / 5
  b = x * a
  return b

def test_simple_constant_folding():
  expect(lots_of_arith, [1], 1)
  typed_fn = parakeet.typed_repr(lots_of_arith, [1], optimize = True)
  assert len(typed_fn.body) == 1, \
      "Insufficiently simplified body: %s" % typed_fn

def const_across_control_flow(b):
  if b:
    x = 1
  else:
    x = 1
  return x

def test_constants_across_control_flow():
  expect(const_across_control_flow, [True], 1)
  typed_fn = parakeet.typed_repr(const_across_control_flow, [True])
  assert len(typed_fn.body) == 1, "Fn body too long: " + str(typed_fn.body)
  stmt = typed_fn.body[0]
  assert isinstance(stmt, syntax.Return)
  assert isinstance(stmt.value, syntax.Const)

def always_true_branch():
  x = 1 + 1
  if x == 2:
    res = 0 + 0
    return res
  else:
    res = 1 * 1 + 0
    return res

def test_always_true():
  expect(always_true_branch, [], 0)
  typed_fn = parakeet.typed_repr(always_true_branch, [])
  assert len(typed_fn.body) == 1, "Fn body too long: " + str(typed_fn.body)
  stmt = typed_fn.body[0]
  assert isinstance(stmt, syntax.Return)
  assert isinstance(stmt.value, syntax.Const)

def always_false_branch():
  x = 1 + 2
  if x == 2:
    res = 1 * 0 + 0
    return res
  else:
    res = 0 + 1 * 1
    return res

def test_always_false():
  expect(always_false_branch, [], 1)
  typed_fn = parakeet.typed_repr(always_false_branch, [])
  assert len(typed_fn.body) == 1, "Fn body too long: " + str(typed_fn.body)
  stmt = typed_fn.body[0]
  assert isinstance(stmt, syntax.Return)
  assert isinstance(stmt.value, syntax.Const)


def g(x):
  def h(xi):
    return xi + 1.0
  return each(h,x)

def nested_add1(X):
  return each(g, X)

class CountLoops(SyntaxVisitor):
  def __init__(self):
    SyntaxVisitor.__init__(self)
    self.count = 0

  def visit_While(self, stmt):
    self.count += 1
    SyntaxVisitor.visit_While(self, stmt)

def count_loops(fn):
  Counter = CountLoops()
  Counter.visit_fn(fn)
  return Counter.count

def test_copy_elimination():
  x = np.array([[1,2,3],[4,5,6]])
  expect(nested_add1, [x], x + 1.0)
  typed_fn = parakeet.typed_repr(nested_add1, [x])
  lowered = lowering.apply(typed_fn)
  n_loops = count_loops(lowered)
  n_expected = 3 if config.opt_loop_unrolling else 2
  assert n_loops <= n_expected, \
      "Too many loops generated! Expected at most 2, got %d" % n_loops

  
if __name__ == '__main__':
  run_local_tests()
