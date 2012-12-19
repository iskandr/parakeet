import closure_type
import core_types
import syntax
import syntax_visitor

class Verify(syntax_visitor.SyntaxVisitor):
  def __init__(self, fn):
    self.fn = fn

  def run(self):
    self.visit_block(self.fn.body)



  def visit_merge(self, phi_nodes):
    for (k, (left_value, right_value)) in phi_nodes.iteritems():
      self.visit_expr(left_value)
      self.visit_expr(right_value)
      assert isinstance(left_value, syntax.Expr), \
          "Invalid expression in phi node: %s" % left_value
      assert isinstance(right_value, syntax.Expr), \
          "Invalid expression in phi node: %s" % right_value
      assert left_value.type == right_value.type
      assert k in self.fn.type_env, "%s not in type env" % k
      assert self.fn.type_env[k] == left_value.type

  def visit_Var(self, expr):
    assert expr.name in self.fn.type_env, \
        "Unknown variable %s" % expr.name

    assert expr.type == self.fn.type_env[expr.name], \
        "Variable %s in fn %s should have type %s but annotated with %s" % \
        (expr.name, self.fn, self.fn.type_env[expr.name], expr.type)

  def visit_PrimCall(self, expr):
    self.visit_expr_list(expr.args)

    for arg in expr.args:
      assert arg.type is not None, \
          "Expected type annotation for %s" % (arg, )
      assert isinstance(arg.type, core_types.ScalarT), \
          "Can't call primitive %s with argument %s of non-scalar type %s" % \
          (expr.fn, arg, arg.type)

  def visit_Return(self, stmt):
    self.visit_expr(stmt.value)
    assert stmt.value.type and stmt.value.type == self.fn.return_type, \
        "Incorrect type for returned value %s, types=(%s,%s)" % \
        (stmt.value, stmt.value.type, self.fn.return_type)

  def visit_TypedFn(self, fn):
    return verify(fn)

  def visit_Assign(self, stmt):
    self.visit_expr(stmt.rhs)
    assert stmt.lhs.type == stmt.rhs.type, \
        "Mismatch between LHS type %s and RHS %s in '%s'" % \
        (stmt.lhs.type, stmt.rhs.type, stmt)

def verify(fn):
  Verify(fn).run()
