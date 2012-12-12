import closure_type
import core_types
import syntax
import syntax_visitor

class Verify(syntax_visitor.SyntaxVisitor):
  def __init__(self, fn):
    self.fn = fn

  def run(self):
    self.visit_block(self.fn.body)

  def visit_generic_expr(self, expr):
    for k in expr.members():
      v = getattr(expr, k)
      if v and isinstance(v, syntax.Expr):
        self.visit_expr(v)
  """
  def visit_If(self, stmt):

    self.visit_merge(stmt.merge)
    self.visit_expr(stmt.cond)
    self.visit_block(stmt.true)
    self.visit_block(stmt.false)
  """

  def visit_merge(self, phi_nodes, both_branches = False):
    for (k, (left_value, right_value)) in phi_nodes.iteritems():
      self.visit_expr(left_value)
      self.visit_expr(right_value)
      assert isinstance(left_value, syntax.Expr), \
          "Invalid expression in phi node: %s" % left_value
      assert isinstance(right_value, syntax.Expr), \
          "Invalid expression in phi node: %s" % right_value
      assert left_value.type == right_value.type
      assert k in self.fn.type_env
      assert self.fn.type_env[k] == left_value.type

  def visit_Var(self, expr):
    assert expr.name in self.fn.type_env, \
        "Unknown variable %s" % expr.name

    assert expr.type == self.fn.type_env[expr.name], \
        "Variable %s should have type %s but annotated with type %s" % \
        (expr.name, self.fn.type_env[expr.name], expr.type)

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
        "Inccorect type for returned value %s" % (stmt.value)

  def visit_TypedFn(self, fn):
    return verify(fn)

  def visit_Assign(self, stmt):
    self.visit_expr(stmt.rhs)
    assert stmt.lhs.type == stmt.rhs.type, \
        "Mismatch between LHS type %s and RHS %s in '%s'" % \
        (stmt.lhs.type, stmt.rhs.type, stmt)

def verify(fn):
  Verify(fn).run()
