import core_types
from syntax import Expr
from syntax_visitor import SyntaxVisitor
from collect_vars import collect_binding_names

class Verify(SyntaxVisitor):
  def __init__(self, fn):
    self.fn = fn
    self.bound = set(fn.arg_names)

  def run(self):
    self.visit_block(self.fn.body)

  def bind_var(self, name):
    assert name not in self.bound, \
        "Error: variable %s already bound" % name
    self.bound.add(name)
    assert name in self.fn.type_env, \
        "Assigned variable %s has no entry in type dictionary" % name

  def phi_value_ok(self, lhs_name, v):
    assert isinstance(v, Expr), \
       "Invalid expression in phi node: %s" % v
    self.visit_expr(v)
    assert v.type is not None, \
       "Error in phi node: value %s has no type annotation" % v
    t = self.fn.type_env[lhs_name]
    assert v.type == t, \
       "Invalid type annotation on %v, expected %s but got %s" % (v,t,v.type)
 
  def visit_merge_loop_start(self, phi_nodes):
    for (k, (v,_)) in phi_nodes.iteritems():
      self.bind_var(k)
      self.phi_value_ok(k,v)
  
  def visit_merge_loop_repeat(self, phi_nodes):
    for (k, (left_value, right_value)) in phi_nodes.iteritems():
      self.phi_value_ok(k, left_value)
      self.phi_value_ok(k, right_value)
      
  def visit_merge_if(self, phi_nodes):
    for (k, (left_value, right_value)) in phi_nodes.iteritems():
      self.bind_var(k)
      self.phi_value_ok(k, left_value)
      self.phi_value_ok(k, right_value)
    

  def visit_Var(self, expr):
    assert expr.name in self.bound, \
        "Variable %s used before assignment" % expr.name
    assert expr.name in self.fn.type_env, \
        "Variable %s has no entry in the type dictionary" % expr.name
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

  def visit_Assign(self, stmt):
    lhs_names = collect_binding_names(stmt.lhs)
    for lhs_name in lhs_names:
      self.bind_var(lhs_name)
    self.visit_expr(stmt.rhs)
    assert stmt.lhs.type == stmt.rhs.type, \
        "Mismatch between LHS type %s and RHS %s in '%s'" % \
        (stmt.lhs.type, stmt.rhs.type, stmt)

  def visit_TypedFn(self, fn):
    return verify(fn)

def verify(fn):
  Verify(fn).run()
