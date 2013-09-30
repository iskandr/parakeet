from .. ndtypes import ArrayT, NoneT, NoneType, ScalarT, ClosureT, TupleT, FnT, Type
from .. ndtypes import lower_rank 

from .. syntax import Expr, Tuple, Var, Index, Closure, TypedFn 
from .. syntax.helpers import get_types
from collect_vars import collect_binding_names
from syntax_visitor import SyntaxVisitor

class Verify(SyntaxVisitor):
  def __init__(self, fn):
    SyntaxVisitor.__init__(self)
    self.fn = fn
    self.bound = set(fn.arg_names)
    self.seen_return = False 

  def bind_var(self, name):
    assert name not in self.bound, \
        "Error: variable %s already bound" % name
    self.bound.add(name)
    assert name in self.fn.type_env, \
        "Assigned variable %s has no entry in type dictionary" % name

  def phi_value_ok(self, lhs_name, v):
    assert isinstance(v, Expr), \
       "Invalid expression in phi node: %s" % (v,)
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
      assert isinstance(arg.type, ScalarT), \
          "Can't call primitive %s with argument %s of non-scalar type %s" % \
          (expr.prim, arg, arg.type)

  def visit_expr(self, expr):
    assert expr is not None
    assert expr.type is not None, \
      "Missing type annotation on %s" % expr 
    SyntaxVisitor.visit_expr(self, expr)

  def visit_ExprStmt(self, stmt):
    self.visit_expr(stmt.value)
    assert stmt.value.type and stmt.value.type.__class__ is NoneT, \
      "Expected effectful expression %s to have type %s but instead got %s" % \
      (stmt.value, NoneType, stmt.value.type)
  
  def check_fn_args(self, fn, args = None, arg_types = None):
    if arg_types is None: 
      assert args is not None 
      arg_types = get_types(args)
    n_given = len(arg_types)
    n_expected = len(fn.input_types)
    assert n_given == n_expected, \
        "Arity mismatch when calling %s, expected %d but got %d" % \
        (fn.name, n_expected, n_given)
    for (i, arg_name) in enumerate(fn.arg_names):
      given_t = arg_types[i]
      expected_t = fn.type_env[arg_name]
      signature_t = fn.input_types[i] 
      assert expected_t == signature_t, \
          "Function %s has inconsistent types %s and %s for arg %s" % \
          (fn.name, expected_t, signature_t, arg_name)
      assert given_t == expected_t, \
          "Given argument %s%s doesn't matched expected '%s' : %s in %s" % \
          ("'%s' : " %  args[i] if args is not None else "", 
           given_t, 
           arg_name, 
           expected_t, 
           fn.name)  
  
  
  def get_fn_and_closure(self, fn):
    if fn.__class__ is Closure: 
      return fn.fn, tuple(get_types(fn.args)) 
    elif fn.__class__ is TypedFn:
      return fn, ()
    elif fn.type.__class__ is FnT:
      return  fn, ()
    else:
      assert isinstance(fn.type, ClosureT), "Unexpected function %s : %s" % (fn, fn.type)
  
  
  def verify_call(self, fn, args):
    fn, closure_arg_types = self.get_fn_and_closure(fn)
    arg_types = []
    for arg in args:
      if isinstance(arg, Type):
        arg_types.append(arg)
      else:
        assert isinstance(arg, Expr)
        arg_types.append(arg.type)
    arg_types = tuple(closure_arg_types) + tuple(arg_types)
    try:
      self.check_fn_args(fn, arg_types = arg_types)
    except:
      print "[verify] Errors in function call %s(%s)" % (fn.name, args) 
      raise 
  
      
  def visit_Call(self, expr):
    self.verify_call(expr.fn, expr.args)
    
  def visit_Map(self, expr):
    if expr.fn.__class__ is Closure: 
      closure_elts = tuple(expr.fn.args) 
      fn = expr.fn.fn 
    else:
      fn = expr.fn
      closure_elts = ()
    elt_types = [lower_rank(arg.type, 1) for arg in expr.args]
    arg_types = tuple(get_types(closure_elts)) + tuple(elt_types)
    args = tuple(closure_elts) + tuple(expr.args)                      
    self.check_fn_args(fn, args, arg_types)
  
  def visit_Return(self, stmt):
    self.visit_expr(stmt.value)
    assert stmt.value.type is not None and stmt.value.type == self.fn.return_type, \
        "Incorrect return type in %s: ret value %s, expected %s but got %s" % \
        (self.fn.name, stmt.value, self.fn.return_type, stmt.value.type)
    self.seen_return = True 
    
  def visit_lhs(self, lhs):
    c = lhs.__class__
    assert c in (Var, Tuple, Index), \
       "Invalid LHS of assignment"
    if c is Tuple:
      for elt in lhs.elts:
        self.visit_lhs(elt)

  def visit_Assign(self, stmt):
    assert stmt.lhs.type is not None, \
        "Missing LHS type for assignment %s" % stmt
    assert stmt.rhs.type is not None, \
        "Missing RHS type for assignment %s" % stmt
    self.visit_lhs(stmt.lhs)
    lhs_names = collect_binding_names(stmt.lhs)
    for lhs_name in lhs_names:
      self.bind_var(lhs_name)
    self.visit_expr(stmt.rhs)
    rhs_t = stmt.rhs.type 
    if stmt.lhs.__class__ is Index:
      array_t = stmt.lhs.value.type 
      assert stmt.lhs.type == rhs_t or rhs_t == array_t.elt_type, \
        "Mismatch between LHS type %s and RHS %s in '%s'" % \
        (stmt.lhs.type, stmt.rhs.type, stmt)
    else:
      assert stmt.lhs.type == stmt.rhs.type, \
        "Mismatch between LHS type %s and RHS %s in '%s'" % \
        (stmt.lhs.type, stmt.rhs.type, stmt)

  def visit_ForLoop(self, stmt):
    assert stmt.var.__class__ is Var
    self.bind_var(stmt.var.name)
    self.visit_expr(stmt.var)

    assert stmt.start.type == stmt.var.type
    self.visit_expr(stmt.start)
    self.visit_merge_loop_start(stmt.merge)

    assert stmt.stop.type == stmt.var.type
    self.visit_expr(stmt.stop)
    self.visit_block(stmt.body)
    
    assert stmt.step.type == stmt.var.type
    self.visit_expr(stmt.step)
    self.visit_merge_loop_repeat(stmt.merge)
    
  def visit_ParFor(self, stmt):
    fn = stmt.fn
    bounds_t = stmt.bounds.type
    if isinstance(bounds_t, TupleT) and len(bounds_t.elt_types) == 1:
      args = (bounds_t.elt_types[0],)
    else:
      args = (stmt.bounds,)
    self.verify_call(fn, args)
    
  def visit_stmt(self, stmt):
    assert stmt is not None
    SyntaxVisitor.visit_stmt(self, stmt)

  def visit_TypedFn(self, fn):
    return verify(fn)

def verify(fn):
  n_input_types = len(fn.input_types)
  n_arg_names = len(fn.arg_names)
  assert n_input_types == n_arg_names, \
     "Function %s has %d input types and %d input names" % \
     (fn.name, n_input_types, n_arg_names)
  for arg_name, input_t in zip(fn.arg_names, fn.input_types):
    assert arg_name in fn.type_env, \
        "Input %s has no entry in %s's type environment" % \
        (arg_name, fn.name)
    t = fn.type_env[arg_name]
    assert input_t == t, \
        "Mismatch between %s's input type %s and the arg %s's type %s" % \
        (fn.name, input_t, arg_name, t)
  try:    
    verifier = Verify(fn) 
    verifier.visit_block(fn.body)
    assert verifier.seen_return, "Never encountered Return statement"
  except:
    print "[verify] Errors in body of function", repr(fn)
    raise 