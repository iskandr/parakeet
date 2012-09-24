from ssa import Assign, Op, Var, Const
from ssa import SyntaxTraversal
from types import type_of_value 

class InferTypes(SyntaxTraversal):
  def stmt_Set(self, stmt, tenv): 
    rhs_type = self.visit_expr(stmt.rhs, tenv)
    tenv[stmt.lhs] = rhs_type

  def expr_Const(self, expr, tenv):
    return type_of_value(expr.value)

  def expr_Var(self, expr, tenv):
    assert expr.id in tenv, \
      "Unknown variable " + expr.id  
    return tenv[expr.id]
  
  def expr_Call(self, expr, tenv):
    fn_expr = expr.fn
    # list of types 
    arg_types = map(self.visit_expr, expr.args)
    # dict of string name -> type mappings
    kwd_types = dict([ (k, self.visit_expr(v)) for \
      (k,v) in expr.kwds.iteritems()])
       
    if isinstance(fn, Var):
      untyped_fn = find_fn(fn_expr.id)
      global_types =  
      combine_args(untyped_fn.arg_names,
        arg_types, 
        untyped_fn.kwds, 
        kwd_types, 
        untyped_fn.global
         
      typed_fn = specialize(fn,   
    
    # if it's an operator  
    # then create a typed operator node
    # if it's a function then specialize it
    # otherwise, throw an error since 
    # only globally known functions are kosher...?
    # what if it was defined in the local scope?
    # def f(x):
    #   def g(y):
    #     return x + y
    #   return g(3)
    # called with f(2.0)
    # ....
    # - give f and g unique IDs like any other
    #    variable
    # - What about the bound closure variables 
    #    of a function? 
    # We can't just put "g" in a global lookup
    # since it depends on references to variables
    # of f's scope. 
    # What if we include in a function's description
    # the names of other variables it relies on
    # and their types? 
    # ...could a function then be returned? 
    # def f(x):
    #   def g(y):
    #     return x + y
    #   return g(3)
    # What if we just do a simple closure conversion?
    # CODE f(closure_f, x):
    #   CODE g(closure_g, y):
    #     x = closure_x[0]
    #     return x + y
    #   g = <package_fn g, x>
    #   CALL(g, 3)
    #   
    # ...but this goes too far, since we're 
    # giving functions first-class representation
    # What if we instead just say:
    #   g.code = "...", 
    #   g.globals = "id1, id2, etc..."
    #   and when typed
    #   g.globals = "id1: t1, id2: t2"
    # later we keep scoped  
   
   

