from ..syntax import Var, Const 
from transform import Transform 

class ConstArgSpecialization(Transform):
  """
  Transforms calls with constant args i.e. f(1,2,x) into f'(x) where values of 1,2 
  have been inlined
  
  TODO: Actually implement function specialization  
  """
  
  def pre_apply(self, old_fn):
    self.consts = {}
  
  def visit_Assign(self, stmt):
    if stmt.lhs.__class__ is Var and stmt.rhs.__class__ is Const:
      self.consts[stmt.lhs.name] = stmt.rhs
    return stmt 
  
  def _split_args(self, arg_names, arg_types, actuals):
    keep_arg_names = []
    keep_arg_types = []
    assigned_args = {}
    for i, actual in enumerate(actuals):
      formal_name = arg_names[i]
      formal_type = arg_types[i]
      assert formal_type == actual.type, \
        "Unexpected type mismatch formal %s != acutal %s" % (formal_type, actual.type) 
      if actual.__class__ is Const:
        assigned_args[formal_name] = actual 
      else:
        keep_arg_names.append(formal_name)
        keep_arg_types.append(formal_type)
    return assigned_args, keep_arg_names, keep_arg_types 
  
  def visit_Call(self, expr):
    actuals = expr.args
    
    # substitute in any constants we found  
    for i in xrange(len(actuals)):
      old_arg = actuals[i]
      if old_arg.__class__ is Var and old_arg.name in self.consts:
        actuals[i] = self.consts[old_arg.name]
    #closure_elts = self.closure_elts(expr.fn)
    #fn = self.get_fn(expr.fn)
    
    # assigned_args, keep_arg_names, keep_arg_types = self._split_args(expr.fn 
    expr.args = actuals   
    return expr 
     
      