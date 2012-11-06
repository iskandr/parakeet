import transform
import prims 
from syntax_helpers import collect_constants, is_one, is_zero, all_constants
import type_inference
import syntax 
import syntax_helpers 



class Simplify(transform.Transform):
  def __init__(self, fn):
    # associate var names with
    #  1) constant values: these should always be replaced
    # 
    #  2) tuples: we can replace these only where accessing elts of the tuple
    #
    #  3) closures: later convert invoke to direct fn calls 
     
    self.env = {}
    self.live_vars = set([])
    transform.Transform.__init__(self, fn)
  
  def collect_live_vars(self, expr):
    if isinstance(expr, syntax.Var):
      print "LIVE VAR", expr.name 
      self.live_vars.add(expr.name)
    elif isinstance(expr, syntax.Tuple):
      for e in expr.elts:
        self.collect_live_vars(e)
    elif isinstance(expr, syntax.Attribute):
      self.collect_live_vars(expr.value)
    elif isinstance(expr, syntax.TupleProj):
      self.collect_live_vars(expr.tuple)
    elif isinstance(expr, syntax.Index):
      self.collect_live_vars(expr.value)
      self.collect_live_vars(expr.index)
    else:
      assert False, \
        "Unexpected left-hand-side expression: " + str(expr) 
  
  def match_var(self, name, rhs):
    if isinstance(rhs, syntax.Var):
      old_val = self.env.get(rhs.name)
      if old_val:
        self.env[name] = old_val 
      else:
        self.env[name] = rhs
    elif isinstance(rhs, (syntax.Tuple, syntax.Const, syntax.Closure)):
      self.env[name] = rhs 
      
  def match(self, lhs, rhs):
    if isinstance(lhs, syntax.Var):
      self.match_var(lhs.name, rhs)      
    elif isinstance(lhs, syntax.Tuple) and isinstance(rhs, syntax.Tuple):
      for (lhs_elt, rhs_elt) in zip(lhs.elts, rhs.elts):
        self.match(lhs_elt, rhs_elt)
    else:
      self.collect_live_vars(lhs)
  def transform_Var(self, expr):
    name = expr.name
    if name in self.env:
      stored_val = self.env[name]
      if isinstance(stored_val, syntax.Const):
        return stored_val 
      elif isinstance(stored_val, syntax.Var):
        # we're replacing one variable for another, so 
        # mark the new one as used 
        self.live_vars.add(stored_val.name)
        return stored_val 
    # if we're still using this variable, so mark it as used
    self.live_vars.add(name)
    return expr
    
  def transform_Assign(self, stmt):
    new_rhs = self.transform_expr(stmt.rhs)
    self.match(stmt.lhs, new_rhs)
    return syntax.Assign(stmt.lhs, new_rhs)
  
  def transform_Invoke(self, expr):
    new_closure = self.transform_expr(expr.closure)
    new_args = self.transform_expr_list(expr.args)
    
    if isinstance(new_closure, syntax.Var) and \
        new_closure.name in self.env:
      new_closure = self.env[new_closure.name]
       
    if isinstance(new_closure, syntax.Closure):
      combined_args = new_closure.args + new_args
      arg_types = syntax_helpers.get_types(combined_args)
      typed_fundef = type_inference.specialize(new_closure.fn, arg_types)
      call_name = typed_fundef.name
      return syntax.Call(call_name, combined_args, type = expr.type)
    else:
      return syntax.Invoke(new_closure, new_args, type = expr.type)
    
  def transform_TupleProj(self, expr):

    idx = expr.index
    assert isinstance(idx, int), \
      "TupleProj index must be an integer, got: " + str(idx) 
    new_tuple = self.transform_expr(expr.tuple)

    if isinstance(new_tuple, syntax.Var) and new_tuple.name in self.env:
      new_tuple = self.env[new_tuple.name]
      
    if isinstance(new_tuple, syntax.Tuple):
      return new_tuple.elts[idx] 
    else:
      return syntax.TupleProj(tuple = new_tuple, index = idx, type = expr.type)
  
  def transform_PrimCall(self, expr):
    args = self.transform_expr_list(expr.args)
    prim = expr.prim  
    if all_constants(args):
      return syntax.Const(value = prim.fn(*collect_constants(args)), type = expr.type)
    elif prim == prims.add:
      if is_zero(args[0]):
        return args[1]
      elif is_zero(args[1]):
        return args[0]   
    elif prim == prims.multiply:
      if is_one(args[0]):
        return args[1]
      elif is_one(args[1]):
        return args[0]
      elif is_zero(args[0])  or is_zero(args[1]):
        return syntax.Const(value = 0, type = expr.type)
    elif prim == prims.divide and is_one(args[1]):
      return args[0]
    return syntax.PrimCall(prim = prim, args = args, type = expr.type)
  
  def post_apply(self, new_fn):
    import dead_code_elim
    print "before DCE", new_fn
    print "live vars", self.live_vars 
    new_fn.body = dead_code_elim.elim_block(new_fn.body, self.live_vars)
    print "after DCE", new_fn 
    return new_fn 
  
def simplify(fn):
  return transform.cached_apply(Simplify, fn) 