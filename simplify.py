import transform 
import prims 
from syntax_helpers import collect_constants, is_one, is_zero, all_constants
import type_inference
import syntax 
import syntax_helpers 
import dead_code_elim


# classes of expressions known to have no side effects 
# and to be unaffected by changes in mutable state as long 
# as all their arguments are SSA variables or constants 
# 
# examples of unsafe expressions: 
#  - Index: the underlying array is mutable, thus the expression depends on 
#    any data modifications
#  - Call: Unless the function is known to contain only safe expressions it 
#    might depend on mutable state or modify it itself 
 


from transform import Transform

class Simplify(Transform):
  def __init__(self, fn):
    # associate var names with
    #  1) constant values: these should always be replaced
    # 
    #  2) tuples: we can replace these only where accessing elts of the tuple
    #
    #  3) closures: later convert invoke to direct fn calls 
     
    self.env = {}
    transform.Transform.__init__(self, fn)
    
  def is_simple(self, expr):
    return isinstance(expr, (syntax.Const, syntax.Var))
  
  def all_safe(self, exprs):
    return all(self.is_safe(e) for e in exprs)
  
  def is_safe(self, expr):
    return self.is_simple(expr) or expr is None or \
        (isinstance(expr, syntax.PrimCall) and self.all_safe(expr.args)) or \
        (isinstance(expr, syntax.Tuple) and self.all_safe(expr.elts)) or \
        (isinstance(expr, syntax.Closure) and self.all_safe(expr.args)) or \
        (isinstance(expr, syntax.ClosureElt) 
         and self.is_safe(expr.closure)) or \
        (isinstance(expr, syntax.TupleProj) and self.is_safe(expr.tuple)) or \
        (isinstance(expr, syntax.Slice) and 
         self.all_safe((expr.start, expr.stop, expr.step)))  or \
        (isinstance(expr, syntax.Array) and self.all_safe(expr.elts)) or \
        (isinstance(expr, syntax.Cast) and self.is_safe(expr.value))
  
  
  def match_var(self, name, rhs):
    if isinstance(rhs, syntax.Var):
      old_val = self.env.get(rhs.name)
      if self.is_simple(old_val):
        self.env[name] = old_val
      else:
        self.env[name] = rhs
    
    elif self.is_safe(rhs):
      self.env[name] = rhs 
      
  def match(self, lhs, rhs):
    if isinstance(lhs, syntax.Var):
      self.match_var(lhs.name, rhs)      
    elif isinstance(lhs, syntax.Tuple) and isinstance(rhs, syntax.Tuple):
      for (lhs_elt, rhs_elt) in zip(lhs.elts, rhs.elts):
        self.match(lhs_elt, rhs_elt)
        
  def transform_Assign(self, stmt):
    new_rhs = self.transform_expr(stmt.rhs)
    self.match(stmt.lhs, new_rhs)
    return syntax.Assign(stmt.lhs, new_rhs)
  
  def transform_Var(self, expr):
    name = expr.name
    original_expr = expr 
    
    while name in self.env: 
        
      expr = self.env[name]
      if isinstance(expr, syntax.Var):
        name = expr.name 
      else:
        break  

    if isinstance(expr, syntax.Const):
      return expr 
    elif name == original_expr.name:
      return original_expr
    else:
      new_var = syntax.Var(name = name, type = original_expr.type)
      return new_var

  
  def transform_Attribute(self, expr):
    v = self.transform_expr(expr.value)
    if isinstance(v, syntax.Struct):
      idx = v.type.field_pos(expr.name)
      return v.args[idx]
    else:
      return syntax.Attribute(v, expr.name, type = expr.type)
  
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
  
  def transform_IntToPtr(self, expr):
    intval = self.transform_expr(expr.value)
    if isinstance(intval, syntax.Var) and intval.name in self.env:
      intval = self.env[expr.name]
      
    # casting a pointer to an integer and casting it back should be a no-op
    if isinstance(intval, syntax.IntToPtr) and expr.type == intval.value.type:
      return intval.value
    else:
      return syntax.IntToPtr(intval, type = expr.type)
  
  def transform_Call(self, expr):
    import closure_type
    fn = self.transform_expr(expr.fn)
    args = self.transform_args(expr.args) 
    if isinstance(fn.type, closure_type.ClosureT) and \
        isinstance(fn.type.fn, syntax.TypedFn):
      closure_elts = self.closure_elts(fn)
      combined_args = closure_elts + tuple(args)
      return syntax.Call(fn.type.fn, combined_args, type = expr.type)
    elif fn != expr.fn or any(e1 != e2 for (e1, e2) in zip(args, expr.args)):
      return syntax.Call(fn, args, type = expr.type)
    else:
      return expr  
  

  
  def transform_args(self, args):
    new_args = []
    for arg in args:
      new_arg = self.transform_expr(arg)
      if self.is_simple(new_arg):
        new_args.append(new_arg)
      else:
        new_var = self.assign_temp(new_arg)
        new_args.append(new_var)
    return new_args 
  
  def transform_Struct(self, expr):
    new_args = self.transform_args(expr.args)
    return syntax.Struct(new_args, type = expr.type)
  
  def transform_PrimCall(self, expr):
    args = self.transform_args(expr.args)
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
  
  def transform_phi_nodes(self, phi_nodes):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      new_left = self.transform_expr(left)
      new_right = self.transform_expr(right)
      if isinstance(new_left, syntax.Const) and \
         isinstance(new_right, syntax.Const)  and \
         new_left.value == new_right.value:
        self.env[k] = new_left
      # WARNING: THIS IS SUSPICIOUS! 
      # How does it interact with loops? 
      elif isinstance(new_left, syntax.Var) and \
           isinstance(new_right, syntax.Var) and \
           new_left.name == new_right.name:
        self.env[k] = new_left
      else:
        result[k] = new_left, new_right 
    return result 
  
  def post_apply(self, new_fn):
    new_fn = dead_code_elim.dead_code_elim(new_fn)
    Transform.post_apply(self, new_fn)
    return new_fn 