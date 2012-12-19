import transform 
import prims 
from syntax_helpers import collect_constants, is_one, is_zero, all_constants
import syntax 

import dead_code_elim
from syntax import Const, Var, Tuple,  TupleProj, Closure, ClosureElt, Cast
from syntax import Slice, Index, Array, ArrayView,  Attribute, Struct
from syntax import PrimCall, Call
from mutability_analysis import TypeBasedMutabilityAnalysis
from scoped_env import ScopedEnv 


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
    transform.Transform.__init__(self, fn)
    # associate var names with any immutable values
    # they are bound to 
    self.bindings = {}

    # which expressions have already been computed
    # and stored in some variable? 
    self.available_expressions = ScopedEnv()
    
    ma = TypeBasedMutabilityAnalysis()
    
    # which types have elements that might 
    # change between two accesses?  
    self.mutable_types = ma.visit_fn(fn)
    
    use_counter = dead_code_elim.VarUseCount()
    self.use_counts = use_counter.visit_fn(fn)
      
  
  
  def immutable_type(self, t):
    return t not in self.mutable_types
  
  def children(self, expr, allow_mutable = False):
    c = expr.__class__ 
    if c is Const or c is Var:
      return ()
    elif c is PrimCall or c is Closure or c is Call:
      return expr.args 
    elif c is ClosureElt:
      return (expr.closure,)
    elif c is Tuple:
      return expr.elts
    elif c is TupleProj:
      return (expr.tuple,)
    elif c is Slice:
      return (expr.start, expr.stop, expr.step)
    elif c is Cast:
      return (expr.value,)
     
    if allow_mutable or self.immutable_type(expr.type):
      if c is Array :
        return expr.elts
      elif c is ArrayView:
        return (expr.data, expr.shape, expr.strides, expr.offset)    
      elif c is Struct:
        return expr.args
      elif c is Attribute:
        return (expr.value,)
    return None 
  
  def is_simple(self, expr):
    return self.children(expr) is ()
  
  def immutable(self, expr):
    child_nodes = self.children(expr, allow_mutable = False)
    if child_nodes is None:
      return False 
    else:
      return all(self.immutable(child) for child in child_nodes)


  def temp(self, expr,   use_count = 1):
    """
    Wrapper around Codegen.assign_temp 
    which also updates bindings and use_counts
    """
    new_var = self.assign_temp(expr)
    self.bindings[new_var.name] = expr
    self.use_counts[new_var.name] = use_count 
    return new_var
   
  def transform_expr(self, expr):
    stored = self.available_expressions.get(expr)
    if stored: 
      return stored
    else:
      return Transform.transform_expr(self, expr )
    
  
  def transform_Var(self, expr):
    name = expr.name
    original_expr = expr 
    while name in self.bindings: 
      expr = self.bindings[name]
      if expr.__class__ is syntax.Var:
        name = expr.name 
      else:
        break  

    if expr.__class__ is Const:
      return expr 
    
    elif expr == original_expr.name:
      return original_expr
    else:
      return syntax.Var(name = name, type = original_expr.type)
      
  
  def transform_Attribute(self, expr):
    v = self.transform_expr(expr.value)
    if v.__class__ is Var and v.name in self.bindings:
      stored_v = self.bindings[v.name]
      c = stored_v.__class__
      if c is Var or c is Struct:
        v = stored_v 
      
    if v.__class__ is Struct:
      idx = v.type.field_pos(expr.name)
      return v.args[idx]
    elif v.__class__ is not Var:
      v = self.temp(v, "struct_temp")
    return Attribute(v, expr.name, type = expr.type)
  
  def transform_TupleProj(self, expr):

    idx = expr.index
    assert isinstance(idx, int), \
      "TupleProj index must be an integer, got: " + str(idx) 
    new_tuple = self.transform_expr(expr.tuple)

    if isinstance(new_tuple, syntax.Var) and new_tuple.name in self.bindings:
      new_tuple = self.bindings[new_tuple.name]
      
    if isinstance(new_tuple, syntax.Tuple):
      return new_tuple.elts[idx] 
    else:
      return syntax.TupleProj(tuple = new_tuple, index = idx, type = expr.type)
  
  
  def transform_Call(self, expr):
    import closure_type
    fn = self.transform_expr(expr.fn)
    args = self.transform_args(expr.args) 
    if isinstance(fn.type, closure_type.ClosureT) and \
        isinstance(fn.type.fn, syntax.TypedFn):
      closure_elts = self.closure_elts(fn)
      combined_args = closure_elts + tuple(args)
      return Call(fn.type.fn, combined_args, type = expr.type)
    elif fn != expr.fn or any(e1 != e2 for (e1, e2) in zip(args, expr.args)):
      return Call(fn, args, type = expr.type)
    else:
      return expr  
  

  
  def transform_args(self, args):
    new_args = []
    for arg in args:
      new_arg = self.transform_expr(arg)
      if self.is_simple(new_arg):
        new_args.append(new_arg)
      else:
        new_args.append(self.temp(new_arg))
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
      if new_left == new_right:
        self.set_binding(k, new_left)
        if not isinstance(new_left, (syntax.Const, syntax.Var)):
          result[k] = new_left, new_right 
      else:
        result[k] = new_left, new_right
    return result 
  
  def set_binding(self, name, value):
    assert value.__class__ is not Var or \
        value.name != name, \
        "Can't set name %s bound to itself" % name 
    self.bindings[name] = value 
  
  
  def bind_var(self, name, rhs):
    if isinstance(rhs, syntax.Var):
      old_val = self.bindings.get(rhs.name)
      if old_val and self.is_simple(old_val):
        self.set_binding(name, old_val)
      else:
        self.set_binding(name, rhs)
    else:
    
      self.set_binding(name, rhs) 
      
  def bind(self, lhs, rhs):
    lhs_class = lhs.__class__ 
    if lhs_class is Var:
      self.bind_var(lhs.name, rhs)
    elif lhs_class is Tuple and rhs.__class__ is Tuple:
      assert len(lhs.elts) == len(rhs.elts)
      for lhs_elt, rhs_elt in zip(lhs.elts, rhs.elts):
        self.bind(lhs_elt, rhs_elt)
  
  def transform_lhs_Index(self, lhs):
    old_idx = lhs.index 
    new_idx = self.transform_expr(old_idx)
    if lhs.value.__class__ is Var:
      stored = self.bindings.get(lhs.value.name)
      if stored and stored.__class__ is Var: 
        return Index(stored, new_idx, type = lhs.type)
    elif new_idx != old_idx:
      return Index(lhs.value, new_idx, type = lhs.type)
    return lhs
     
  def transform_lhs_Attribute(self, lhs):
    return lhs 
  
  def transform_Assign(self, stmt):
    lhs = stmt.lhs 
    lhs_class = lhs.__class__
    old_rhs = stmt.rhs 
    rhs = self.transform_expr(old_rhs)
    rhs_class = rhs.__class__ 
     
    if lhs_class is Index:
      lhs = self.transform_lhs_Index(lhs)
    elif lhs_class is Attribute:
      lhs = self.transform_lhs_Attribute(lhs)
    else:
      self.bind(lhs, rhs)
     
    if lhs_class is Var and \
       rhs_class not in (Var, Const) and \
        self.immutable(rhs) and \
        rhs not in self.available_expressions:
      self.available_expressions[rhs] = lhs
    elif rhs_class is Var:
      rhs_name = rhs.name
      if rhs_name in self.bindings and \
          self.use_counts.get(rhs.name, 1) == 1:
        rhs = self.bindings[rhs_name]
        self.use_counts[rhs_name] = 0    
    new_stmt = syntax.Assign(lhs, rhs) if rhs != old_rhs else stmt
    return new_stmt 
  
  def transform_If(self, stmt):
    self.available_expressions.push()
    stmt = Transform.transform_If(self, stmt)
    self.available_expressions.pop()
    return stmt 
  
  
  def transform_While(self, stmt):
    self.available_expressions.push()
    stmt = Transform.transform_While(self, stmt)
    _ = self.available_expressions.pop()
    return stmt 
    
  def transform_Return(self, stmt):
    new_value = self.transform_expr(stmt.value)
    value_class = new_value.__class__
    if value_class not in (Var, Const):
      return syntax.Return(self.temp(new_value))
    elif new_value == stmt.value:
      return stmt 
    else:
      return syntax.Return (new_value)  
    
    
  def post_apply(self, new_fn):
    new_fn = dead_code_elim.dead_code_elim(new_fn)
    Transform.post_apply(self, new_fn)
    return new_fn 
