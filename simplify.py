import prims
import syntax
from syntax import Assign, RunExpr 
from syntax import Const, Var, Tuple,  TupleProj, Closure, ClosureElt, Cast
from syntax import Slice, Index, Array, ArrayView,  Attribute, Struct
from syntax import PrimCall, Call
from syntax_helpers import collect_constants, is_one, is_zero, all_constants
from adverbs import Map, Reduce, Scan, AllPairs, Adverb 

from core_types import NoneT, ScalarT 

import subst
import transform
from transform import Transform
from scoped_dict import ScopedDictionary

from mutability_analysis import TypeBasedMutabilityAnalysis
from collect_vars import collect_var_names
from use_analysis import use_count

# classes of expressions known to have no side effects
# and to be unaffected by changes in mutable state as long
# as all their arguments are SSA variables or constants
#
# examples of unsafe expressions:
#  - Index: the underlying array is mutable, thus the expression depends on
#    any data modifications
#  - Call: Unless the function is known to contain only safe expressions it
#    might depend on mutable state or modify it itself

class Simplify(Transform):
  def __init__(self):
    transform.Transform.__init__(self)
    # associate var names with any immutable values
    # they are bound to
    self.bindings = {}

    # which expressions have already been computed
    # and stored in some variable?
    self.available_expressions = ScopedDictionary()

  def pre_apply(self, fn):
    ma = TypeBasedMutabilityAnalysis()

    # which types have elements that might
    # change between two accesses?
    self.mutable_types = ma.visit_fn(fn)

    self.use_counts = use_count(fn)

  def immutable_type(self, t):
    return t not in self.mutable_types

  def children(self, expr, allow_mutable = False):
    c = expr.__class__
    
    if c is Const or c is Var:
      return ()
    elif c is PrimCall or c is Closure:
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
    elif c in (Map, AllPairs):
      if expr.out is None: 
        return expr.args 
      elif allow_mutable: 
        return tuple(expr.args) + (expr.out,)
      else:
        return None  
    elif c in (Scan, Reduce):  
      args = tuple(expr.args)
      init = (expr.init,) if expr.init else ()
      if expr.out is None: 
        return init + args 
      elif allow_mutable:
        return init + args + (expr.out,)
      else:
        return None
    elif c is Call:
      # assume all Calls might modify their arguments 
      if allow_mutable or all(self.immutable(arg) for arg in expr.args):
        return expr.args 
      else:
        return None 
      
    if allow_mutable or self.immutable_type(expr.type):
      if c is Array:
        return expr.elts
      elif c is ArrayView:
        return (expr.data, expr.shape, expr.strides, expr.offset,
                expr.total_elts)
      elif c is Struct:
        return expr.args
      elif c is Attribute:
        return (expr.value,)
    return None


        
  def immutable(self, expr):
    if expr.__class__ is Const or expr.__class__ is Tuple:
      return True 
    elif expr.type in self.mutable_types:
      return False 
    child_nodes = self.children(expr, allow_mutable = False)
    if child_nodes is None:
      result =  False
    else:
      result = all(self.immutable(child) for child in child_nodes)
    return result  

  def temp(self, expr, use_count = 1):
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
      return Transform.transform_expr(self, expr)

  def transform_Var(self, expr):
    name = expr.name
    original_expr = expr
    while name in self.bindings:
      expr = self.bindings[name]
      if expr.__class__ is Var:
        name = expr.name
      else:
        break

    if expr.__class__ is Const:
      return expr
    elif expr == original_expr:
      return original_expr
    else: 
      return syntax.Var(name = name, type = original_expr.type)

  def transform_Cast(self, expr):
    v = self.transform_expr(expr.value)
    if v.type == expr.type:
      return v
    elif v.__class__ is Const and isinstance(expr.type, ScalarT):
      return Const(expr.type.dtype.type(v.value), type = expr.type)
    else:
      expr.value = v 
      return expr 
      
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
      v = self.temp(v, "struct")
    return Attribute(v, expr.name, type = expr.type)

  def transform_TupleProj(self, expr):
    idx = expr.index
    assert isinstance(idx, int), \
        "TupleProj index must be an integer, got: " + str(idx)
    new_tuple = self.transform_expr(expr.tuple)

    if new_tuple.__class__ is Var and new_tuple.name in self.bindings:
      new_tuple = self.bindings[new_tuple.name]

    if new_tuple.__class__ is Tuple:
      return new_tuple.elts[idx]
    elif new_tuple.__class__ is Struct:
      return new_tuple.args[idx]
    if not self.is_simple(new_tuple):
      new_tuple = self.assign_temp(new_tuple, "tuple")
    return syntax.TupleProj(tuple = new_tuple, index = idx, type = expr.type)
    
  def transform_Call(self, expr):
    import closure_type
    fn = self.transform_expr(expr.fn)
    args = self.transform_args(expr.args)
    if fn.type.__class__ is closure_type.ClosureT and \
       fn.type.fn.__class__ is syntax.TypedFn:
      closure_elts = self.closure_elts(fn)
      combined_args = closure_elts + args
      return Call(fn.type.fn, combined_args, type = expr.type)
    else:
      expr.fn = fn 
      expr.args = args 
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

  def transform_Array(self, expr):
    expr.elts = self.transform_args(expr.elts)
    return expr 
  
  def transform_Index(self, expr):
    expr.value = self.transform_expr(expr.value)
    expr.index = self.transform_expr(expr.index)
    if expr.value.__class__ is Array and expr.index.__class__ is Const:
      assert isinstance(expr.index.value, (int, long)) and \
             len(expr.value.elts) > expr.index.value
      
      return expr.value.elts[expr.index.value]
    if expr.value.__class__ is not Var:
      expr.value = self.temp(expr.value)
    return expr

  def transform_Struct(self, expr):
    new_args = self.transform_args(expr.args)
    return syntax.Struct(new_args, type = expr.type)

  def transform_PrimCall(self, expr):
    args = self.transform_args(expr.args)
    prim = expr.prim
    if all_constants(args):
      return syntax.Const(value = prim.fn(*collect_constants(args)),
                          type = expr.type)
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

  def temp_in_block(self, expr, block, name = None):
    """
    If we need a temporary variable not in
    the current top scope but in a particular block,
    then use this function.
    (this function also modifies the bindings
    dictionary)
    """
    var = self.fresh_var(expr.type, name)
    block.append(Assign(var, expr))
    self.bindings[var.name] = expr
    return var
  
  def transform_merge(self, phi_nodes, left_block, right_block):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      new_left = self.transform_expr(left)
      new_right = self.transform_expr(right)

      if not isinstance(new_left, (Const, Var)):
        new_left = self.temp_in_block(new_left, left_block)
      if not isinstance(new_right, (Const, Var)):
        new_right = self.temp_in_block(new_right, right_block)

      if new_left == new_right:
        # if both control flows yield the same value then
        # we don't actually need the phi-bound variable, we can just
        # replace the left value everywhere
        self.set_binding(k, new_left)
      else:
        result[k] = new_left, new_right
    return result

  def set_binding(self, name, value):
    assert value.__class__ is not Var or \
        value.name != name, \
        "Can't set name %s bound to itself" % name
    self.bindings[name] = value

  def bind_var(self, name, rhs):
    if rhs.__class__ is Var:
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
  
  def transform_RunExpr(self, stmt):
    """
    Don't run an expression unless it possibly has a side effect
    """
    v = self.transform_expr(stmt.value)
    if self.immutable(v):
      return None 
    else:
      stmt.value = v 
      return stmt 
  def transform_Assign(self, stmt):
    lhs = stmt.lhs
    rhs = self.transform_expr(stmt.rhs)

    lhs_class = lhs.__class__
    rhs_class = rhs.__class__
    if lhs_class is Index:
      lhs = self.transform_lhs_Index(lhs)
      if isinstance(rhs, Adverb) and rhs.out is None:
        rhs.out = lhs 
        return RunExpr(rhs)
    elif lhs_class is Attribute:
      lhs = self.transform_lhs_Attribute(lhs)
    elif lhs_class is Var:
      if rhs.type.__class__ is NoneT and self.use_counts.get(lhs.name,0) == 0:
        return self.transform_stmt(RunExpr(rhs))    
      else: 
        self.bind_var(lhs.name, rhs)
        if rhs_class not in (Var, Const) and \
            self.immutable(rhs) and \
            rhs not in self.available_expressions:
          self.available_expressions[rhs] = lhs
    else:
      self.bind(lhs, rhs)

    if rhs_class is Var and \
       rhs.name in self.bindings and \
       self.use_counts.get(rhs.name, 1) == 1:
      self.use_counts[rhs.name] = 0
      rhs = self.bindings[rhs.name]
    stmt.lhs = lhs
    stmt.rhs = rhs
    return stmt

  def transform_block(self, stmts):
    self.available_expressions.push()
    new_stmts = Transform.transform_block(self, stmts)
    self.available_expressions.pop()
    return new_stmts

  def transform_If(self, stmt):
    stmt.true = self.transform_block(stmt.true)
    stmt.false = self.transform_block(stmt.false)
    stmt.merge = self.transform_merge(stmt.merge,
                                      left_block = stmt.true,
                                      right_block = stmt.false)
    stmt.cond = self.transform_expr(stmt.cond)
    return stmt

  def transform_loop_condition(self, expr, outer_block, loop_body, merge):
    """
    Normalize loop conditions so they are just simple variables
    """
    if self.is_simple(expr): 
      return self.transform_expr(expr)
    else:
      loop_carried_vars = [name for name in collect_var_names(expr)
                           if name in merge]
      if len(loop_carried_vars) == 0:
        return expr

      left_values = [merge[name][0] for name in loop_carried_vars]
      right_values = [merge[name][1] for name in loop_carried_vars]
      
      left_cond = subst.subst_expr(expr, dict(zip(loop_carried_vars, left_values)))
      if not self.is_simple(left_cond):
        left_cond = self.temp_in_block(left_cond, outer_block, name = "cond")
      
      right_cond = subst.subst_expr(expr, dict(zip(loop_carried_vars, right_values))) 
      if not self.is_simple(right_cond):
        right_cond = self.temp_in_block(right_cond, loop_body, name = "cond")
      
      cond_var = self.fresh_var(left_cond.type, "cond")
      merge[cond_var.name] = (left_cond, right_cond)
      return cond_var

  def transform_While(self, stmt):
    stmt.body = self.transform_block(stmt.body)
    stmt.merge = self.transform_merge(stmt.merge,
                                      left_block = self.blocks.current(),
                                      right_block = stmt.body)
    stmt.cond = \
        self.transform_loop_condition(stmt.cond,
                                      outer_block = self.blocks.current(),
                                      loop_body = stmt.body,
                                      merge = stmt.merge)
    return stmt

  def transform_Return(self, stmt):
    new_value = self.transform_expr(stmt.value)
    if new_value != stmt.value:
      if self.is_simple(new_value):
        stmt.value = new_value
      else:
        stmt.value = self.temp(new_value)
    return stmt
    