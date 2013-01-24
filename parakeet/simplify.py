import prims
import subst
import syntax
import syntax_helpers
import transform

from adverbs import AllPairs, Map, Reduce, Scan
from array_type import ArrayT 
from closure_type import ClosureT
from collect_vars import collect_var_names
from core_types import NoneT, ScalarT
from mutability_analysis import TypeBasedMutabilityAnalysis
from scoped_dict import ScopedDictionary
from syntax import AllocArray, Assign, ExprStmt
from syntax import Const, Var, Tuple, TupleProj, Closure, ClosureElt, Cast
from syntax import Slice, Index, Array, ArrayView, Attribute, Struct
from syntax import PrimCall, Call, TypedFn, Fn
from syntax_helpers import collect_constants, is_one, is_zero, all_constants
from syntax_helpers import get_types, slice_none_t, const_int
from transform import Transform
from tuple_type import TupleT
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
    # WARNING: this is only valid
    # if attributes are immutable
    elif c is Attribute:
      return (expr.value,)
    elif c is Slice:
      return (expr.start, expr.stop, expr.step)
    elif c is Cast:
      return (expr.value,)
    elif c is Map or c is AllPairs:
      return expr.args
    elif c is Scan or c is Reduce:
      args = tuple(expr.args)
      init = (expr.init,) if expr.init else ()
      return init + args
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
      elif c is AllocArray:
        return (expr.shape,)
      elif c is Attribute:
        return (expr.value,)
    return None

  def immutable(self, expr):
    c = expr.__class__
    if c is Const:
      return True
    elif c is Tuple or c is TupleProj or \
         c is Closure or c is ClosureElt:
      return True
    # WARNING: making attributes always immutable
    elif c in (Attribute, Struct, Slice, ArrayView):
      return True
    # elif c is Attribute and expr.value.type.__class__ is TupleT:
    #  return True
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
    Wrapper around Codegen.assign_temp which also updates bindings and
    use_counts
    """

    new_var = self.assign_temp(expr)
    self.bindings[new_var.name] = expr
    self.use_counts[new_var.name] = use_count
    return new_var

  def transform_expr(self, expr):
    if not self.is_simple(expr):
      stored = self.available_expressions.get(expr)
      if stored is not None:
        return stored
    return Transform.transform_expr(self, expr)

  def transform_Var(self, expr):
    name = expr.name
    prev_expr = expr
    while name in self.bindings:
      prev_expr = expr 
      expr = self.bindings[name]
      if expr.__class__ is Var:
        name = expr.name
      else:
        break
    if expr.__class__ is Const:
      return expr
    else:
      return prev_expr
  
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
      elif c is AllocArray:
        if expr.name == 'shape':
          return self.transform_expr(stored_v.shape)
        elif expr.name == 'strides':
          return self.transform_expr(stored_v.strides)
        elif expr.name == 'data':
          return self.transform_expr(stored_v.data)

    if v.__class__ is Struct:
      idx = v.type.field_pos(expr.name)
      return v.args[idx]
    elif v.__class__ is not Var:
      v = self.temp(v, "struct")
    if expr.value == v:
      return expr
    else:
      return Attribute(value = v, name = expr.name, type = expr.type)
    
  
  def transform_Closure(self, expr):
    expr.args = tuple(self.transform_args(expr.args))
    return expr

  def transform_Tuple(self, expr):
    expr.elts = tuple( self.transform_args(expr.elts))
    return expr

  def transform_TupleProj(self, expr):
    idx = expr.index
    assert isinstance(idx, int), \
        "TupleProj index must be an integer, got: " + str(idx)
    new_tuple = self.transform_expr(expr.tuple)

    if new_tuple.__class__ is Var and new_tuple.name in self.bindings:
      tuple_expr = self.bindings[new_tuple.name]
      if tuple_expr.__class__ is Tuple:
        return tuple_expr.elts[idx]
      elif tuple_expr.__class__ is Struct:
        return tuple_expr.args[idx]

    if not self.is_simple(new_tuple):
      new_tuple = self.assign_temp(new_tuple, "tuple")
    expr.tuple = new_tuple
    return expr

  def transform_ClosureElt(self, expr):
    idx = expr.index
    assert isinstance(idx, int), \
        "ClosureElt index must be an integer, got: " + str(idx)
    new_closure = self.transform_expr(expr.closure)
    if new_closure.__class__ is Var and new_closure.name in self.bindings:
      closure_expr = self.bindings[new_closure.name]
      if closure_expr.__class__ is Closure:
        return closure_expr.args[idx]

    if not self.is_simple(new_closure):
      new_closure = self.assign_temp(new_closure, "closure")
    expr.closure = new_closure
    return expr

  def transform_Call(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_args(expr.args)
    if fn.type.__class__ is ClosureT:
      closure_elts = self.closure_elts(fn)
      combined_args = closure_elts + args
      if fn.type.fn.__class__ is TypedFn:
        fn = fn.type.fn
      else:
        assert isinstance(fn.type.fn, Fn)
        import type_inference
        fn = type_inference.specialize(fn, get_types(combined_args))
      assert fn.return_type == expr.type
      return Call(fn, combined_args, type = fn.return_type)
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
    expr.elts = tuple(self.transform_args(expr.elts))
    return expr

  def transform_Index(self, expr):
    expr.value = self.transform_expr(expr.value)
    expr.index = self.transform_expr(expr.index)
    if expr.value.__class__ is Array and expr.index.__class__ is Const:
      assert isinstance(expr.index.value, (int, long)) and \
             len(expr.value.elts) > expr.index.value

      return expr.value.elts[expr.index.value]
    if expr.value.__class__ is not Var:
      expr.value = self.temp(expr.value, "array")
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
      x,y = args
      if is_zero(x):
        return y
      elif is_zero(y):
        return x
    elif prim == prims.multiply:
      x,y = args
      if is_one(x):
        return y
      elif is_one(y):
        return x
      elif is_zero(x):
        return x
      elif is_zero(y):
        return y
      
    elif prim == prims.divide:
      x,y = args 
      if is_one(y):
        return x
      
    elif prim == prims.power:
      x,y = args
      if is_one(y):
        return self.cast(x, expr.type)
      elif is_zero(y):
        return syntax_helpers.one(expr.type)
      elif y.__class__ is Const and y.value == 2:
        return self.cast(self.mul(x, x, "sqr"), expr.type)

    expr.args = args
    return expr 
  
  def transform_Reduce(self, expr):
    
    init = self.transform_expr(expr.init)
    if not self.is_simple(init):
      expr.init = self.assign_temp(init, 'init')
    else:
      expr.init = init
    expr.args = self.transform_args(expr.args)
    expr.fn = self.transform_expr(expr.fn)
    expr.combine = self.transform_expr(expr.combine)
    return expr  
  
  def temp_in_block(self, expr, block, name = None):
    """
    If we need a temporary variable not in the current top scope but in a
    particular block, then use this function. (this function also modifies the
    bindings dictionary)
    """
    var = self.fresh_var(expr.type, name)
    block.append(Assign(var, expr))
    self.bindings[var.name] = expr
    return var

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
    # lhs.value = self.transform_expr(lhs.value)
    lhs.index = self.transform_expr(lhs.index)
    if lhs.value.__class__ is Var:
      stored = self.bindings.get(lhs.value.name)
      if stored and stored.__class__ is Var:
        lhs.value = stored
    else:
      lhs.value = self.assign_temp(lhs.value, "array")
    return lhs

  def transform_lhs_Attribute(self, lhs):
    # lhs.value = self.transform_expr(lhs.value)
    return lhs

  def transform_ExprStmt(self, stmt):
    """Don't run an expression unless it possibly has a side effect"""

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
    if lhs_class is Var:
      if rhs.type.__class__ is NoneT and self.use_counts.get(lhs.name,0) == 0:
        return self.transform_stmt(ExprStmt(rhs))
      else:
        self.bind_var(lhs.name, rhs)
        if rhs_class is not Var and \
           rhs_class is not Const and \
           self.immutable(rhs):
          self.available_expressions.setdefault(rhs, lhs)
    elif lhs_class is Tuple:
      self.bind(lhs, rhs)
    # assigning x[i] = x[i]
    # does nothing
    elif lhs_class is Index:
      if rhs_class is Index and \
         lhs.value == rhs.value and \
         lhs.index == rhs.index:
        # kill effect-free writes like x[i] = x[i]
        return None
      elif rhs_class is Var and \
           lhs.value.__class__ is Var and \
           lhs.value.name == rhs.name and \
           lhs.index.type.__class__ is TupleT and \
           all(elt_t == slice_none_t for elt_t in lhs.index.type.elt_types):
        # also kill x[:] = x
        return None
      else:
        lhs = self.transform_lhs_Index(lhs)
        # when assigning x[j] = [1,2,3]
        # just rewrite it as a sequence of element assignments 
        # to avoid 
        if lhs.type.__class__ is ArrayT and \
           lhs.type.rank == 1 and \
           rhs.__class__ is Array:
          lhs_slice = self.assign_temp(lhs, "lhs_slice")
          for (elt_idx, elt) in enumerate(rhs.elts):
            lhs_idx = self.index(lhs_slice, const_int(elt_idx), temp = False)
            self.assign(lhs_idx, elt)
          return None
        elif not self.is_simple(rhs):
          rhs = self.assign_temp(rhs)
    else:
      assert lhs_class is Attribute
      assert False, "Considering making attributes immutable"
      lhs = self.transform_lhs_Attribute(lhs)

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

  def transform_If(self, stmt):
    stmt.true = self.transform_block(stmt.true)
    stmt.false = self.transform_block(stmt.false)
    stmt.merge = self.transform_merge(stmt.merge,
                                      left_block = stmt.true,
                                      right_block = stmt.false)
    stmt.cond = self.transform_expr(stmt.cond)
    return stmt

  def transform_loop_condition(self, expr, outer_block, loop_body, merge):
    """Normalize loop conditions so they are just simple variables"""

    if self.is_simple(expr):
      return self.transform_expr(expr)
    else:
      loop_carried_vars = [name for name in collect_var_names(expr)
                           if name in merge]
      if len(loop_carried_vars) == 0:
        return expr

      left_values = [merge[name][0] for name in loop_carried_vars]
      right_values = [merge[name][1] for name in loop_carried_vars]

      left_cond = subst.subst_expr(expr, dict(zip(loop_carried_vars,
                                                  left_values)))
      if not self.is_simple(left_cond):
        left_cond = self.temp_in_block(left_cond, outer_block, name = "cond")

      right_cond = subst.subst_expr(expr, dict(zip(loop_carried_vars,
                                                   right_values)))
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

  def transform_ForLoop(self, stmt):
    stmt.start = self.transform_expr(stmt.start)
    stmt.stop = self.transform_expr(stmt.stop)
    stmt.step = self.transform_expr(stmt.step)
    stmt.body = self.transform_block(stmt.body)
    stmt.merge = self.transform_merge(stmt.merge,
                                      left_block = self.blocks.current(),
                                      right_block = stmt.body)

    # if a loop is only going to run for one iteration, might as well get rid of
    # it
    if stmt.start.__class__ is Const and \
       stmt.stop.__class__ is Const and \
       stmt.step.__class__ is Const:
      if stmt.start.value >= stmt.stop.value:
        for (var_name, (input_value, _)) in stmt.merge.iteritems():
          var = Var(var_name, input_value.type)
          self.blocks.append(Assign(var, input_value))
        return None
      elif stmt.start.value + stmt.step.value >= stmt.stop.value:
        for (var_name, (input_value, _)) in stmt.merge.iteritems():
          var = Var(var_name, input_value.type)
          self.blocks.append(Assign(var, input_value))
        self.assign(stmt.var, stmt.start)
        self.blocks.top().extend(stmt.body)
        return None
    return stmt

  def transform_Return(self, stmt):
    new_value = self.transform_expr(stmt.value)
    """
    if new_value.__class__ is Var and \
       new_value.name in self.use_counts and \
       self.use_counts[new_value.name] == 1 and \
       new_value.name in self.bindings:
      stored = self.bindings[stmt.value.name]
      if self.immutable(stored) and stored.__class__ is not AllPairs:
        print "Replacing %s => %s" % (stmt, stored)
        stmt.value = stored
        return stmt
    """
    if new_value != stmt.value:
      stmt.value = new_value
    return stmt
