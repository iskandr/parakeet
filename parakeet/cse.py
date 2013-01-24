
from adverbs import Map, AllPairs, Reduce, Scan
from mutability_analysis import TypeBasedMutabilityAnalysis
from scoped_dict import ScopedDictionary
from syntax import Var, Const, Call, Closure, PrimCall, ClosureElt
from syntax import Array, ArrayView, Attribute, AllocArray
from syntax import Tuple, TupleProj, Slice, Cast, Struct
from transform import Transform 

class CSE(Transform):
  
  def pre_apply(self, fn):
    # which expressions have already been computed
    # and stored in some variable?
    self.available_expressions = ScopedDictionary()
    
    ma = TypeBasedMutabilityAnalysis()
    # which types have elements that might
    # change between two accesses?
    self.mutable_types = ma.visit_fn(fn)
  
  def transform_expr(self, expr):
    if self.is_simple(expr):
      return expr 
    stored = self.available_expressions.get(expr)
    if stored is not None:
      return stored
    else:
      return Transform.transform_expr(self, expr)
  
  def transform_block(self, stmts):
    self.available_expressions.push()
    new_stmts = Transform.transform_block(self, stmts)
    self.available_expressions.pop()
    return new_stmts
  
  def transform_Assign(self, stmt):
    stmt.rhs = self.transform_expr(stmt.rhs)
    
    if stmt.lhs.__class__ is Var and \
       not self.is_simple(stmt.rhs) and \
       self.immutable(stmt.rhs) and \
       stmt.rhs not in self.available_expressions:
      self.available_expressions[stmt.rhs] = stmt.lhs
    return stmt
  
  
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
