import config
import verify

import syntax
from syntax import If, Assign, While, Return, RunExpr
from syntax import Var, Tuple, Index, Attribute, Const
from syntax import PrimCall, Struct, Alloc, Cast
from syntax import TupleProj, Slice, ArrayView
from syntax import Call, TypedFn

from args import ActualArgs
from codegen import Codegen

class Transform(Codegen):
  def __init__(self, verify = config.opt_verify,
                     reverse = False,
                     require_types = True):
    Codegen.__init__(self)
    self.fn = None
    self.verify = verify
    self.reverse = reverse
    self.require_types = require_types

  def lookup_type(self, name):
    assert self.type_env is not None
    return self.type_env[name]

  def transform_if_expr(self, maybe_expr):
    if isinstance(maybe_expr, syntax.Expr):
      return self.transform_expr(maybe_expr)
    elif isinstance(maybe_expr, tuple):
      return tuple([self.transform_if_expr(x) for x in maybe_expr])
    elif isinstance(maybe_expr, list):
      return [self.transform_if_expr(x) for x in maybe_expr]
    elif isinstance(maybe_expr, ActualArgs):
      return maybe_expr.transform(self.transform_expr)
    else:
      return maybe_expr

  def transform_generic_expr(self, expr):
    for member_name in expr.members():
      old_value = getattr(expr, member_name)
      new_value = self.transform_if_expr(old_value)
      setattr(expr, member_name, new_value)
    return expr

  def find_method(self, expr, prefix = "transform_"):
    method_name = prefix + expr.node_type()

    if hasattr(self, method_name):
      return getattr(self, method_name)
    else:
      return None

  """
  Common cases for expression transforms: we don't need to create a method for
  every sort of expression but these run faster and allocate less memory than
  transform_generic_expr
  """
  def transform_Var(self, expr):
    return expr

  def transform_Tuple(self, expr):
    expr.elts = tuple(self.transform_expr(elt) for elt in expr.elts)
    return expr

  def transform_Const(self, expr):
    return expr

  def transform_Index(self, expr):
    expr.value = self.transform_expr(expr.value)
    expr.index = self.transform_expr(expr.index)
    return expr

  def transform_Attribute(self, expr):
    expr.value = self.transform_expr(expr.value)
    return expr

  def transform_PrimCall(self, expr):
    expr.args = self.transform_expr_tuple(expr.args)
    return expr

  def transform_Call(self, expr):
    expr.fn = self.transform_expr(expr.fn)
    expr.args = self.transform_expr_tuple(expr.args)
    return expr

  def transform_Alloc(self, expr):
    expr.count = self.transform_expr(expr.count)
    return expr

  def transform_Struct(self, expr):
    expr.args = self.transform_expr_tuple(expr.args)
    return expr

  def transform_Cast(self, expr):
    expr.value = self.transform_expr(expr.value)
    return expr

  def transform_TupleProj(self, expr):
    expr.tuple = self.transform_expr(expr.tuple)
    return expr

  def transform_TypedFn(self, expr):
    """
    By default, don't do recursive transformation of referenced functions
    """

    return expr

  def transform_Slice(self, expr):
    expr.start = self.transform_expr(expr.start) if expr.start else None
    expr.stop = self.transform_expr(expr.stop) if expr.stop else None
    expr.step = self.transform_expr(expr.step) if expr.step else None
    return expr

  def transform_ArrayView(self, expr):
    expr.data = self.transform_expr(expr.data)
    expr.shape = self.transform_expr(expr.shape)
    expr.strides = self.transform_expr(expr.strides)
    expr.offset = self.transform_expr(expr.offset)
    expr.total_elts = self.transform_expr(expr.total_elts)
    return expr

  def transform_expr(self, expr):
    """
    Dispatch on the node type and call the appropriate transform method
    """

    expr_class = expr.__class__
    if expr_class is Var:
      result = self.transform_Var(expr)
    elif expr_class is Const:
      result = self.transform_Const(expr)
    elif expr_class is Tuple:
      result = self.transform_Tuple(expr)
    elif expr_class is TupleProj:
      result = self.transform_TupleProj(expr)
    elif expr_class is Index:
      result = self.transform_Index(expr)
    elif expr_class is Slice:
      result = self.transform_Slice(expr)
    elif expr_class is Attribute:
      result = self.transform_Attribute(expr)
    elif expr_class is PrimCall:
      result = self.transform_PrimCall(expr)
    elif expr_class is Struct:
      result = self.transform_Struct(expr)
    elif expr_class is Alloc:
      result = self.transform_Alloc(expr)
    elif expr_class is Cast:
      result = self.transform_Cast(expr)
    elif expr_class is ArrayView:
      result = self.transform_ArrayView(expr)
    elif expr_class is TypedFn:
      result = self.transform_TypedFn(expr)
    elif expr_class is Call:
      result = self.transform_Call(expr)

    else:
      method = self.find_method(expr, "transform_")
      if method:
        result = method(expr)
      else:
        result = self.transform_generic_expr(expr)
    if result is None:
      return expr
    else:
      return result

  def transform_lhs_Var(self, expr):
    return self.transform_Var(expr)

  def transform_lhs_Tuple(self, expr):
    return self.transform_Tuple(expr)

  def transform_lhs_Index(self, expr):
    return self.transform_Index(expr)

  def transform_lhs_Attribute(self, expr):
    return self.transform_Attribute(expr)

  def transform_lhs(self, lhs):
    """
    Overload this is you want different behavior for transformation of left-hand
    side of assignments
    """

    lhs_class = lhs.__class__
    if lhs_class is Var:
      return self.transform_lhs_Var(lhs)
    elif lhs_class is Tuple:
      return self.transform_lhs_Tuple(lhs)
    elif lhs_class is Index:
      return self.transform_lhs_Index(lhs)
    elif lhs_class is Attribute:
      return self.transform_lhs_Attribute(lhs)

    lhs_method = self.find_method(lhs, prefix = "transform_lhs_")
    if lhs_method:
      return lhs_method(lhs)

    method = self.find_method(lhs, prefix = "transform_")
    assert method, "Unknown expression of type %s" % lhs_class
    return method(lhs)

  def transform_expr_list(self, exprs):
    return [self.transform_expr(e) for e in exprs]

  def transform_expr_tuple(self, exprs):
    return tuple(self.transform_expr_list(exprs))
  def transform_merge(self, phi_nodes):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      new_left = self.transform_expr(left)
      new_right = self.transform_expr(right)
      result[k] = new_left, new_right
    return result

  def transform_Assign(self, stmt):
    stmt.rhs = self.transform_expr(stmt.rhs)
    stmt.lhs = self.transform_lhs(stmt.lhs)
    return stmt

  def transform_RunExpr(self, stmt):
    stmt.value = self.transform_expr(stmt.value)
    return stmt

  def transform_Return(self, stmt):
    stmt.value = self.transform_expr(stmt.value)
    return stmt

  def transform_If(self, stmt):
    stmt.true = self.transform_block(stmt.true)
    stmt.false = self.transform_block(stmt.false)
    stmt.merge = self.transform_merge(stmt.merge)
    stmt.cond = self.transform_expr(stmt.cond)
    return stmt

  def transform_While(self, stmt):
    stmt.body = self.transform_block(stmt.body)
    stmt.merge = self.transform_merge(stmt.merge)
    stmt.cond = self.transform_expr(stmt.cond)
    return stmt

  def transform_stmt(self, stmt):
    stmt_class = stmt.__class__
    if stmt_class is Assign:
      return self.transform_Assign(stmt)
    elif stmt_class is While:
      return self.transform_While(stmt)
    elif stmt_class is If:
      return self.transform_If(stmt)
    elif stmt_class is Return:
      return self.transform_Return(stmt)
    elif stmt_class is RunExpr:
      return self.transform_RunExpr(stmt)
    else:
      assert False, "Unexpected statement %s" % stmt_class

  def transform_block(self, stmts):
    self.blocks.push()
    for old_stmt in (reversed(stmts) if self.reverse else stmts):
      new_stmt = self.transform_stmt(old_stmt)
      if new_stmt is not None:
        self.blocks.append_to_current(new_stmt)
    new_block = self.blocks.pop()
    if self.reverse:
      new_block.reverse()
    return new_block

  def pre_apply(self, old_fn):
    pass

  def post_apply(self, new_fn):
    pass

  def apply(self, fn):
    self.fn = fn

    if config.print_functions_before_transforms:
      print
      print "Running transform %s" % self.__class__.__name__
      print "--- before ---"
      print repr(self.fn)
      print

    fn = self.pre_apply(self.fn)

    if fn is None:
      fn = self.fn

    self.type_env = fn.type_env
    fn.body = self.transform_block(fn.body)
    fn.type_env = self.type_env
    new_fn = self.post_apply(fn)
    if new_fn is None:
      new_fn = fn

    if config.print_functions_after_transforms:
      print
      print "Done with  %s" % self.__class__.__name__
      print "--- after ---"
      print repr(new_fn)
      print

    if self.verify:
      verify.verify(new_fn)
    return new_fn

class MemoizedTransform(Transform):
  _cache = {}

  def apply(self, fn):
    key = (self.__class__.__name__, fn.name)
    if key in self._cache:
      return self._cache[key]
    else:
      new_fn = Transform.apply(self, fn)
      self._cache[key] = new_fn
      return new_fn

def apply_pipeline(fn, transforms):
  for T in transforms:
    if type(T) == type:
      fn = T().apply(fn)
    else:
      assert isinstance(T, Transform)
      fn = T.apply(fn)
  return fn
