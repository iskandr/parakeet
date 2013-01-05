import adverb_semantics
import ast_conversion
import numpy as np
import syntax
import syntax_helpers
import types

from args import ActualArgs
from common import dispatch
from core_types import ScalarT, StructT

class InterpSemantics(adverb_semantics.AdverbSemantics):
  def size_along_axis(self, value, axis):
    assert len(value.shape) > axis, \
        "Can't get %d'th element of %s with shape %s" % \
        (axis, value, value.shape)
    return value.shape[axis]

  def is_tuple(self, x):
    return isinstance(x, tuple)

  def is_none(self, x):
    return x is None

  def rank(self, value):
    return np.rank(value)

  def int(self, x):
    return int(x)

  def bool(self, x):
    return bool(x)

  def add(self, x, y):
    return x + y

  def sub(self, x, y):
    return x - y

  def shape(self, x):
    return np.shape(x)

  def elt_type(self, x):
    return x.dtype if hasattr(x, 'dtype') else type(x)

  def alloc_array(self, elt_type, dims):
    return np.zeros(dims, dtype = elt_type)

  def shift_array(self, arr, offset):
    return arr[offset:]

  def index(self, arr, idx):
    return arr[idx]

  def tuple(self, elts):
    return tuple(elts)

  def concat_tuples(self, t1, t2):
    return tuple(t1) + tuple(t2)

  def setidx(self, arr, idx, v):
    arr[idx] = v

  def loop(self, start_idx, stop_idx, body):
    for i in xrange(start_idx, stop_idx):
      body(i)

  class Accumulator:
    def __init__(self, value):
      self.value = value

    def get(self):
      return self.value

    def update(self, new_value):
      self.value = new_value

  def accumulate_loop(self, start_idx, stop_idx, body, init):
    acc = self.Accumulator(init)
    for i in xrange(start_idx, stop_idx):
      body(acc, i)
    return acc.get()

  def check_equal_sizes(self, sizes):
    assert len(sizes) > 0
    first = sizes[0]
    assert all(sz == first for sz in sizes[1:])

  def slice_value(self, start, stop, step):
    return slice(start, stop, step)

  def invoke(self, fn, args):
    return eval_fn(fn, args)

  none = None
  null_slice = slice(None, None, None)
  def identity_function(self, x):
    return x

adverb_evaluator = InterpSemantics()

class ReturnValue(Exception):
  def __init__(self, value):
    self.value = value

class ClosureVal:
  def __init__(self, fn, fixed_args):
    self.fn = fn
    self.fixed_args = tuple(fixed_args)

  def __call__(self, args):
    if isinstance(args, ActualArgs):
      args = args.prepend_positional(self.fixed_args)
    else:
      args = self.fixed_args + tuple(args)
    return eval_fn(self.fn, args)

def eval_fn(fn, actuals):
  if isinstance(fn, syntax.TypedFn):
    assert len(fn.arg_names) == len(actuals), \
      "Wrong number of args, expected %s but given %s" % \
      (fn.arg_names, actuals)
    env = {}

    for (k,v) in zip(fn.arg_names, actuals):
      env[k] = v
  elif isinstance(fn, syntax.Fn):
    # untyped functions have a more complicated args object
    # which deals with named args, variable arity, etc..
    env = fn.args.bind(actuals)
  elif isinstance(fn, ClosureVal):
    return fn(actuals)
  else:
    return fn(*actuals)

  def eval_args(args):
    if isinstance(args, (list, tuple)):
      return map(eval_expr, args)
    else:
      return args.transform(eval_expr)

  def eval_expr(expr):
    if hasattr(expr, 'wrapper'):
      expr = expr.wrapper
    assert isinstance(expr, syntax.Expr), "Not an expression-- %s : %s" % \
         (expr, type(expr))
    def expr_Const():
      return expr.value

    def expr_Attribute():
      value = eval_expr(expr.value)
      return getattr(value, expr.name)

    def expr_Array():
      elt_values = map(eval_expr, expr.elts)
      return np.array(elt_values)

    def expr_Index():
      array = eval_expr(expr.value)
      index = eval_expr(expr.index)
      return array[index]

    def expr_PrimCall():
      return expr.prim.fn (*eval_args(expr.args))
    
    def expr_Slice():
      return slice(eval_expr(expr.start), eval_expr(expr.stop),
                   eval_expr(expr.step))

    def expr_Var():
      return env[expr.name]

    def expr_Call():
      fn = eval_expr(expr.fn)
      arg_values = eval_args(expr.args)
      return eval_fn(fn, arg_values)

    def expr_Closure():
      if isinstance(expr.fn, (syntax.Fn, syntax.TypedFn)):
        fundef = expr.fn
      else:
        assert isinstance(expr.fn, str)
        fundef = syntax.Fn.registry[expr.fn]
      closure_arg_vals = map(eval_expr, expr.args)
      return ClosureVal(fundef, closure_arg_vals)

    def expr_Fn():
      return ClosureVal(expr, [])

    def expr_TypedFn():
      return ClosureVal(expr, [])

    def expr_Cast():
      x = eval_expr(expr.value)
      t = expr.type
      assert isinstance(t, ScalarT)
      # use numpy's conversion function
      return t.dtype.type(x)

    def expr_Struct():
      assert expr.type, "Expected type on %s!" % expr
      assert isinstance(expr.type, StructT), \
          "Expected %s : %s to be a struct" % (expr, expr.type)
      elts = map(eval_expr, expr.args)
      return expr.type.ctypes_repr(elts)

    def expr_Tuple():
      return tuple(map(eval_expr, expr.elts))

    def expr_TupleProj():
      return eval_expr(expr.tuple)[expr.index]

    def expr_ClosureElt():
      assert isinstance(expr.closure, syntax.Expr), \
          "Invalid closure expression-- %s : %s" % \
          (expr.closure, type(expr.closure))
      clos = eval_expr(expr.closure)
      return clos.fixed_args[expr.index]

    def expr_Map():
      fn = eval_expr(expr.fn)
      args = eval_args(expr.args)
      axis = syntax_helpers.unwrap_constant(expr.axis)
      return adverb_evaluator.eval_map(fn, args, axis)

    def expr_AllPairs():
      fn = eval_expr(expr.fn)
      x,y = eval_args(expr.args)
      axis = syntax_helpers.unwrap_constant(expr.axis)
      return adverb_evaluator.eval_allpairs(fn, x, y, axis)

    def expr_Reduce():
      map_fn = eval_expr(expr.fn)
      combine_fn = eval_expr(expr.combine)
      args = eval_args(expr.args)
      init = eval_expr(expr.init) if expr.init else None
      axis = syntax_helpers.unwrap_constant(expr.axis)
      return adverb_evaluator.eval_reduce(map_fn, combine_fn, init, args, axis)

    def expr_Scan():
      map_fn = eval_expr(expr.fn)
      combine = eval_expr(expr.combine)
      emit = eval_expr(expr.emit)
      args = eval_args(expr.args)
      init = eval_expr(expr.init)
      axis = syntax_helpers.unwrap_constant(expr.axis)
      return adverb_evaluator.eval_scan(map_fn, combine, emit, init, args, axis)

    result = dispatch(expr, 'expr')
    # we don't support python function's inside parakeet,
    # they have to be translated into Parakeet functions
    if isinstance(result, types.FunctionType):
      fundef = ast_conversion.translate_function_value(result)
      return ClosureVal(fundef, fundef.python_nonlocals())
    else:
      return result

  def eval_merge_left(phi_nodes):
    for result, (left, _) in phi_nodes.iteritems():
      env[result] = eval_expr(left)

  def eval_merge_right(phi_nodes):
    for result, (_, right) in phi_nodes.iteritems():
      env[result] = eval_expr(right)

  def assign(lhs, rhs, env):
    if isinstance(lhs, syntax.Var):
      env[lhs.name] = rhs
    elif isinstance(lhs, syntax.Tuple):
      assert isinstance(rhs, tuple)
      for (elt, v) in zip(lhs.elts, rhs):
        assign(elt, v, env)
    elif isinstance(lhs, syntax.Index):
      arr = eval_expr(lhs.value)
      idx = eval_expr(lhs.index)
      arr[idx] = rhs

  def eval_stmt(stmt):
    if isinstance(stmt, syntax.Return):
      v = eval_expr(stmt.value)
      raise ReturnValue(v)
    elif isinstance(stmt, syntax.Assign):
      value = eval_expr(stmt.rhs)
      assign(stmt.lhs, value, env)

    elif isinstance(stmt, syntax.If):
      cond_val = eval_expr(stmt.cond)
      if cond_val:
        eval_block(stmt.true)
        eval_merge_left(stmt.merge)
      else:
        eval_block(stmt.false)
        eval_merge_right(stmt.merge)

    elif isinstance(stmt, syntax.While):
      eval_merge_left(stmt.merge)
      while eval_expr(stmt.cond):
        eval_block(stmt.body)
        eval_merge_right(stmt.merge)
    elif isinstance(stmt, syntax.ForLoop):
      start = eval_expr(stmt.start)
      stop = eval_expr(stmt.stop)
      step = eval_expr(stmt.step)
      eval_merge_left(stmt.merge)
      for i in xrange(start, stop, step):
        env[stmt.var.name] = i
        eval_block(stmt.body)
        eval_merge_right(stmt.merge)
    else:
      raise RuntimeError("Not implemented: %s" % stmt)

  def eval_block(stmts):
    for stmt in stmts:
      eval_stmt(stmt)

  try:
   
    eval_block(fn.body)

  except ReturnValue as r:
    return r.value
  except:
    raise

def run_python_fn(python_fn, args, kwds):
  untyped  = ast_conversion.translate_function_value(python_fn)
  # should eventually roll this up into something cleaner, since
  # top-level functions are really acting like closures over their
  # global dependencies
  global_args = [python_fn.func_globals[n] for n in untyped.nonlocals]
  all_positional = global_args + list(args)
  actuals = args.FormalArgs(all_positional, kwds)
  return eval_fn(untyped, actuals)
