import numpy as np 

import syntax
import ast_conversion 
from function_registry import untyped_functions
from common import dispatch 
from core_types import ScalarT, StructT
import types 

class ReturnValue(Exception):
  def __init__(self, value):
    self.value = value 


class ClosureVal:
  def __init__(self, fn, fixed_args):
    self.fn = fn 
    self.fixed_args = fixed_args

from args import match 

import adverb_helpers

def eval_fn(fn, actuals):
  env = fn.args.bind(actuals)
    
  def eval_expr(expr): 
    assert isinstance(expr, syntax.Expr), "Not an expression: %s" % expr    
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
      arg_vals = map(eval_expr, expr.args)

      return expr.prim.fn (*arg_vals)
    
    def expr_Call():
      fundef = untyped_functions[expr.fn]
      arg_vals = map(eval_expr, expr.args)
      return eval_fn(fundef, *arg_vals) 
    
    def expr_Prim():
      return expr.value.fn
    
    def expr_Slice():
      return slice(eval_expr(expr.lower), eval_expr(expr.upper), eval_expr(expr.step)) 
    
    def expr_Var():
      return env[expr.name]
    
    def expr_Invoke():
      # for the interpreter Invoke and Call are identical since
      # we're dealing with runtime reprs for functions, prims, and 
      # closures which are just python Callables
      clos = eval_expr(expr.closure)
      arg_vals = map(eval_expr, expr.args)
      combined_arg_vals = clos.fixed_args + arg_vals
      return eval_fn(clos.fn, combined_arg_vals)
      
    def expr_Closure():
      fundef = untyped_functions[expr.fn]
      closure_arg_vals = map(eval_expr, expr.args) 
      return ClosureVal(fundef, closure_arg_vals)
    
    def expr_Cast():
      x = eval_expr(expr.value)
      t = expr.type
      assert isinstance(t, ScalarT)
      # use numpy's conversion function 
      return t.dtype.type(x)
    
    def expr_Struct():
      assert expr.type, "Expected type on %s!" % expr 
      assert isinstance(expr.type, StructT), "Expected %s : %s to be a struct" % (expr, expr.type)
      elts = map(eval_expr, expr.args)
      return expr.type.ctypes_repr(elts)
    
    def expr_Tuple():
      return tuple(map(eval_expr, expr.elts))
    
    def expr_TupleProj():
      return eval_expr(expr.tuple)[expr.index]
    
    def call(fn, args):
      if isinstance(fn, ClosureVal):
        return eval_fn(fn.fn, fn.fixed_args + args)
      else:
        return fn(*args)
    
    def create_slice_idx(ndims, dim, i):
      return tuple([i if d == dim else None for d in range(ndims)])
    
    def nested_arg(arg, dim, i):
      if isinstance(arg, np.ndarray):
        idx = create_slice_idx(np.rank(arg), dim, i)
        return arg[idx]
      else:
        return arg 

    def tile(x, outer_shape):
      if isinstance(x, np.ndarray):
        return np.tile(x, tuple(outer_shape) + (1,))
      return np.tile(x, tuple(outer_shape))
    
    def expr_Map():
      fn = eval_expr(expr.fn)
      args = map(eval_expr, expr.args)
      max_shape = ()
      for arg in args:
        if isinstance(arg, np.ndarray):
          assert max_shape == () or max_shape == arg.shape
          max_shape = arg.shape
      assert len(max_shape) == 1
      n = max_shape[0]
      result = [None] * n 
      axis = 0
      for i in xrange(n):
        nested_args = [nested_arg(arg, axis, i) for arg in args]
        result[i]  = call(fn, nested_args)
      return np.array(result)
    
    def expr_AllPairs():
      fn = eval_expr(expr.fn)
      assert len(expr.args) == 2
      x = eval_expr(expr.args[0])
      y = eval_expr(expr.args[1])
      
      if expr.axis is None:
        axis = 0
        x = np.ravel(x)
        y = np.ravel(y)
      else:
        import syntax_helpers
        axis = syntax_helpers.unwrap_constant(expr.axis)
         
      nx = x.shape[axis]
      ny = y.shape[axis]
      first_x = nested_arg(x, axis, 0)
      first_args = [first_x, nested_arg(y, axis, 0)]
      print fn, first_args
      first_elt = call(fn, first_args)
      result = tile(first_elt, (nx, ny))
      for j in xrange(ny):
        result[0,j] = call(fn, [first_x, nested_arg(y, axis, j)])
      for i in xrange(1, nx):
        for j in xrange(0, ny):
          args = [nested_arg(x, axis, i), nested_arg(y, axis, j)] 
          result[i, j] = call(fn, args)
      return result 
      
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
      assign(stmt.lhs, eval_expr(stmt.rhs), env)
      
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
    else: 
      raise RuntimeError("Not implemented: %s" % stmt)
    
  def eval_block(stmts):
    #print "eval", stmts 
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
  actuals = args.Args(all_positional, kwds)
  return eval_fn(untyped, actuals) 
  