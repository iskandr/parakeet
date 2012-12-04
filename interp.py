import numpy as np 

import syntax
import ast_conversion 
from common import dispatch 
from core_types import ScalarT, StructT
import types 
import syntax_helpers


from adverb_interp import adverb_evaluator
from args import ActualArgs

class ReturnValue(Exception):
  def __init__(self, value):
    self.value = value 


class ClosureVal:
  def __init__(self, fn, fixed_args):
    self.fn = fn 
    self.fixed_args = fixed_args
    
  def __call__(self, *args):
    return call(self, list(args))
 
 
def call(fn, args):
  if isinstance(fn, ClosureVal):
    return eval_fn(fn.fn, fn.fixed_args + args)
  else:
    return fn(*args)

def create_slice_idx(ndims, dim, i):
  return tuple([i if d == dim else Ellipsis for d in range(ndims)])
    
def nested_arg(arg, dim, i):
  if isinstance(arg, np.ndarray):
    idx = create_slice_idx(np.rank(arg), dim, i)
    
    result = arg[idx]
    return result 
  else:
    return arg 

def nested_args(args, dim, i):
  return [nested_arg(arg, dim, i) for arg in args]

def tile_array(x, outer_shape):
  if isinstance(x, np.ndarray):
    return np.tile(x, tuple(outer_shape) + (1,))
  return np.tile(x, tuple(outer_shape))
    
def max_shape(vals):
  result = ()
  for arg in vals:
    if isinstance(arg, np.ndarray):
      assert result == () or result == arg.shape
      result = arg.shape
  return result 
    
def ravel_list(xs):
  return [np.ravel(x) if isinstance(x, np.ndarray) else x for x in xs]  

def eval_fn(fn, actuals):
  if hasattr(fn, 'arg_names'):
    env = {}
    for (k,v) in zip(fn.arg_names, actuals): 
      env[k] = v
  else:
    # untyped functions have a more complicated args object
    # which deals with named args, variable arity, etc.. 
    env = fn.args.bind(actuals)

  
  def eval_args(args):
    if isinstance(args, (list, tuple)):
      return map(eval_expr, args)
    else:
      return args.transform(eval_expr)
       
  
  def eval_expr(expr): 
    assert isinstance(expr, syntax.Expr), "Not an expression-- %s : %s" % (expr, type(expr))    
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

    
    def expr_Prim():
      return expr.value.fn
    
    def expr_Slice():
      return slice(eval_expr(expr.start), eval_expr(expr.stop), eval_expr(expr.step)) 
    
    def expr_Var():
      return env[expr.name]
    
    def expr_Call():
      # for the interpreter Invoke and Call are identical since
      # we're dealing with runtime reprs for functions, prims, and 
      # closures which are just python Callables
 
        
      clos = eval_expr(expr.fn)
      arg_values = eval_args(expr.args)
      
      if isinstance(clos, ClosureVal):
        fn = clos.fn 
        closure_args = clos.fixed_args
      else:
        fn = clos 
        closure_args = []
      
      if isinstance(arg_values, list):
        combined_arg_vals = closure_args + arg_values  
      else:
        assert isinstance(expr.args, ActualArgs)
        combined_arg_vals = arg_values.prepend_positional(closure_args)
      if isinstance(fn, (syntax.TypedFn, syntax.Fn)): 
        return eval_fn(fn, combined_arg_vals)
      else:
        assert hasattr(fn, '__call__'), \
          "Unexpected function %s" % (fn,)
        fn(*combined_arg_vals)
      
    def expr_Closure():
      if isinstance(expr.fn, syntax.Fn):
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
      assert isinstance(expr.type, StructT), "Expected %s : %s to be a struct" % (expr, expr.type)
      elts = map(eval_expr, expr.args)
      return expr.type.ctypes_repr(elts)
    
    def expr_Tuple():
      return tuple(map(eval_expr, expr.elts))
    
    def expr_TupleProj():
      return eval_expr(expr.tuple)[expr.index]
    
    def expr_ClosureElt():
      assert isinstance(expr.closure, syntax.Expr), \
        "Invalid closure expression-- %s : %s" % (expr.closure, type(expr.closure))
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
  
