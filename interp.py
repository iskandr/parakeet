import numpy as np 

import syntax
import ast_conversion 
from function_registry import untyped_functions
from common import dispatch 
from core_types import ScalarT, StructT
import type_conv


class ReturnValue(Exception):
  def __init__(self, value):
    self.value = value 


class Closure:
  def __init__(self, fn, fixed_args):
    self.fn = fn 
    self.fixed_args = fixed_args

from args import match 
  
def eval_fn(fn, actuals):

  env = fn.args.bind(actuals)
    
  def eval_expr(expr): 
    assert isinstance(expr, syntax.Expr), "Not an expression: %s" % expr    
    def expr_Const():
      return expr.value
    
    
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
      return Closure(fundef, closure_arg_vals)
    
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
    
    return dispatch(expr, 'expr')
      
  def eval_merge_left(phi_nodes):
    for result, (left, _) in phi_nodes.iteritems():
      env[result] = eval_expr(left)
      
  def eval_merge_right(phi_nodes):
    for result, (_, right) in phi_nodes.iteritems():
      env[result] = eval_expr(right)
       
  def eval_stmt(stmt):
    if isinstance(stmt, syntax.Return):
      v = eval_expr(stmt.value)

      raise ReturnValue(v)
    elif isinstance(stmt, syntax.Assign):
      match(stmt.lhs, eval_expr(stmt.rhs), env)
      
    elif isinstance(stmt, syntax.If):
      cond_val = eval_expr(stmt.cond)
      if cond_val:
        eval_block(stmt.true)
        eval_merge_left(stmt.merge)
      else:
        eval_block(stmt.false)
        eval_merge_right(stmt.merge)
        
        
    elif isinstance(stmt, syntax.While):
      eval_merge_left(stmt.merge_before)
      ran_once = False
      while eval_expr(stmt.cond):
        ran_once = True
        eval_block(stmt.body)
        eval_merge_right(stmt.merge_before)
      if ran_once:
        eval_merge_right(stmt.merge_after)
      else:
        eval_merge_left(stmt.merge_after)
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
  