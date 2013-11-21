
import numpy as np 

from .. ndtypes import ArrayT, TupleT   
from .. syntax import TypedFn, Var
from ..analysis import SyntaxVisitor
from abstract_value import one, zero, Const, unknown, Tuple, Array, Struct, abstract_tuple  





def bind_list(arg_names, abstract_values):
  assert len(arg_names) == len(abstract_values)
  env = {}
  for name, abstract_value in zip(arg_names, abstract_values):
    env[name] = abstract_value
  return env 

_cache = {}
def symbolic_call(fn, symbolic_inputs):
  key = fn.cache_key, tuple(symbolic_inputs)
  if key in _cache:
    return _cache[key]
  analysis = FindConstantValues(fn, symbolic_inputs)
  return_value = analysis.visit_fn(fn)
  result = (analysis.env, return_value)
  _cache[key] = result 
  return result 

class FindConstantValues(SyntaxVisitor):
  def __init__(self, fn, abstract_values):
    self.env = bind_list(fn.arg_names, abstract_values)
    self.return_value = None 
    
  def visit_expr(self, expr):
    result = SyntaxVisitor.visit_expr(self, expr)
    if result is None: 
      return unknown
    else: 
      return result 
  
  def visit_fn(self, fn):
    SyntaxVisitor.visit_fn(self, fn)
    
    if self.return_value is None:
      return unknown 
    else:
      return self.return_value
  
  def visit_Call(self, expr):
    if expr.fn.__class__ is TypedFn: 
      args = self.visit_expr_list(expr.args)
      _, return_value = symbolic_call(expr.fn, args)
      return return_value 
    else:
      return unknown 

  def visit_Var(self, expr):
    return self.env.get(expr.name, unknown)
  
  def visit_Const(self, expr):
    if expr.value == 0:
      return zero
    elif expr.value == 1:
      return one 
    elif isinstance(expr.value, int):
      return Const(int(expr.value))
    else:
      return unknown
  
  def visit_Attribute(self, expr):
    value = self.visit_expr(expr.value)
    if value.__class__ is Tuple:
      # assume tuples lowered into structs at this point
      pos = expr.value.type.field_pos(expr.name)
      return value.elts[pos]
    elif value.__class__ is Array and expr.name == 'strides':
      return value.strides
    elif value.__class__ is Struct and expr.name in value.fields:
      return value.fields[expr.name]
    else:
      return unknown

  def visit_TupleProj(self, expr):
    value = self.visit_expr(expr.tuple)
    if isinstance(value, Tuple):
      return value.elts[expr.index]
    else:
      return unknown 
    
  def visit_PrimCall(self, expr):
    abstract_values = self.visit_expr_list(expr.args)
    if all(v.__class__ is Const for v in abstract_values):
      ints = [v.value for v in abstract_values]
      return Const(expr.prim.fn(*ints))
    else:
      return unknown

  def visit_Struct(self, expr):
    if expr.type.__class__ is TupleT:
      return abstract_tuple(self.visit_expr_list(expr.args))
    elif expr.type.__class__ is ArrayT:
      stride_pos = expr.type.field_pos("strides")
      stride_arg = expr.args[stride_pos]
      stride_val = self.visit_expr(stride_arg)
      return Array(stride_val)
    else:
      return unknown   
    
  def visit_Alloc(self, expr):
    return unknown 
  
  def visit_Index(self, expr):
    return unknown 
  
  def visit_Assign(self, stmt):
    if stmt.lhs.__class__ is Var:
      value = self.visit_expr(stmt.rhs)
      assert value is not None, \
        "%s : %s returned None in ConstantStride analysis" % (stmt.rhs, stmt.rhs.node_type()) 
      self.env[stmt.lhs.name] = value 
      
  def visit_Return(self, stmt):
    value = self.visit_expr(stmt.value)
    if self.return_value is None:
      self.return_value = value 
    elif self.return_value != value:
      self.return_value = unknown 