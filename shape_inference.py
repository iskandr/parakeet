import syntax
from syntax_visitor import SyntaxVisitor
from syntax_helpers import get_types
import type_inference  

import core_types 
import array_type 
import tuple_type 
import closure_type 
from symbolic_shape import \
  Var, Const, Shape, Tuple, Closure, Slice, Scalar, UnknownScalar, Unknown 
from symbolic_shape import unknown_scalar, unknown_value, const, array 
from symbolic_shape import combine_list, increase_rank, lower_ranks, dim_list 
from symbolic_shape import is_one, is_zero 

import symbolic_shape  

class InputConverter():
  """
  Turn a list of input types into a list of abstract values, 
  numbering the input arrays and scalars but preserving the 
  structure of tuples, closures, and slices
  """
  def __init__(self):
    self.counter = 0
  
  def fresh_var(self):
    n = self.counter 
    self.counter += 1
    return Var(n)
    
  def value_from_type(self, t):
    if isinstance(t, core_types.ScalarT):
      return self.fresh_var()
    elif isinstance(t, array_type.ArrayT):
      dim_vars = [self.fresh_var() for _ in range(t.rank)]
      return Shape(dim_vars) 
    elif isinstance(t, tuple_type.TupleT):
      elt_values = self.values_from_types(t.elt_types)
      return Tuple(elt_values) 
    elif isinstance(t, array_type.SliceT):
      start = self.value_from_type(t.start_type)
      stop = self.value_from_type(t.stop_type)
      step = self.value_from_type(t.step_type)
      return Slice(start, stop, step)
    elif isinstance(t, closure_type.ClosureT):
      arg_vals = self.values_from_types(t.arg_types)
      return Closure(t.fn, arg_vals) 
    else:
      assert False, "Unsupported type: %s" % t
    
  def values_from_types(self, arg_types):
    values = []
    for t in arg_types:
      v = self.value_from_type(t)
      values.append(v)
    return values

import adverb_semantics

class ShapeSemantics(adverb_semantics.AdverbSemantics):
  def size_along_axis(self, value, axis):
    assert isinstance(value, Shape)
    return value.dims[axis]
    
  def is_tuple(self, x):
    return isinstance(x, Tuple)

  def is_none(self, x):
    return isinstance(x, Const) and x.value is None 

  def rank(self, value):
    if isinstance(value, Shape):
      return const(value.rank)
    else:
      return const(0) 

  def int(self, x):
    return const(x)

  def bool(self, x):
    return const(x)

  def add(self, x, y):
    if is_zero(x):
      return y 
    elif is_zero(y):
      return x
    elif isinstance(x, Const) and isinstance(y, Const):
      return const(x.value + y.value)
    else:
      return symbolic_shape.Add(x,y)

  def sub(self, x, y):
    if is_zero(y):
      return x
    elif isinstance(x, Const) and isinstance(y, Const):
      return const(x.value - y.value)

  def shape(self, x):
    return Tuple(x.dims)
    
  def elt_type(self, x):
    return "DON'T CARE ABOUT ELT TYPES"
  
  def alloc_array(self, _, dims):
    return Shape(dims)
  
  def index(self, arr, idx):
    return unknown_scalar

  def tuple(self, elts):
    return Tuple(tuple(elts))

  def concat_tuples(self, t1, t2):
    return Tuple(t1.elts + t2.elts)
  
  def setidx(self, arr, idx, v):
    pass 

  def loop(self, start_idx, stop_idx, body):
    body(start_idx)
  
  class Accumulator(object):
    def __init__(self, v):
      self.v = v 
      
    def update(self, new_v):
      self.v = new_v 
  
    def get(self):
      return self.v 
    
  def accumulate_loop(self, start_idx, stop_idx, body, init):
    acc = self.Accumulator(init)
    body(acc, start_idx)
    return acc.get()    

  def check_equal_sizes(self, sizes):
    pass 
  
  def slice_value(self, start, stop, step):
    Slice(start, stop, step)

  def invoke(self, fn, args):
    assert False     
  
  none = None
  null_slice = slice(None, None, None)
  def identity_function(self, x):
    return x
  
shape_semantics = ShapeSemantics()

class ShapeInference(SyntaxVisitor):
  def __init__(self):
    self._clear()

  def _clear(self):
    self.value_env = {}
    self.equivalence_classes = {}
    
  
  def unify_scalar_var(self, x, y):
    assert isinstance(x, Var), "Expected scalar variable, but got: " + str(x)
    assert isinstance(y, Scalar), "Expected scalar, but got: " + str(y)
    if isinstance(y, UnknownScalar):
      return x
    equivs = self.equivalence_classes.get(x, set([]))
    equivs.add(y)
    for var in equivs:
      self.equivalence_classes[var] = equivs
    if isinstance(y, Const):
      for var in equivs:
        self.value_env[var] = y
      return y
    else:
      return var
    
  def unify_scalar_pairs(self, xs, ys):
    result_elts = []
    for xi, yi in zip(xs.elts, ys.elts):
      result_elts.append(self.unify_scalars(xi, yi))
    return result_elts 
  
  def unify_scalar_list(self, values):
    assert len(values) > 0
    acc = unknown_scalar 
    for v in values:
      acc = self.unify_scalars(acc, v)  
    return acc 
  
  def unify_scalars(self, x, y):
    if isinstance(x, Unknown):
      return y
    elif isinstance(y, Unknown):
      return x
    elif isinstance(x, Var):
      return self.unify_scalar_var(x, y)
    elif isinstance(y, Var):
      return self.unify_scalar_var(y, x)
    else:
      raise RuntimeError("Unsupported by unify: %s, %s" % (x,y))

  def visit_merge(self, merge):
    for (k, (l,r)) in merge.iteritems():
      self.value_env[k] = l.combine(r)
      
  def visit_Const(self, expr):

    return Const(expr.value)
  
  def visit_PrimCall(self, expr):
    return unknown_scalar 
  
  def visit_Var(self, expr):
    name = expr.name
    if name in self.value_env: 
      return self.value_env[name]
    elif name in self.equivalence_classes:
      for other_name in self.equivalence_classes[name]:
        if other_name in self.value_env:
          return self.value_env[other_name] 
    raise RuntimeError("Unknown variable: %s" %  expr)
  
  def visit_Tuple(self, expr):
    return Tuple(self.visit_expr_list(expr.elts))

  def visit_Array(self, expr):
    elts = self.visit_expr_list(expr.elts)
    elt = combine_list(elts)
    n = len(elts)
    res = increase_rank(elt, 0, const(n))
    return res 
  
  def visit_Map(self, expr):
    axis = expr.axis 
    arg_shapes = self.visit_expr_list(expr.args)
    return shape_semantics.eval_map(expr.fn, arg_shapes, axis)
  
  def visit_Reduce(self, expr):
    axis = expr.axis 
    fn = expr.fn 
    combine = expr.combine 
    arg_shapes = self.visit_expr_list(expr.args)
    init = self.visit_expr(self.init) if self.init else None 
    shape_semantics.eval_reduce(fn, combine, init, arg_shapes, axis)
  
  def bind(self, lhs, rhs):
    if isinstance(lhs, syntax.Tuple):
      assert isinstance(rhs, Tuple)
      for l,r in zip(lhs.elts, rhs.elts):
        self.bind(l,r)
    else:
      assert isinstance(lhs, syntax.Var), \
        "Unexpected LHS: " + str(lhs)
      self.value_env[lhs.name] = rhs 
      
  def visit_Assign(self, stmt):
    rhs = self.visit_expr(stmt.rhs)
    self.bind(stmt.lhs, rhs) 
    
  def visit_Return(self, stmt):
    new_value = self.visit_expr(stmt.value)
    old_value = self.value_env.get("$return", unknown_value)
    combined = old_value.combine(new_value)
    self.value_env["$return"] = combined 
    
  def visit_fn(self, fn):
    self._clear()
    arg_types = [fn.type_env[name] for name in fn.args]
    input_values = InputConverter().values_from_types(arg_types)
    for n,v in zip(fn.args, input_values):
      self.value_env[n] = v 
    self.visit_block(fn.body)
    return self.value_env["$return"] 
  
_symbolic_shape_cache = {}
def call_shape_expr(typed_fn):

  if isinstance(typed_fn, str):
    typed_fn = syntax.TypedFn.registry[typed_fn]
    
  if typed_fn.name in _symbolic_shape_cache:
    return _symbolic_shape_cache[typed_fn.name]
  else:
    shape_inference = ShapeInference()
    result_abstract_value = shape_inference.visit_fn(typed_fn)

    _symbolic_shape_cache[typed_fn.name] = result_abstract_value
    return result_abstract_value



def bind(lhs, rhs, env):
  if isinstance(lhs, Var):
    env[lhs.num] = rhs 
  elif isinstance(lhs, Shape):
    assert isinstance(rhs, Shape)
    bind_pairs(lhs.dims, rhs.dims, env)
  elif isinstance(lhs, Closure):
    assert isinstance(rhs, Closure)
    bind_pairs(lhs.args, rhs.args, env)
  elif isinstance(lhs, Tuple):
    assert isinstance(rhs, Tuple)
    bind_pairs(lhs.elts, rhs.elts)
  else:
    raise RuntimeError("Unexpected shape LHS: %s" % lhs)
    
def bind_pairs(xs, ys, env):
  assert len(xs) == len(ys), \
    "Can't bind %s and %s due to unequal lengths" % (xs, ys)
  for (x,y) in zip(xs,ys):
    bind(x,y,env)
  
def subst(x, env):
  if isinstance(x, Var):
    assert x in env, "Unknown variable %s" % x
    return env[x]
  elif isinstance(x, Scalar):
    return x 
  elif isinstance(x, Shape):
    return array(*subst_list(x.dims, env))
  elif isinstance(x, Tuple):
    return tuple(*subst_list(x.elts, env))
  elif isinstance(x, Closure):
    return Closure(x.fn, subst_list(x.args, env))
  else:
    raise RuntimeError("Unexpected abstract expression: %s" % x)
    
def subst_list(xs, env):
  return [subst(x, env) for x in xs]



def symbolic_call(typed_fn, abstract_inputs):
  # result in terms of variables like input0, (shape: input1, input2), etc..
  abstract_result_value = call_shape_expr(typed_fn)
  shape_formals = InputConverter().values_from_types(typed_fn.input_types)
  env = {}
  bind_pairs(shape_formals, abstract_inputs, env)
  
  return subst(abstract_result_value, env)
  
  
  
  


import types 
import numpy as np 
from common import dispatch




def eval_shape(symbolic_shape, inputs):
  """
  Evaluate symbolic shapes into concrete shapes
  """
  
  def transform_value(x):
    """
    Replace arrays with their shapes, 
    and recursively replace any instances of arrays
    in data structures like tuples also with their shapes
    """
    if isinstance(x, np.ndarray):
      return x.shape 
    elif isinstance(x, list):
      return np.array(x).shape 
    elif isinstance(x, tuple):
      return tuple(transform_value(elt) for elt in x)
    else:
      assert isinstance(x, (int, long, float, complex, types.NoneType)), \
        "Unexpected value " + str(x)
      return x
  input_shapes = [transform_value(x) for x in inputs]

  
  def eval_abstract_value(v):
    def eval_Input():
      return input_shapes[v.pos]
    
    def eval_Const():
      return v.value 
    
    def eval_Shape():
      return tuple(eval_abstract_values(v.dims))
    
    def eval_Dim():
      return eval_abstract_value(v.array)[v.dim]
    
    def eval_Tuple():
      return Tuple(eval_abstract_values(v.elts))
    
    def eval_Sub():
      x = eval_abstract_value(v.x)
      y = eval_abstract_value(v.y)
      return x - y
    
    def eval_Closure():
      return  Closure(v.untyped_fn, eval_abstract_values(v.args))
    
    return dispatch(v, "eval")
    
  def eval_abstract_values(xs):
    return [eval_abstract_value(x) for x in xs]
  return eval_abstract_value(symbolic_shape)


    
      
      
  