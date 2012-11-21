import syntax
from syntax_visitor import SyntaxVisitor
from syntax_helpers import get_types
import type_inference  
from node import Node  




class ValueMismatch(Exception):
  """
  Raise this exception whenever two incompatible
  abstract values get combined
  """
  def __init__(self, v1, v2):
    self.v1 = v1
    self.v2 = v2 

  def __str__(self):
    return "ValueMismatch(%s, %s)" % (self.v1, self.v2)

  def __repr__(self):
    return str(self)
  
  
class AbstractValue(Node):
  pass 

class Unknown(Node):
  def __eq__(self, other):
    return isinstance(other, Unknown)
  
  def combine(self, other):
    return other 
  
  def __str__(self):
    return "<unknown>"
  
  def __repr__(self):
    return str(self)

unknown_value = Unknown()


class Scalar(AbstractValue):
  """
  Base class for all scalar operations
  """
  rank = 0
  
class UnknownScalar(Scalar):
  def __eq__(self, other):
    return isinstance(other, UnknownScalar)
  
  def combine(self, other):
    assert isinstance(other, Scalar), \
      "Can't combine scalar with %s" % other 
    return self 
  
  def __str__(self):
    return "Scalar"
  
unknown_scalar = UnknownScalar()

class Const(Scalar):
  def __init__(self, value):
    self.value = value 
    
  def __eq__(self, other):
    return isinstance(other, Const) and other.value == self.value
    
  def __str__(self):
    return "Const(%d)" % self.value 

  def combine(self, other):
    if self == other:
      return self
    elif isinstance(other, Scalar):
      return unknown_scalar
    else:
      raise ValueMismatch(self, other)

def const(x):
  return Const(x)

def is_zero(d):
  return isinstance(d, Const) and d.value == 0

def is_one(d):
  return isinstance(d, Const) and d.value == 1

class Sub(Scalar):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __eq__(self, other):
    return isinstance(other, Sub) and \
      self.x == other.x and \
      self.y == other.y
      
  def __str__(self):
    return "%s - %s" % (self.x, self.y)
  
  def __repr__(self):
    return str(self) 
  
  def __hash__(self):
    return hash( (self.x, self.y) )
  
class Var(Scalar):
  def __init__(self, num):
    self.num = num 
    
  def __eq__(self, other):
    return isinstance(other, Var) and self.num == other.num
  
  def __hash__(self):
    return hash(self.num)
  
  def __str__(self):
    return "x%d" % self.num 
  
  def __repr__(self):
    return str(self)
  
  def combine(self, other):
    if self == other:
      return self 
    else:
      # combining two different variables returns an unknown scalar 
      return unknown_scalar 


class Shape(AbstractValue):
  def __init__(self, dims):
    assert len(dims) > 0
    self.dims = [const(d) if isinstance(d, int) else d for d in dims] 
    self.rank = len(dims)
      
  def __eq__(self, other):
    return isinstance(other, Shape) and \
      len(self.dims) == len(other.dims) and \
      all(d1 == d2 for (d1,d2) in zip(self.dims, other.dims) )
  
  def __str__(self):
    return "Shape(%s)" % (", ".join(str(d) for d in self.dims))
  
  def __repr__(self):
    return str(self)
  
  def combine(self, other):
    if isinstance(other, Shape) and other.rank == self.rank:
      dims = combine_pairs(self.dims, other.dims)
      return array(*dims)
    raise ValueMismatch(self, other)

def array(*dims):
  return Shape(dims) 


def dim(shape, d):
  if isinstance(shape, Shape):
    return shape.dims[d]
  else:
    # if the shape isn't available, getting the d'th 
    # dimension returns an unknown scalar 
    return unknown_scalar 
  
def dim_list(shapes, d, exclude_scalars=False):
  if exclude_scalars:
    shapes = [s for s in shapes if not is_scalar(s)]
  return [dim(s,d) for s in shapes]


def array_of_unknown_shape(rank):
  return Shape([unknown_scalar] * rank)

def lower_rank(x, axis):
  assert isinstance(x, Shape), \
    "Can't decrease rank of %s" % x
  # by convention, lowering a scalar returns a scalar 
  if  axis >= x.rank or x.rank == 1:
    return unknown_scalar

  new_dims = []
  for (i,d) in enumerate(x.dims):
    if i != axis:
      new_dims.append(d)
  return array(*new_dims)

def lower_ranks(xs, axis):
  return [lower_rank(x, axis) for x in xs]
    
def increase_rank(x, axis, dim_expr):
  if isinstance(dim_expr, int):
    dim_expr = const(dim_expr)
    
  if isinstance(x, Shape):
    # we're taking dims d1...dm and constructing a 
    # new shape d1...d_new...dm by inserting d_new
    # in the slot of the given axis 
    assert axis <= x.rank 
    if axis < len(x.dims):
      new_dims = []
      for (i,d) in enumerate(x.dims):
        if i == axis:
          new_dims.append(dim_expr)
        new_dims.append(d)
    else:
      new_dims = [d for d in x.dims]
      new_dims.append(dim_expr)
    return array(*new_dims)
  elif is_scalar(x):
    return Shape([dim_expr])        
  else:
    raise RuntimeError("Don't know how to raise rank of value %s" % x)
   

def is_scalar(v):
  return isinstance(v, Scalar) 

class Slice(AbstractValue):
  def __init__(self, start, stop, step):
    self.start = start
    self.stop = stop 
    self.step = step 
    
  def __eq__(self, other):
    return isinstance(other, Slice) and \
      self.start == other.start and \
      self.stop == other.stop and \
      self.step == other.step 
  
  def combine(self, other):
    if isinstance(other, Slice):
      start = self.start.combine(other.start)
      stop = self.stop.combine(other.stop)
      step = self.step.combine(other.step)
      return Slice(start, stop, step)
    else:
      raise ValueMismatch(self, other)


class Tuple(AbstractValue):
  def __init__(self, elts):
    self.elts = elts 
    
  def __eq__(self, other):
    return isinstance(other, Tuple) and \
      len(self.elts) == len(other.elts) and \
      all(e1 == e2 for (e1, e2) in zip(self.elts, other.elts))
  
  def __str__(self):
    return "Tuple(%s)" % ", ".join(str(e) for e in self.elts)
  
  def combine(self, other):
    if isinstance(other, Tuple):
      if len(self.elts) == len(other.elts):
        return Tuple(combine_pairs(self.elts, other.elts))
    raise ValueMismatch(self, other)

class Closure(AbstractValue):
  def __init__(self, untyped_fn, args):
    self.untyped_fn = untyped_fn 
    self.args = args 
  
  def __str__(self):
    return "Closure(fn = %s, %s)" % \
      (self.untyped_fn, ", ".join(str(e) for e in self.elts))
  
  def __eq__(self, other):
    return isinstance(other, Closure) and \
      self.untyped_fn == other.untyped_fn and \
      len(self.arg_shapes) == len(other.arg_shapes) and \
      all(v1 == v2 for (v1,v2) in zip(self.args, other.args))
  
  def combine(self, other):
    if isinstance(other, Closure):
      # TODO: Implement sets of closures like we have in the type system 
      if self.untyped_fn == other.untyped_fn and \
         len(self.args) == len(other.args) :
        combined_args = combine_pairs(self.args, other.args)
        return Closure(self.untyped_fn, combined_args)
    raise ValueMismatch(self, other)  


def combine_list(xs):
  acc = unknown_value
  for x in xs:
    acc = acc.combine(x)
  return acc 

def combine_pairs(xs, ys):
  return [xi.combine(yi) for (xi, yi) in zip(xs, ys)]

import core_types 
import array_type 
import tuple_type 
import closure_type 

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
    """
    elif isinstance(x, Shape):
      assert isinstance(y, Shape), \
        "Expected array, got: %s" % y
      x_rank = len(x.dims)
      y_rank = len(y.dims)
      assert x_rank == y_rank, \
        "Can't unify arrays of rank %d and %d" % \
        (x_rank, y_rank)
      return array(*self.unify_pairs(x.dims, y.dims))
    elif isinstance(y, Shape):
      return self.unify_arrays(y,x)
    elif isinstance(x, Closure):
      assert isinstance(y, Closure), "Expected closure, got " + str(y)
      assert (x.untyped_fn == y.untyped_fn) and len(x.args) == len(y.args), \
        "Can't yet support joining different closures"
      return Closure(self.unify_pairs(x.args, y.args))
    elif isinstance(x, Tuple):
      assert isinstance(y, Tuple), "Expected tuple, got " + str(y)
      assert len(x.elts) == len(y.elts)
      return Tuple(self.unify_pairs(x.elts, y.elts))
    elif isinstance(x, Slice):
      assert isinstance(y, Slice), "Expected slice, got " + str(y)
      start = self.unify(x.start, y.start)
      stop = self.unify(x.stop, y.stop)
      step = self.unify(x.step, y.step)
      return Slice(start, stop, step)
    else:
      return x.combine(y)
    """
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
    assert expr.axis is not None 
    axis = expr.axis 
    arg_shapes = self.visit_expr_list(expr.args)
    # collect the dim size of all array arguments along the adverb axis
    niters_list = dim_list(arg_shapes, axis, exclude_scalars=True)
    outer_dim = self.unify_scalar_list(niters_list)
    elt_shapes = lower_ranks(arg_shapes, axis)
   
    # result in terms of input scalar variables
    closure_t = expr.fn.type
    arg_types = get_types(expr.args)
    elt_types = array_type.lower_ranks(arg_types, 1)
    
    typed_fn = type_inference.get_invoke_specialization(closure_t, elt_types)

   
    elt_result_shape = symbolic_call(typed_fn, elt_shapes)
    return increase_rank(elt_result_shape, axis, outer_dim) 
  
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
  print typed_fn 
  if isinstance(typed_fn, str):
    import function_registry
    typed_fn = function_registry.typed_functions[typed_fn]
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


    
      
      
  