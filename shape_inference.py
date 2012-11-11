import syntax
from syntax_visitor import SyntaxVisitor 
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


def dim(a, d):
  if isinstance(a, Shape):
    return a.dims[d]
  else:
    # if the shape isn't available, getting the d'th 
    # dimension returns an unknown scalar 
    return unknown_scalar 


def array_of_unknown_shape(rank):
  return Shape([unknown_scalar] * rank)

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
""""
class Elt(AbstractValue):
  #Either the elt of a tuple, or the 
  #start/stop/step of a slice
  
  def __init__(self, value, pos):
    self.value = value 
    self.pos = pos 
    
  
  def __eq__(self, other):
    return isinstance(other, Elt) and \
      self.value == other.value  and \
      self.pos == other.pos  

  def __hash__(self):
    return hash((self.value, self.pos))
  
  def __str__(self):
    return "elt(%s, %d)" % (self.value, self.pos)
  
  def __repr__(self):
    return str(self)
  
  def combine(self, other):
    raise RuntimeError("What do you mean?")
"""

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
def value_from_type(t, counter):
  if isinstance(t, core_types.ScalarT):
    return Var(counter), counter + 1
  elif isinstance(t, array_type.ArrayT):
    new_counter = counter + t.rank 
    dim_vars = [Var(c) for c in xrange(counter, new_counter)]
    return Shape(dim_vars), new_counter 
  elif isinstance(t, tuple_type.TupleT):
    elt_values, counter  = values_from_types(t.elt_types, counter, True)
    return Tuple(elt_values), counter 
  elif isinstance(t, array_type.SliceT):
    start, counter = value_from_type(t.start_type, counter)
    stop, counter = value_from_type(t.stop_type, counter)
    step, counter = value_from_type(t.step_type, counter)
    return Slice(start, stop, step), counter 
  else:
    assert False, "Unsupported type: %s" % t
    
def values_from_types(arg_types, counter = 0, return_counter = False):
  """
  Turn a list of input types into a list of abstract values, 
  numbering the input arrays and scalars but preserving the 
  structure of tuples, closures, and slices
  """
  values = []
  for t in arg_types:
    v, counter = value_from_type(t, counter)
    values.append(v)
  if return_counter:
    return values, counter
  else:
    return values   
  

class ShapeInference(SyntaxVisitor):
  
  def __init__(self):
    self._clear()

  def _clear(self):
    self.value_env = {}
    self.equivalence_classes = {}
    
  
  def unify_scalar_var(self, var, y):
    """
    The first argument is already known to a be DimSize node, 
    add the other arg to the equivalence class and, 
    if it's a ground value (a constant scalar), overwrite 
    everyone's value with the constant. 
    """
    assert is_scalar(y), "Expected dim size to be scalar: " + str(y)
    equivs = self.equivalence_classes.get(var, set([]))
    equivs.add(y)
    for var in equivs:
      self.equivalence_classes[var] = equivs
    if isinstance(y, Const):
      for var in equivs:
        self.value_env[var] = y
      return y
    else:
      return var
    
  def unify_pairs(self, xs, ys):
    result_elts = []
    for xi, yi in zip(xs.elts, ys.elts):
      result_elts.append(self.unify(xi, yi))
    return result_elts 
  
  def unify_list(self, values):
    acc = values[0]
    for v in values[1:]:
      acc = self.unify(acc, v)  
    return acc 
  
  def unify(self, x, y):
    if isinstance(x, Unknown):
      return y
    elif isinstance(y, Unknown):
      return x
    elif isinstance(x, Var):
      return self.unify_scalar_var(x, y)
    elif isinstance(y, Var):
      return self.unify_scalar_var(y, x)
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
  
  def visit_merge(self, merge):
    for (k, (l,r)) in merge.iteritems():
      self.value_env[k] = l.combine(r)
      
  def visit_Const(self, expr):

    return Const(expr.value)
  
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
    print "elts", elts
    print "elt", elt
    print "n", n
    res = increase_rank(elt, 0, const(n))
    print "res", res 
    return res 
    
  def visit_Map(self, expr):
    arg_shapes = self.visit_expr_list(expr.args)
  
  def bind(self, lhs, rhs):
    if isinstance(lhs, syntax.Tuple):
      assert isinstance(rhs, Tuple)
      for l,r in zip(lhs.elts, rhs.elts):
        self.bind(l,r)
    else:
      assert isinstance(lhs, syntax.Var), \
        "Unexpected LHS: " + str(lhs)
      name = lhs.name
      if name in self.value_env:
        rhs = self.unify(self.value_env[name], rhs)
      self.env[name] = rhs 
      
  def visit_Assign(self, stmt):
    rhs = self.visit_expr(stmt.rhs)
    self.bind(self.lhs, rhs) 
    
  def visit_Return(self, stmt):
    
    new_value = self.visit_expr(stmt.value)
    print "new_value", new_value
    old_value = self.value_env.get("$return", unknown_value)
    print "old_value", old_value 
    combined = old_value.combine(new_value)
    print "combined", combined 
    self.value_env["$return"] = combined 
    
  def visit_fn(self, fn):
    self._clear()
    arg_names = fn.args.arg_slots
    arg_types = [fn.type_env[name] for name in arg_names]
    input_values = values_from_types(arg_types)
    for n,v in zip(arg_names, input_values):
      self.value_env[n] = v 
    self.visit_block(fn.body)

    return self.value_env["$return"] 
  

def symbolic_call_shape(typed_fn):
  shape_inference = ShapeInference()
  return shape_inference.visit_fn(typed_fn)

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


    
      
      
  