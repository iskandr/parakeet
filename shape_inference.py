import syntax
from syntax_visitor import SyntaxVisitor 
from node import Node  

   
"""
class Max(ShapeElt):
  def __init__(self, d1, d2):
    self.d1 = d1
    self.d2 = d2
    
  def __eq__(self, other):
    return \
      (self.d1 == other.d1 and self.d2 == other.d2) or \
      (self.d2 == other.d1 and self.d1 == other.d1)  

def max_dim(d1, d2):
  if d1 == d2:
    return d1
  elif is_zero(d1):
    return d2 
  elif is_zero(d2):
    return d1 
  elif isinstance(d1, Const) and isinstance(d2, Const):
    if d1.value < d2.value:
      return d2 
    else:
      return d1 
  else:
    return Max(d1, d2)
"""



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

class UnknownValue(Node):
  def __eq__(self, other):
    return isinstance(other, UnknownValue)
  
  def combine(self, other):
    return other 
  
  def __str__(self):
    return "<unknown>"
  
  def __repr__(self):
    return str(self)

unknown_value = UnknownValue()

class ConstValue(AbstractValue):
  def __init__(self, value):
    self.value = value 
    
  def __eq__(self, other):
    return isinstance(other, ConstValue) and other.value == self.value
    
  def __str__(self):
    return "Const(%d)" % self.value 

  def combine(self, other):
    if self == other:
      return self
    elif is_scalar(other):
      return scalar
    else:
      raise ValueMismatch(self, other)

def const(x):
  return ConstValue(x)

def is_zero(d):
  return isinstance(d, ConstValue) and d.value == 0

def is_one(d):
  return isinstance(d, ConstValue) and d.value == 1

      
class ScalarVar(AbstractValue):
  def __init__(self, name):
    self.name = name 
  
  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other):
    return self.name == other.name 

  def combine(self, other):
    if self == other:
      return self
    else:
      return scalar 
  
class DimSize(AbstractValue):
  """
  The value producted by x.shape[d]
  """
  def __init__(self, array_name, dim):
    self.array_name = array_name
    self.dim = dim 
    
  def __eq__(self, other):
    return isinstance(other, DimSize) and \
      self.array_name == other.array_name and \
      self.dim == other.dim
  
  def __str__(self):
    return "DimSize(%s, %d)" % (self.array_name, self.dim)
  
  def __repr__(self):
    return str(self)
  
  def __hash__(self):
    return hash("%s!%d" % (self.var, self.dim))
  
  def combine(self, other):
    if self == other:
      return self 
    else:
      raise ValueMismatch(self, other)

class ArrayValue(AbstractValue):
  def __init__(self, dims):
    self.dims = [const(d) if isinstance(d, int) else d for d in dims] 
    self.rank = len(dims)
      
  def __eq__(self, other):
    return isinstance(other, ArrayValue) and \
      len(self.dims) == len(other.dims) and \
      all(d1 == d2 for (d1,d2) in zip(self.dims, other.dims) )
  
  def __str__(self):
    if self.rank == 0:
      return "Scalar"
    else:
      return "Array(%s)" % (", ".join(str(d) for d in self.dims))
  
  def __repr__(self):
    return str(self)
  
  def combine(self, other):
    if is_scalar(other):
      return self
    elif is_scalar(self):
      return other
    elif isinstance(other, ArrayValue) and other.rank == self.rank:
      dims = combine_pairs(self.dims, other.dims)
      return array(*dims)
    raise ValueMismatch(self, other)

def array(*dims):
  return ArrayValue(dims) 

# represent scalars with 0-rank array s
scalar = array() 

def increase_rank(x, axis, dim_expr):
  if isinstance(dim_expr, int):
    dim_expr = const(dim_expr)
    
  if isinstance(x, ArrayValue):
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
    return ArrayValue([dim_expr])        
  else:
    raise RuntimeError("Don't know how to raise rank of value %s" % x)
    
def is_scalar(v):
  return isinstance(v, (ConstValue, ScalarVar, DimSize)) or \
    (isinstance(v, ArrayValue) and v.rank == 0)

class SliceValue(AbstractValue):
  def __init__(self, start, stop, step):
    self.start = start
    self.stop = stop 
    self.step = step 
    
  def __eq__(self, other):
    return isinstance(other, SliceValue) and \
      self.start == other.start and \
      self.stop == other.stop and \
      self.step == other.step 
  
  def combine(self, other):
    if isinstance(other, SliceValue):
      start = self.start.combine(other.start)
      stop = self.stop.combine(other.stop)
      step = self.step.combine(other.step)
      return SliceValue(start, stop, step)
    else:
      raise ValueMismatch(self, other)




class TupleValue(AbstractValue):
  def __init__(self, elts):
    self.elts = elts 
    
  def __eq__(self, other):
    return isinstance(other, TupleValue) and \
      len(self.elts) == len(other.elts) and \
      all(e1 == e2 for (e1, e2) in zip(self.elts, other.elts))
  
  def __str__(self):
    return "Tuple(%s)" % ", ".join(str(e) for e in self.elts)
  
  def combine(self, other):
    if isinstance(other, TupleValue):
      if len(self.elts) == len(other.elts):
        return TupleValue(combine_pairs(self.elts, other.elts))
    raise ValueMismatch(self, other)
  
class ClosureValue(AbstractValue):
  def __init__(self, untyped_fn, args):
    self.untyped_fn = untyped_fn 
    self.args = args 
  
  def __str__(self):
    return "Closure(fn = %s, %s)" % \
      (self.untyped_fn, ", ".join(str(e) for e in self.elts))
  
  def __eq__(self, other):
    return isinstance(other, ClosureValue) and \
      self.untyped_fn == other.untyped_fn and \
      len(self.arg_shapes) == len(other.arg_shapes) and \
      all(v1 == v2 for (v1,v2) in zip(self.args, other.args))
  
  def combine(self, other):
    if isinstance(other, ClosureValue):
      # TODO: Implement sets of closures like we have in the type system 
      if self.untyped_fn == other.untyped_fn and \
         len(self.args) == len(other.args) :
        combined_args = combine_pairs(self.args, other.args)
        return ClosureValue(self.untyped_fn, combined_args)
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

def input_shape(arg_name, arg_type):
  if isinstance(arg_type, array_type.ArrayT):
    rank = arg_type.rank
    dims = [DimSize(arg_name, i) for i in xrange(rank)]
    return array(*dims)
  elif isinstance(arg_type, array_type.ScalarT):
    return ScalarVar(arg_name) 
  else:
    raise RuntimeError("Input type not handled: %s" % arg_type)
  
class ShapeInference(SyntaxVisitor):
  
  def __init__(self):
    self._clear()

  def _clear(self):
    self.value_env = {}
    self.equivalence_classes = {}
    
  
  def unify_scalar_var(self, unknown, y):
    """
    The first argument is already known to a be DimSize node, 
    add the other arg to the equivalence class and, 
    if it's a ground value (a constant scalar), overwrite 
    everyone's value with the constant. 
    """
    assert is_scalar(y), "Expected dim size to be scalar: " + str(y)
    equivs = self.equivalence_classes.get(unknown, set([]))
    equivs.add(y)
    for var in equivs:
      self.equivalence_classes[var] = equivs
    if isinstance(y, ConstValue):
      for var in equivs:
        self.value_env[var] = y
      return y
    else:
      return unknown
    
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
    if isinstance(x, UnknownValue):
      return y
    elif isinstance(y, UnknownValue):
      return x
    elif isinstance(x, (DimSize, ScalarVar)):
      return self.unify_scalar_var(x, y)
    elif isinstance(y, (DimSize, ScalarVar)):
      return self.unify_scalar_var(y, x)
    elif isinstance(x, ArrayValue):
      assert isinstance(y, ArrayValue), \
        "Expected array, got: %s" % y
      x_rank = len(x.dims)
      y_rank = len(y.dims)
      assert x_rank == y_rank, \
        "Can't unify arrays of rank %d and %d" % \
        (x_rank, y_rank)
      return array(*self.unify_pairs(x.dims, y.dims))
    elif isinstance(y, ArrayValue):
      return self.unify_arrays(y,x)
    
    elif isinstance(x, ClosureValue):
      assert isinstance(y, ClosureValue), "Expected closure, got " + str(y)
      assert (x.untyped_fn == y.untyped_fn) and len(x.args) == len(y.args), \
        "Can't yet support joining different closures"
      return ClosureValue(self.unify_pairs(x.args, y.args))
    elif isinstance(x, TupleValue):
      assert isinstance(y, TupleValue), "Expected tuple, got " + str(y)
      assert len(x.elts) == len(y.elts)
      return TupleValue(self.unify_pairs(x.elts, y.elts))
    elif isinstance(x, SliceValue):
      assert isinstance(y, SliceValue), "Expected slice, got " + str(y)
      start = self.unify(x.start, y.start)
      stop = self.unify(x.stop, y.stop)
      step = self.unify(x.step, y.step)
      return SliceValue(start, stop, step)
    else:
      return x.combine(y)
  
  def visit_merge(self, merge):
    for (k, (l,r)) in merge.iteritems():
      self.value_env[k] = l.combine(r)
      
  def visit_Const(self, expr):

    return ConstValue(expr.value)
  
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
    return TupleValue(self.visit_expr_list(expr.elts))

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
      assert isinstance(rhs, TupleValue)
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
    for arg_name in fn.args.arg_slots:
      arg_type = fn.type_env[arg_name]
      s = input_shape(arg_name, arg_type)
      print "input", arg_name, arg_type, s 
      self.value_env[arg_name] = s 
    self.visit_block(fn.body)

    return self.value_env["$return"] 
  

    
def symbolic_call_shape(typed_fn):
  shape_inference = ShapeInference()
  return shape_inference.visit_fn(typed_fn)
  
      
  