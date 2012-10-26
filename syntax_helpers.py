import syntax 
import core_types 
import tuple_type 

const_none = syntax.Const(None, type = core_types.NoneType)

def const_int(n, t = core_types.Int64):
  return syntax.Const(n, type = t)

def const_float(f, t = core_types.Float64):
  return syntax.Const(f, type = t)

def const_bool(b, t = core_types.Bool):
  return syntax.Const(b, type = t)

def is_python_int(x):
  return isinstance(x, (int, long))

def is_python_float(x):
  return isinstance(x, float)

def is_python_bool(x):
  return isinstance(x, bool)

def is_python_scalar(x):
  return isinstance(x,  (bool, int, long, float))

def const_scalar(x):
  if is_python_int(x):
    return const_int(x)
  elif isinstance(x, bool):
    return const_bool(x)
  else:
    assert isinstance(x, float)
    return const_float(x)

def make_tuple(elts):
  elt_types = get_types(elts)
  tuple_t = tuple_type.make_tuple_type(elt_types)
  return syntax.Tuple(elts, type = tuple_t)

def const_tuple(*python_values):
  return make_tuple(map(const, python_values))

def const(x):
  if is_python_scalar(x):
    return const_scalar(x)
  elif isinstance(x, tuple):
    return const_tuple(*map(const, x))
  else:
    assert x is None, \
      "Can't convert Python value %s into a Parakeet constant" % x
    return const_none 
    

def wrap_constant(x):
  """
  If given value isn't already an expression
  turn it into one with the const helper
  """
  if isinstance(x, syntax.Expr):
    return x
  else:
    return const(x)
  
def wrap_constants(xs):
  return map(wrap_constant, xs)
  
def get_type(expr):
  return expr.type

def get_types(exprs):
  return [expr.type for expr in exprs]


def is_zero(expr):
  return isinstance(expr, syntax.Const) and expr.value == 0

def is_one(expr):
  return isinstance(expr, syntax.Const) and expr.value == 1

def is_constant(expr):
  return isinstance(expr, syntax.Const)

def all_constants(exprs):
  return all(map(is_constant, exprs))

def collect_constant(expr):
  return expr.value

def collect_constants(exprs):
  return map(collect_constant, exprs)
