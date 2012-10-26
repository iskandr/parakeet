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

def zero(t):
  if isinstance(t, core_types.FloatT):
    x = 0.0
  elif isinstance(t, core_types.BoolT):
    x = False
  else:
    assert isinstance(t, core_types.IntT)
    x = 0
  return syntax.Const(x, type = t)

false = zero(core_types.Bool)
zero_i32 = zero(core_types.Int32)
zero_i64 = zero(core_types.Int64)
zero_f32 = zero(core_types.Float32)
zero_f64 = zero(core_types.Float64)

def one(t):
  if isinstance(t, core_types.FloatT):
    x = 1.0
  elif isinstance(t, core_types.BoolT):
    x = True
  else:
    assert isinstance(t, core_types.IntT)
    x = 1
  return syntax.Const(x, type = t)

true = one(core_types.Bool)
one_i32 = one(core_types.Int32)
one_i64 = one(core_types.Int64)
one_f32 = one(core_types.Float32)
one_f64 = one(core_types.Float64)

def is_python_int(x):
  return isinstance(x, (int, long))

def is_python_float(x):
  return isinstance(x, float)

def is_python_bool(x):
  return isinstance(x, bool)

def is_python_scalar(x):
  return isinstance(x,  (bool, int, long, float))

def is_python_constant(x):
  if isinstance(x, tuple):
    return all(map(is_python_constant, x))
  else:
    return is_python_scalar(x)

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

def unwrap_constant(x):
  if isinstance(x, syntax.Expr):
    assert isinstance(x, syntax.Const)
    return x.value
  else:
    assert is_python_constant(x)


def wrap_constant(x):
  """
  If given value isn't already an expression
  turn it into one with the const helper
  """
  if is_python_constant(x):
    return const(x)
  else:
    assert isinstance(x, syntax.Expr), "Expected expression, got %s" % x 
    return x
  
def wrap_constants(xs):
  return map(wrap_constant, xs)

def wrap_var(x):
  if isinstance(x, str):
    return syntax.Var(x)
  else:
    assert isinstance(x, syntax.Var), "Expected a variable, got %s" % x

def wrap_vars(xs):
  return map(wrap_var, xs)
  
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
