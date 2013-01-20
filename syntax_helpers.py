import args
import array_type
import syntax
import tuple_type

from core_types import Int64, Int32, Float32, Float64, Bool
from core_types import FloatT, IntT, BoolT, NoneType, ScalarT
from syntax import Const

def const_int(n, t = Int64):
  return Const(n, type = t)

def const_float(f, t = Float64):
  return Const(f, type = t)

def const_bool(b, t = Bool):
  return Const(b, type = t)

def zero(t):
  if isinstance(t, FloatT):
    x = 0.0
  elif isinstance(t, BoolT):
    x = False
  else:
    assert isinstance(t, IntT)
    x = 0
  return Const(x, type = t)

false = zero(Bool)
zero_i32 = zero(Int32)
zero_i64 = zero(Int64)
zero_f32 = zero(Float32)
zero_f64 = zero(Float64)

def one(t):
  if isinstance(t, FloatT):
    x = 1.0
  elif t.__class__ is BoolT:
    x = True
  else:
    assert isinstance(t, IntT)
    x = 1
  return Const(x, type = t)

true = one(Bool)
one_i32 = one(Int32)
one_i64 = one(Int64)
one_f32 = one(Float32)
one_f64 = one(Float64)

none_t = NoneType
none = Const(None, type = none_t)

slice_none_t = array_type.make_slice_type(none_t, none_t, none_t)
slice_none = syntax.Slice(none, none, none, type = slice_none_t)

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
    return x is None or is_python_scalar(x)

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
    return none

def unwrap_constant(x):
  if isinstance(x, syntax.Expr):
    assert x.__class__ is Const, \
        "Expected constant, got %s" % (x,)
    return x.value
  else:
    assert is_python_constant(x)
    return x

def wrap_if_constant(x):
  """
  If given value isn't already an expression turn it into one with the const
  helper
  """

  if is_python_constant(x):
    return const(x)
  else:
    assert isinstance(x, syntax.Expr), "Expected expression, got " + str(x)
    return x

def wrap_constants(xs):
  return map(wrap_if_constant, xs)

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
  if isinstance(exprs, args.ActualArgs):
    return exprs.transform(get_type)
  else:
    return [expr.type for expr in exprs]

def is_zero(expr):
  return expr.__class__ is Const and expr.value == 0

def is_one(expr):
  return expr.__class__ is Const and expr.value == 1

def is_false(expr):
  return expr.__class__ is Const and expr.value == False

def is_true(expr):
  return expr.__class__ is Const and expr.value == True

def is_none(expr):
  return expr.__class__ is Const and expr.value == None

def is_constant(expr):
  return expr.__class__ is Const

def all_constants(exprs):
  return all(map(is_constant, exprs))

def collect_constant(expr):
  return expr.value

def collect_constants(exprs):
  return map(collect_constant, exprs)

def is_scalar(expr):
  return isinstance(expr.type, ScalarT)

def all_scalars(exprs):
  return all(map(is_scalar, exprs))

def is_identity_fn(fn):
  return len(fn.arg_names) == 1 and len(fn.body) == 1 and \
         fn.body[0].__class__ is syntax.Return and \
         fn.body[0].value.__class__ is syntax.Var and \
         fn.body[0].value.name == fn.arg_names[0]
