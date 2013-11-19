import numpy as np 

from .. ndtypes import ( Int8, Int24, Int32, Int64,  Float32, Float64, Bool, FloatT, IntT, BoolT, 
                        NoneType, ScalarT, make_slice_type, make_tuple_type, ClosureT, 
                        FnT, Type, make_closure_type, ArrayT)

from array_expr import Slice
from expr  import Const, Var, Expr, Closure, ClosureElt 
from tuple_expr import Tuple 
from stmt import Return  
from untyped_fn import UntypedFn
from typed_fn import TypedFn 

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
zero_i8 = zero(Int8)
zero_i24 = zero(Int24)
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
one_i8 = zero(Int8)
one_i32 = one(Int32)
one_i64 = one(Int64)
one_f32 = one(Float32)
one_f64 = one(Float64)


none_t = NoneType
none = Const(None, type = none_t)

slice_none_t = make_slice_type(none_t, none_t, none_t)
slice_none = Slice(none, none, none, type = slice_none_t)

def is_python_int(x):
  return isinstance(x, (int, 
                        long, 
                        np.int8, 
                        np.int16, 
                        np.int32, 
                        np.int64, 
                        np.uint8, 
                        np.uint32,
                        np.uint64))

def is_python_float(x):
  return isinstance(x, (float, np.float32, np.float64 ))

def is_python_bool(x):
  return isinstance(x, (bool, np.bool, np.bool8, np.bool_))

def is_python_scalar(x):
  return isinstance(x,  (bool, int, long, float)) or isinstance(x, np.ScalarType)

def is_python_constant(x):
  if isinstance(x, tuple):
    return all(map(is_python_constant, x))
  else:
    return x is None or is_python_scalar(x)

def const_scalar(x):
  if is_python_bool(x):
    return const_bool(x)
  elif is_python_int(x):
    return const_int(x)
  else:
    assert is_python_float(x), "Unexpected value %s" % x 
    return const_float(x)

def make_tuple(elts):
  elt_types = get_types(elts)
  tuple_t = make_tuple_type(elt_types)
  return Tuple(elts, type = tuple_t)

def const_tuple(*python_values):
  return make_tuple(map(const, python_values))

def const(x):
  
  if is_python_scalar(x):
    return const_scalar(x)
  elif isinstance(x, tuple):
    return const_tuple(*map(const, x))
  elif isinstance(x, Expr):
    return x
  else:
    assert x is None, \
        "Can't convert Python value %s into a Parakeet constant" % x
    return none

def unwrap_constant(x):
  if isinstance(x, Expr):
    if x.__class__ is Tuple:
      return tuple(unwrap_constant(elt) for elt in x.elts)
    elif x.type is NoneType:
      return None
    elif x.__class__ is UntypedFn:
      return x 
    assert x.__class__ is Const, \
        "Expected constant, got %s : %s" % (x, x.type)
    return x.value
  
  else:
    assert is_python_constant(x)
    return x
  
def unwrap_constants(xs):
  return [unwrap_constant(x) for x in xs]

def wrap_if_constant(x):
  """
  If given value isn't already an expression turnhelpers it into one with the const
  helper
  """

  if is_python_constant(x):
    return const(x)
  else:
    assert isinstance(x, Expr), "Expected expression, got " + str(x)
    return x

def wrap_constants(xs):
  return map(wrap_if_constant, xs)

def wrap_var(x):
  if isinstance(x, str):
    return Var(x)
  else:
    assert isinstance(x, Var), "Expected a variable, got %s" % x

def wrap_vars(xs):
  return map(wrap_var, xs)

def get_type(expr):
  return expr.type

def get_types(exprs):
  if hasattr(exprs, 'transform'):
    return exprs.transform(get_type)
  else:
    return [expr.type for expr in exprs]
  
def get_elt_type(expr):
  t = expr.type 
  if isinstance(t, ScalarT):
    return t 
  else:
    assert isinstance(t, ArrayT), "Expected array or scalar, not %s : %s" % (expr, t)
    return t.elt_type 

def get_elt_types(ts):
  return [get_elt_type(t) for t in ts]

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
         fn.body[0].__class__ is Return and \
         fn.body[0].value.__class__ is Var and \
         fn.body[0].value.name == fn.arg_names[0]

def get_fn(maybe_closure):
  if isinstance(maybe_closure, TypedFn):
    return maybe_closure 
  elif isinstance(maybe_closure, (FnT, ClosureT, Closure)):
    return maybe_closure.fn 
  elif isinstance(maybe_closure.type, (FnT, ClosureT)):
    return maybe_closure.type.fn 
  else:
    assert False, "Can't get function from %s" % maybe_closure 

def return_type(maybe_closure):
  return get_fn(maybe_closure).return_type

def get_closure_args(maybe_closure):
  if isinstance(maybe_closure, Type):
    assert isinstance(maybe_closure, (FnT, ClosureT))
    maybe_closure = maybe_closure.fn 
  if maybe_closure.__class__ is Closure:
    return tuple(maybe_closure.args)
  elif maybe_closure.type.__class__ is ClosureT:
    return tuple(ClosureElt(maybe_closure, i, type = arg_type)
                 for i, arg_type in enumerate(maybe_closure.type.arg_types))
  else:
    return ()
  
def make_closure(fn, closure_args):
  old_closure_args = get_closure_args(fn)
  fn = get_fn(fn)
  closure_args = tuple(closure_args)
  combined_closure_args = old_closure_args + closure_args 
  t = make_closure_type(fn, combined_closure_args)
  return Closure(fn, combined_closure_args, type = t)
  
def gen_arg_names(n, base_names):
  results = []

  m = len(base_names)
  for i in xrange(n):
    curr = base_names[i % m]
    cycle = i / m
    if cycle > 0:
      curr = "%s_%d" % (curr, cycle+1)
    results.append(curr)
  return results

def gen_data_arg_names(n):
  return gen_arg_names(n, ['x', 'y', 'z', 'a', 'b', 'c', 'd'])

def gen_fn_arg_names(n):
  return gen_arg_names(n, ['f', 'g', 'h', 'p', 'q', 'r', 's'])

