import __builtin__


from .. frontend.decorators import jit, macro
from .. ndtypes import TypeValueT, ScalarT, make_array_type
from .. syntax import one_i64, zero_i64
from .. syntax import Range, Return, Cast, UntypedFn, TypedFn, AllocArray
from .. syntax import Tuple, DelayUntilTyped

from adverbs import imap 
from numpy_types import float64   


@macro
def arange(n, *xs):
  count = __builtin__.len(xs)
  assert 0 <= count <= 2, "Too many args for range: %s" % ((n,) + tuple(xs))
  if count == 0:
    return Range(zero_i64, n, one_i64)
  elif count == 1:
    return Range(n, xs[0], one_i64)  
  else:
    return Range(n, xs[0], xs[1])
 
 
@macro
def empty(shape, dtype = float64):

  def typed_empty(shape, dtype):
    # HACK! 
    # In addition to TypeValue, allow casting functions 
    # to be treated as dtypes 
    if isinstance(dtype, UntypedFn):
      assert len(dtype.body) == 1
      stmt = dtype.body[0]
      assert stmt.__class__ is Return 
      expr = stmt.value 
      assert expr.__class__ is Cast
      elt_t = expr.type 
    elif isinstance(dtype, TypedFn):
      elt_t = dtype.return_type
    else:
      assert isinstance(dtype.type, TypeValueT), \
         "Invalid dtype %s " % (dtype,)
      elt_t = dtype.type.type 
    assert isinstance(elt_t, ScalarT), \
       "Array element type %s must be scalar" % (elt_t,)  
    if isinstance(shape, ScalarT):
      shape = Tuple((shape,))
    rank = len(shape.type.elt_types)
    arr_t = make_array_type(elt_t, rank)
    return AllocArray(shape = shape, elt_type = elt_t, type = arr_t)
  return DelayUntilTyped(values=(shape,dtype), fn = typed_empty) 

@jit 
def empty_like(x, dtype = None):
  if dtype is None:
    return empty(x.shape, x.dtype)
  else:
    return empty(x.shape, dtype)
  
@jit 
def zeros(shape, dtype = float64):
  zero = dtype(0)
  return imap(lambda _: zero, shape)

@jit
def zeros_like(x, dtype = None):
  if dtype is None:
    dtype = x.dtype
  return zeros(x.shape, dtype)

@jit
def ones(shape, dtype = float64):
  one = dtype(1)
  return imap(lambda _: one, shape)

@jit
def ones_like(x, dtype = None):
  if dtype is None:
    dtype = x.dtype
  return ones(x.shape)

@jit
def copy(x):
  return [xi for xi in x]
