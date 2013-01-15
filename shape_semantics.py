from adverb_semantics import AdverbSemantics
from shape import  Const, Shape, Tuple, Closure, ConstSlice
from shape import Slice, Scalar, Add, Sub, Div
from shape import any_scalar,  const
from shape import is_zero, is_one, make_shape

class ShapeSemantics(AdverbSemantics):
  def size_along_axis(self, value, axis):
    assert isinstance(value, Shape)
    return value.dims[axis]

  def is_tuple(self, x):
    return isinstance(x, Tuple)

  def is_none(self, x):
    return isinstance(x, Const) and x.value is None

  def rank(self, value):
    if isinstance(value, Shape):
      return value.rank
    else:
      return 0

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
      return Add(x,y)

  def sub(self, x, y):
    if is_zero(y):
      return x
    elif isinstance(x, Const) and isinstance(y, Const):
      return const(x.value - y.value)
    else:
      return Sub(x, y)

  def div(self, x, y):
    assert not is_zero(y)
    if is_one(y):
      return x
    elif isinstance(x, Const) and isinstance(y, Const):
      return const(int(x.value / y.value))
    else:
      return Div(x, y)

  def shape(self, x):
    if isinstance(x, Shape):
      return Tuple(x.dims)
    else:
      return Tuple(())

  def elt_type(self, x):
    return "DON'T CARE ABOUT ELT TYPES"

  def alloc_array(self, _, dims):
    return make_shape(dims)

  def index(self, arr, idx):
    if isinstance(arr, Scalar):
      return arr
    assert arr.__class__ is Shape
    if isinstance(idx, (Scalar, Slice)):
      indices = [idx]
    elif idx.__class__ is Tuple:
      indices = idx.elts
    else:
      assert False, "Unexpected index: %s" % (idx,)
    result_dims = []
    for (i, curr_idx) in enumerate(indices):
      old_dim = arr.dims[i]
      if curr_idx is None or \
         (isinstance(curr_idx, Const) and curr_idx.value is None):
        result_dims.append(old_dim)
      elif isinstance(curr_idx, Scalar):
        pass
      elif curr_idx.__class__ is ConstSlice:
        result_dims.append(curr_idx.nelts)
      else:
        assert curr_idx.__class__ is Slice, \
          "Unsupported index %s" % curr_idx
        if isinstance(curr_idx.start, Const):
          if curr_idx.start.value is None:
            lower = const(0)
          elif curr_idx.start.value < 0:
            lower = self.sub(old_dim, curr_idx.start)
          else:
            lower = curr_idx.start
        else:
          lower = any_scalar

        if isinstance(curr_idx.stop, Const):
          if curr_idx.stop.value is None:
            upper = old_dim
          elif curr_idx.stop.value < 0:
            upper = self.sub(old_dim, curr_idx.stop)
          else:
            upper = curr_idx.stop
        else:
          upper = any_scalar

        n = self.sub(upper, lower)
        step = curr_idx.step
        if step and \
            isinstance(step, Const) and \
            step.value is not None and \
            step.value != 1:
          n = self.div(n, step)
        result_dims.append(n)
    n_original = len(arr.dims)
    n_idx= len(indices)
    if n_original > n_idx:
      result_dims.extend(arr.dims[n_idx:])

    return make_shape(result_dims)

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
    
    # if all elements of the slice are constant 
    # then we can ignore the exact start/stop/step 
    # and track only the number of elements in the 
    # slice
    if start.__class__ is Const and \
       stop.__class__ is Const and \
       stop.value is not None and \
       step.__class__ is Const:
      start_val = start.value
      if start_val is None:
        start_val = 0
      step_val = step.value
      if step_val is None:
        step_val = 1
      nelts = (stop.value - start_val) / step_val
      return ConstSlice(nelts)
    else:
      return Slice(start, stop, step)

  def invoke(self, fn, args):
    if fn.__class__ is Closure:
      args = tuple(fn.args) + tuple(args)
      fn = fn.fn
    import shape_inference
    return shape_inference.symbolic_call(fn, args)

  none = None
  null_slice = slice(None, None, None)

  def identity_function(self, x):
    return x
