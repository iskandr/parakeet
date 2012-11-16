import numpy as np

class BaseSemantics:
  """
  Gosh, I really wish I had OCaml functors or Haskell's type classes
  so I could more cleanly parameterize all this code by these abstract types:
    Fn       -- representations of functions
                i.e. the interpreter domain might
                have a function "lambda x: x+1" whereas
                in the type domain this function is represented
                by "Int -> Int"
    Value    -- arrays, scalars, and tuples
    DimSize  -- abstract representation of shape elements

  Also, for code generation of loops we need
    Idx          -- representation of loop indices (possibly same as DimSize?)
    DelayedValue -- mapping from index to abstract value
    Acc          -- object with
  """

  def size_along_axis(self, value, axis):
    return value.shape[axis]

  def rank(self, value):
    return np.rank(value)

  def accumulator(self, v):
    return [v]

  def get_acc(self, acc):
    return acc[0]

  def set_acc(self, acc, v):
    acc[0] = v

  def const_int(self, x):
    return int(x)

  def add(self, x, y):
    return x + y

  def sub(self, x, y):
    return x - y

  def array(self, size, elt):
    return np.array([elt] * size)

  def shift_array(self, arr, offset):
    return arr[offset:]

  def index(self, arr, idx):
    return arr[idx]

  def tuple(self, elts):
    return tuple(elts)

  def setidx(self, arr, idx, v):
    print "arr", arr
    print "idx", idx
    print "value", v
    arr[idx] = v

  def loop(self, start_idx, stop_idx, body):
    for i in xrange(start_idx, stop_idx):
      body(i)

  def check_equal_sizes(self, sizes):
    assert len(sizes) > 0
    first = sizes[0]
    assert all(sz == first for sz in sizes[1:])

  def slice_value(self, start, stop, step):
    return slice(start, stop, step)

  def apply(self, fn, args):
    return fn(*args)


  none = None
  null_slice = slice(None, None, None)
  def identity_function(self, x):
    return x
  def trivial_combiner(self, x, y):
    return y

class AdverbSemantics(BaseSemantics):
  """
  Describe the behavior of adverbs in terms of
  lower-level value and iteration constructs.

  To get something other than an unfathomably slow
  interpreter, override all the methods of BaseSemantics
  and make them work for some other domain (such as types,
  shapes, or compiled expressions)
  """
  def apply_to_delayed(self, fn, args, idx):
    curr_args = [x(idx) for x in args]
    return self.apply(fn, curr_args)

  def build_slice_indices(self, rank, axis, idx):
    indices = []
    for i in xrange(rank):
      if i == axis:
        indices.append(idx)
      else:
        s = self.slice_value(self.none, self.none, self.const_int(1))
        indices.append(s)
    return self.tuple(indices)

  def slice_along_axis(self, arr, axis, idx):
    r = self.rank(arr)
    index_tuple = self.build_slice_indices(r, axis, idx)
    return self.index(arr, index_tuple)

  def delayed_elt(self, x, axis):
    return lambda idx: self.slice_along_axis(x, axis, idx)

  def adverb_prelude(self, map_fn, xs, axis):
    if not isinstance(xs, (list, tuple)):
      xs = [xs]
    axis_sizes = [self.size_along_axis(x, axis)
                  for x in xs
                  if self.rank(x) >= axis]
    assert len(axis_sizes) > 0

    # all arrays should agree in their dimensions along the
    # axis we're iterating over
    self.check_equal_sizes(axis_sizes)
    elts = [self.delayed_elt(x, axis) for x in xs]
    def delayed_map_result(idx):
      return self.apply_to_delayed(map_fn, elts, idx)
    return axis_sizes[0], delayed_map_result

  def eval_map(self, f,  values, axis):
    return self.eval_scan(
      map_fn = f,
      combine = self.trivial_combiner,
      emit = self.identity_function,
      init = None,
      values = values,
      axis = axis)

  def eval_reduce(self, map_fn, combine, init, values, axis):
    prefixes = self.eval_scan(
      map_fn = map_fn,
      combine = combine,
      emit = self.identity_function,
      init = init,
      values = values,
      axis = axis )
    return self.index(prefixes, self.const_int(-1))

  def eval_scan(self, map_fn, combine, emit, init, values, axis):
    niters, delayed_map_result = self.adverb_prelude(map_fn, values, axis)

    if init is None:
      init = delayed_map_result(0)
    else:
      # combine the provided initializer with
      # transformed first value of the data
      # in case we need to coerce up
      init = self.apply(combine, [init, delayed_map_result(0)])
    start_idx = self.const_int(1)
    first_output = self.apply(emit, [init])
    result = self.array(niters, first_output)

    acc = self.accumulator(init)
    emitted_elt_repr = lambda idx: self.apply(emit, [self.get_acc(acc)])
    def loop_body(idx):
      output_indices = self.build_slice_indices(self.rank(result), axis, idx)

      return [
        self.set_acc(acc,
          self.apply(combine, [self.get_acc(acc), delayed_map_result(idx)])),
        self.setidx(result, output_indices, emitted_elt_repr(idx))
      ]
    self.loop(start_idx, niters, loop_body)
    return result
