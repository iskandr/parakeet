import adverb_semantics 
import numpy as np 

class InterpSemantics(adverb_semantics.AdverbSemantics):
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

  def int(self, x):
    return int(x)

  def add(self, x, y):
    return x + y

  def sub(self, x, y):
    return x - y

  def repeat_array(self, size, elt):
    return np.array([elt] * size)

  def shift_array(self, arr, offset):
    return arr[offset:]

  def index(self, arr, idx):
    return arr[idx]

  def tuple(self, elts, name = None):
    return tuple(elts)

  def setidx(self, arr, idx, v):
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

  def invoke(self, fn, args):
    return fn(*args)

  none = None
  null_slice = slice(None, None, None)
  def identity_function(self, x):
    return x
  def trivial_combiner(self, x, y):
    return y

adverb_evaluator = InterpSemantics()