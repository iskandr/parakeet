import adverb_semantics 
import numpy as np 

class Accumulator:
  def __init__(self, value):
    self.value = value 
  
  def get(self):
    return self.value 
  
  def update(self, new_value):
    self.value = new_value 

class InterpSemantics(adverb_semantics.AdverbSemantics):
  def size_along_axis(self, value, axis):
    assert len(value.shape) > axis, \
      "Can't get %d'th element of %s with shape %s" % (axis, value, value.shape)
    return value.shape[axis]

  def is_tuple(self, x):
    return isinstance(x, tuple)

  def rank(self, value):
    return np.rank(value)

  def int(self, x):
    return int(x)

  def add(self, x, y):
    return x + y

  def sub(self, x, y):
    return x - y

  def shape(self, x):
    return x.shape 
    
  def elt_type(self, x):
    return x.dtype if hasattr(x, 'dtype') else type(x) 
  
  def alloc_array(self, size, elt_type):
    return np.zeros(size, dtype = elt_type)

  def shift_array(self, arr, offset):
    return arr[offset:]

  def index(self, arr, idx):
    return arr[idx]

  def tuple(self, elts):
    return tuple(elts)

  def concat_tuples(self, t1, t2):
    return tuple(t1) + tuple(t2)
  
  def setidx(self, arr, idx, v):
    arr[idx] = v

  def loop(self, start_idx, stop_idx, body):
    for i in xrange(start_idx, stop_idx):
      body(i)

  def accumulate_loop(self, start_idx, stop_idx, body, init):
    acc = Accumulator(init)
    for i in xrange(start_idx, stop_idx):
      body(acc, i)
    return acc.get()    

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