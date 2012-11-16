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
  
  def apply_to_delayed(self, fn, args, idx):
    curr_args = [x(idx) for x in args]
    return self.apply(fn, curr_args)
  
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

  def build_slice_indices(self, rank, axis, idx):
    indices = []
    for i in xrange(rank):
      if i == axis:
        indices.append(idx)
      else:
        indices.append(self.slice(self.none, self.none, self.const_int(1)))
    return self.tuple(indices)
  
  def slice_along_axis(self, arr, axis, idx):
    r = self.rank(arr)
    index_tuple = self.build_slice_indices(r, axis, idx)
    return self.index(arr, index_tuple)
  
  def delayed_elt(self, x, axis):
    return lambda idx: self.slice_along_axis(x, axis, idx)
  
  def adverb_prelude(self, xs, axis):
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
    return axis_sizes[0], elts
  
  def eval_map(self, f,  xs, axis):
    return self.eval_scan(
      mapper = f, 
      combine = self.trivial_combiner,  
      emit = self.identity_function,  
      init = None,
      inclusive = True,  
      values = xs, 
      axis = axis) 
  
  
  def eval_reduce(self, f, combine, init, xs, axis):
    prefixes = self.eval_scan(
      mapper = f, 
      combine = combine, 
      emit = self.identity_function,  
      init = init, 
      inclusive = False, 
      values = xs, 
      axis = axis )
    return prefixes[-1]
  
  def eval_scan(self, mapper, combine, emit, init, inclusive, values, axis):
    niters, elts = self.adverb_prelude(values, axis)
    delayed_map_result = lambda idx: self.apply_to_delayed(mapper, elts, idx)
    zero = self.const_int(0)
    one = self.const_int(1)
    if init is None:
      init = delayed_map_result(0)
      start_idx = one 
    else:
      start_idx = zero
    result_size = self.sub(niters, start_idx)
    if inclusive:
      result = self.array(self.add(result_size, one), init)  
      # start_idx = self.add(start_idx, one)  
    else:
      result = self.array(result_size, init)
 
    acc = self.accumulator(init)
    emitted_elt_repr = lambda idx: self.apply(emit, [self.get_acc(acc)])
    def loop_body(idx):
      return [
        self.set_acc(acc, 
          self.apply(combine, [self.get_acc(acc), delayed_map_result(idx)])),
        self.setidx(result, idx, emitted_elt_repr(idx))
      ]
    self.loop(start_idx, niters, loop_body)
    return result 
  

if __name__ == '__main__':
  interp = AdverbSemantics()
  x = np.array([1,4,9])
  y = interp.eval_map(np.sqrt, [x], 0)
  print "Map Input", np.sqrt, x
  print "Output", y
  print "Expected output", np.sqrt(x)
  print
  
  total = interp.eval_reduce(lambda x: x, np.add,  0, [x], 0) 
  print "Reduce Input", np.add, 0, x
  print "Reduce output", total 
  print "Expected output", np.sum(x)
  print 
  
  prefixes = interp.eval_scan(
    mapper = lambda x:x, 
    combine = np.add, 
    emit = lambda x: x, 
    init = 0, 
    inclusive = False, 
    values = [x], 
    axis = 0)
  print "Scan Input", x
  print "Scan output", prefixes
  print "Expected Scan output", np.cumsum(x)
  
  
  
  
  
        
      
      
  
  
  
  