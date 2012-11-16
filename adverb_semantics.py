
"""
Gosh, I really wish I had OCaml functors or Haskell's type classes
so I could more cleanly parameterize all this code by abstract
Elt, Value, Fn, etc... types
"""  


class AbstractSemantics:
  
  @classmethod 
  def elt_repr(cls, value, axis):
    """
    Slice this value along the given axis
    """ 
    raise NotImplementedError
  
  @classmethod 
  def apply_repr(cls, fn, elts):
    # returns a computation representing how
    # get the result for each element 
    raise NotImplementedError 
  
  @classmethod 
  def size_along_axis(cls, value, axis):
    raise NotImplementedError
  
  @classmethod 
  def has_axis(cls, value, axis):
    raise NotImplementedError
  
  @classmethod 
  def collect(cls, elt_result, niters, axis):
    raise NotImplementedError  
  
  @classmethod 
  def accumulator(cls, init, elts):
    raise NotImplementedError
  
  @classmethod 
  def last_result(cls, elt_result, niters, axis):
    raise NotImplementedError 
  
  @classmethod 
  def adverb_prelude(cls, xs, axis):
    if not isinstance(xs, (list, tuple)):
      xs = [xs]
    # exclude scalars
    args_with_axis = [x for x in xs if cls.has_axis(x, axis)]
    axis_sizes = [cls.size_along_axis(x, axis) for x in args_with_axis]
    assert len(axis_sizes) > 0
    sz = axis_sizes[0]
    # all arrays should agree in their dimensions along the 
    # axis we're iterating over 
    assert all(other_size == sz for other_size in axis_sizes[1:])
    
    elts = [cls.elt_repr(x, axis) for x in xs]
    return sz, elts
  
  @classmethod  
  def eval_map(cls, f,  xs, axis):
    """
    Abstract specification of a map's behavior
    which relies on a semantics object S to 
    give specific/concrete meaning to this function
    for particular input argument types. 
    """
    niters, elts = cls.adverb_prelude(xs, axis)
    elt_result = cls.apply_repr(f, elts)
    return cls.collect(elt_result, niters, axis)
  
  @classmethod 
  def eval_reduce(cls, local_f, combine, init, xs, axis):
    niters, elts = cls.adverb_prelude(xs, axis)
    elt_result = cls.apply_repr(local_f, elts)
    acc = cls.accumulator(init)
    acc_result = cls.accumulate(combine, [acc, elt_result], niters)
    return acc_result[niters]
    
  #@classmethod 
  #def eval_scan(cls, mapper, combine, init, xs, axis):
  #  niters, elts = cls.adverb_prelude(xs, axis)
  #  elt_result = cls.apply_repr(mapper, elts)
  #  acc = cls.accumulator(init, elt_result)
  #  return cls.collect(acc, niters, axis)
  
  @classmethod
  def eval_scan(cls, mapper, combine, collect, init, axis, xs):
     
  
import numpy as np
 
class InterpSemantics(AbstractSemantics):
  """
  Instantiate the abstract abverb semantics with:
     type value = np.ndarray or scalar
     type dimsize = int 
     type elt_computation = (idx -> value)
  """
  @classmethod 
  def has_axis(cls, value, axis):
    return np.rank(value) > axis

  @classmethod 
  def size_along_axis(cls, value, axis):
    return value.shape[axis]

  @classmethod 
  def elt_repr(cls, value, axis):
    r = np.rank(value)
    if r < axis:
      return lambda idx: value 
    else:
      select_all = slice(None, None, None)
      indices = [select_all] * r 
      print indices, axis, value, r 
      def delayed_slice(idx):
        indices[axis] = idx
        return value[tuple(indices)]
      return delayed_slice
  
  @classmethod   
  def apply_repr(cls, fn, elt_reprs):
    return lambda idx: fn(*[e(idx) for e in elt_reprs]) 
  

  @classmethod
  def accumulator(cls, combine, init, delayed_elt):
    return lambda idx: return acc(idx)
    def delayed(idx):
      elt_value = delayed_elt(idx)
      
    return delayed
  @classmethod 
  def accumulate(cls, combine, init, map_repr, niters):
    acc = init 
    for idx in xrange(niters):
      acc = combine(acc, map_repr(idx))
    return acc 
   
  @classmethod 
  def accumulate_and_collect(cls, f, init, elt_reprs, niters):
    result = [None] * niters
    acc = init 
    for idx in xrange(niters):
      acc = f(acc, *[e(idx) for e in elt_reprs])
      result[idx] = acc
    return result 
  
  @classmethod
  def collect(cls, delayed_fn, niters, axis):
    elts = [delayed_fn(i) for i in xrange(niters)]
    return np.array(elts)

if __name__ == '__main__':
  x = np.array([1,4,9])
  y = InterpSemantics.eval_map(np.sqrt, [x], 0)
  print "Map Input", np.sqrt, x
  print "Output", y
  print "Expected output", np.sqrt(x)
  print
  
  total = InterpSemantics.eval_reduce(lambda x: x, np.add,  0, [x], 0) 
  print "Reduce Input", np.add, 0, x
  print "Reduce output", total 
  print "Expected output", np.sum(x)
  print 
  
  prefixes = InterpSemantics.eval_scan(
    mapper = lambda x:x, 
    combine = np.add, 
    init = 0, 
    values = [x], 
    axis = 0)
  
  
  
  
        
      
      
  
  
  
  