
from abc import ABCMeta, abstractmethod 

"""
Gosh, I really wish I had OCaml functors or Haskell's type classes
so I could more cleanly parameterize all this code by abstract
Elt, Value, Fn, etc... types
"""

class AbstractDimSize:
  __meta__ = ABCMeta
  
  def __eq__(self, other):
    return False 


class AbstractValue:
  __meta__ = ABCMeta 
  
  def elt(self, axis):
    """
    Slice this value along the given axis
    """ 
    return self 

  def lower(self):
    """    
    If the domain of elts is different from values, 
    then this function should project a value used
    as an accumulator down to the elt set
    """
    return self 
  
  @abstractmethod
  def size_along_axis(self, axis):
    pass 
  
  @abstractmethod 
  def has_axis(self, axis):
    pass 
  
class AbstractFn:
  __meta__ = ABCMeta 
  
  @abstractmethod 
  def delayed_apply(self, elts):
    # returns a computation representing how
    # get the result for each element 
    pass 
  

def niters(xs, axis):
  # exclude scalars
  args_with_axis = [x for x in xs if x.has_axis(axis)]
  axis_sizes = [x.size_along_axis(axis) for x in args_with_axis]
  assert len(axis_sizes) > 0
  sz = axis_sizes[0]
  # all arrays should agree in their dimensions along the 
  # axis we're iterating over 
  assert all(other_size == sz for other_size in axis_sizes[1:])
  return sz 
  

def collect(elt_result, d, axis):
  pass 
  
def eval_map(f, xs, axis):
  d = niters(xs)
  elts = [x.elt(axis) for x in xs]
  elt_result = f.delayed_apply(elts)
  return collect(elt_result, d, axis)

def eval_reduce(f, init, xs, axis):
  pass 

def eval_scan(f, init, xs, axis):
  pass 
