from tree import TreeLike
import numpy as np
import math 


prim_lookup = {}

def find_prim(fn):
  return prim_lookup[fn]

def is_prim(fn):
  return fn in prim_lookup

class Prim:
    
  def __init__(self, fn, name = None, nin = None, nout = None):
    self.fn = fn
    if name:
      self.name = name
    else: 
      self.name = fn.__name__
    
    if nin:
      self.nin = nin
    elif hasattr(fn, 'nin'):
      self.nin = fn.nin  
    else:
      assert hasattr(fn, 'func_code')
      self.nin = fn.func_code.co_argcount 
      
    if nout:
      self.nout = nout
    elif hasattr(fn, 'nout'):
      self.nout = fn.nout 
    else:
      self.nout = 1
    
    prim_lookup[fn] = self 
  
  def __call__(self, *args, **kwds):
    return self.fn(*args, **kwds)

class Float(Prim):
  """Always returns a float"""
  pass   

class Logical(Prim):
  """Expects boolean inputs, returns a boolean"""
  pass 

class Cmp(Prim):
  """Takes two arguments of any type, returns a boolean"""
  pass 

class ArrayProp(Prim):
  """Array properties: shape and strides, return tuples of ints"""

sqrt = Float(np.sqrt)
log = Float(np.log)
sqrt = Float(np.sqrt) 
log10 = Float(np.log10)   
log2 = Float(np.log2)  
cos = Float(np.cos) 
cosh = Float(np.cosh) 
sin = Float(np.sin)  
sinh = Float(np.sinh) 
sinc = Float(np.sinc) 
tan = Float(np.tan) 
tanh = Float(np.tanh)

logical_and = Logical(np.logical_and)
logical_not = Logical(np.logical_not) 
logical_or = Logical(np.logical_or)
logical_xor = Logical(np.logical_xor) 

add = Prim(np.add) 
subtract = Prim(np.subtract) 
multiply = Prim(np.multiply) 
divide = Prim(np.divide)

equal = Cmp(np.equal)
not_equal = Cmp(np.not_equal)
less = Cmp(np.less)
less_equal = Cmp(np.less_equal)
greater = Cmp(np.greater)
greater_equal = Cmp(np.greater_equal)

shape = ArrayProp(np.shape)
strides = ArrayProp(lambda x: x.strides, name="strides") 
  