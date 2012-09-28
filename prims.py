from tree import TreeLike
import numpy as np
import math 


prim_lookup_by_value = {}

def find_prim(fn):
  return prim_lookup_by_value[fn]

prim_lookup_by_op_name = {}

def op_name(op):
  return op.__class__.__name__

def is_ast_op(op):
  return op_name(op) in prim_lookup_by_op_name

def find_ast_op(op):
  name = op_name(op)
  if name in prim_lookup_by_op_name:
    return prim_lookup_by_op_name[name]
  else:
    raise RuntimeError("Operator not implemented: %s" % name)


def is_prim(fn):
  return fn in prim_lookup_by_value

class Prim:
    
  def __init__(self, fn, python_op_name = None,  name = None, nin = None, nout = None):
    self.fn = fn
    prim_lookup_by_value[fn] = self
    
    self.python_op_name = python_op_name
    if python_op_name is not None:
      prim_lookup_by_op_name[python_op_name] = self
       
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
  
  def __call__(self, *args, **kwds):
    return self.fn(*args, **kwds)
  
  def __repr__(self):
    return "prim(%s)" % self.name 

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

logical_and = Logical(np.logical_and, 'BitAnd')
logical_not = Logical(np.logical_not, 'Invert') 
logical_or = Logical(np.logical_or, 'BitOr')
logical_xor = Logical(np.logical_xor, 'BitXor') 

add = Prim(np.add, 'Add') 
subtract = Prim(np.subtract, 'Sub') 
multiply = Prim(np.multiply, 'Mult') 
divide = Prim(np.divide, 'Div')

equal = Cmp(np.equal, 'Eq')
not_equal = Cmp(np.not_equal, 'NotEq')
less = Cmp(np.less, 'Lt')
less_equal = Cmp(np.less_equal, 'LtE')
greater = Cmp(np.greater, 'Gt')
greater_equal = Cmp(np.greater_equal, 'GtE')

shape = ArrayProp(np.shape)
strides = ArrayProp(lambda x: x.strides, name="strides") 
  