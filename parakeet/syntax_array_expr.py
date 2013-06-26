from syntax_expr import Expr 


#############################################################################
#
#  Array Operators: It's not scalable to keep adding first-order operators
#  at the syntactic level, so eventually we'll need some more extensible
#  way to describe the type/shape/compilation semantics of array operators
#
#############################################################################

class Len(Expr):
  _members = ['value']

class ConstArray(Expr):
  _members = ['shape', 'value']

class ConstArrayLike(Expr):
  """
  Create an array with the same shape as the first arg, but with all values set
  to the second arg
  """

  _members = ['array', 'value']

class Range(Expr):
  _members = ['start', 'stop', 'step']

class AllocArray(Expr):
  """Allocate an unfilled array of the given shape and type"""
  _members = ['shape', 'elt_type']
  
  def children(self):
    yield self.shape


class ArrayView(Expr):
  """Create a new view on already allocated underlying data"""

  _members = ['data', 'shape', 'strides', 'offset', 'size']

  def children(self):
    yield self.data
    yield self.shape
    yield self.strides
    yield self.offset
    yield self.size

class Ravel(Expr):
  _members = ['array']
  
  def children(self):
    return (self.array,)

class Reshape(Expr):
  _members = ['array', 'shape']
  
  def children(self):
    yield self.array 
    yield self.shape

class Shape(Expr):
  _members = ['array']
  
class Strides(Expr):
  _members = ['array']
    
class Transpose(Expr):
  _members = ['array']
  
  def children(self):
    yield self.array 
    
class Where(Expr):
  """
  Given a boolean array, returns its true indices 
  """
  _members = ['array']
  
  def children(self):
    yield self.array 
