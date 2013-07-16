from expr import Expr 


class Array(Expr):
  _members = ['elts']

  def node_init(self):
    self.elts = tuple(self.elts)

  def children(self):
    return self.elts

  def __hash__(self):
    return hash(self.elts)


class Index(Expr):
  _members = ['value', 'index']
  
  def __eq__(self, other):
    return other.__class__ is Index and \
           other.value == self.value and \
           other.index == self.index
  
  def __hash__(self):
    return hash((self.value, self.index))
  
  def children(self):
    yield self.value
    yield self.index

  def __str__(self):
    return "%s[%s]" % (self.value, self.index)


class Slice(Expr):
  _members = ['start', 'stop', 'step']

  def __str__(self):
    return "slice(%s, %s, %s)"  % \
        (self.start.short_str(),
         self.stop.short_str(),
         self.step.short_str())

  def __repr__(self):
    return str(self)

  def children(self):
    yield self.start
    yield self.stop
    yield self.step

  def __eq__(self, other):
    return other.__class__ is Slice and \
           other.start == self.start and \
           other.stop == self.stop and \
           other.step == self.step
           
  def __hash__(self):
    return hash((self.start, self.stop, self.step))

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
    
    
