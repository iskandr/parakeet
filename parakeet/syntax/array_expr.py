from seq_expr import SeqExpr 

class ArrayExpr(SeqExpr):
  """
  Common base class for first-order array operations 
  that don't change the underlying data 
  """
  pass

class Array(ArrayExpr):
  _members = ['elts']

  def node_init(self):
    self.elts = tuple(self.elts)

  def children(self):
    return self.elts

  def __hash__(self):
    return hash(self.elts)

class Slice(ArrayExpr):
  _members = ['start', 'stop', 'step']

  def __str__(self):
    return "%s:%s:%s"  % \
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

class ConstArray(ArrayExpr):
  _members = ['shape', 'value']

class ConstArrayLike(ArrayExpr):
  """
  Create an array with the same shape as the first arg, but with all values set
  to the second arg
  """

  _members = ['array', 'value']

class Range(ArrayExpr):
  _members = ['start', 'stop', 'step']
  
  def __str__(self):
    return "Range(start = %s, stop = %s, step = %s)" % (self.start, self.stop, self.step)

class AllocArray(ArrayExpr):
  """Allocate an unfilled array of the given shape and type"""
  _members = ['shape', 'elt_type'] # TODO: support a 'fill' field 
  
  def children(self):
    yield self.shape
    
  def __str__(self):
    return "AllocArray(shape = %s, elt_type = %s)" % (self.shape, self.elt_type)


class ArrayView(ArrayExpr):
  """Create a new view on already allocated underlying data"""

  _members = ['data', 'shape', 'strides', 'offset', 'size']

  def children(self):
    yield self.data
    yield self.shape
    yield self.strides
    yield self.offset
    yield self.size

class Ravel(ArrayExpr):
  _members = ['array']
  
  def children(self):
    return (self.array,)

  def __str__(self):
    return "Ravel(%s)" % self.array 

class Reshape(ArrayExpr):
  _members = ['array', 'shape']
  
  def children(self):
    yield self.array 
    yield self.shape
    
  def __str__(self):
    return "Reshape(%s, %s)" % (self.array, self.shape)

class Shape(ArrayExpr):
  _members = ['array']
  
  def __str__(self):
    return "Shape(%s)" % self.array 
  
class Strides(ArrayExpr):
  _members = ['array']
  
  def __str__(self):
    return "Strides(%s)" % self.array 
  
    
class Transpose(ArrayExpr):
  _members = ['array']
  
  def children(self):
    yield self.array
  
  def __str__(self):
    return "%s.T" % self.array 
  
class Tile(ArrayExpr):
  _members = ["array", "reps"]
  
  def children(self):
    yield self.array 
    yield self.reps 
    
  def __str__(self):
    return "Tile(%s, %s)" % (self.array, self.reps)
    
