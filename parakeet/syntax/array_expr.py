from seq_expr import SeqExpr 

class ArrayExpr(SeqExpr):
  """
  Common base class for first-order array operations 
  that don't change the underlying data 
  """
  def __init__(self, array, type = None, source_info = None):
    self.array = array 
    self.type = type 
    self.source_info = source_info 
    
  def children(self):
    yield self.array 

class Array(ArrayExpr):
  def __init__(self, elts, type = None, source_info = None):
    self.elts = tuple(elts) 
    self.type = type 
    self.source_info = source_info 

  def children(self):
    return self.elts

  def __hash__(self):
    return hash(self.elts)

class Slice(ArrayExpr):
  def __init__(self, start, stop, step, type = None, source_info = None):
    self.start = start 
    self.stop = stop 
    self.step = step 
    self.type = type 
    self.source_info = source_info 


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
  def __init__(self, shape, value, type = None, source_info = None):
    self.shape = shape 
    self.value = value 
    self.type = type 
    self.source_info = source_info 

  def children(self):
    yield self.shape 
    yield self.value 
  
class ConstArrayLike(ArrayExpr):
  """
  Create an array with the same shape as the first arg, but with all values set
  to the second arg
  """

  def __init__(self, array, value, type = None, source_info = None):
    self.array = array 
    self.value = value 
    self.type = type 
    self.source_info = source_info 
  
  def children(self):
    yield self.array 
    yield self.value   

class Range(ArrayExpr):
  def __init__(self, start, stop, step, type = None, source_info = None):
    self.start = start 
    self.stop = stop 
    self.step = step 
    self.type = type 
    self.source_info = source_info 

  def children(self):
    yield self.start 
    yield self.stop 
    yield self.step 
  
  
  def __str__(self):
    return "Range(start = %s, stop = %s, step = %s)" % (self.start, self.stop, self.step)

class AllocArray(ArrayExpr):
  """Allocate an unfilled array of the given shape and type"""
  def __init__(self, shape, elt_type, type = None, source_info = None):
    # TODO: support a 'fill' field 
    self.shape = shape 
    self.elt_type = elt_type 
    self.type = type 
    self.source_info = source_info 

  def children(self):
    yield self.shape
    
  def __str__(self):
    return "AllocArray(shape = %s, elt_type = %s)" % (self.shape, self.elt_type)


class ArrayView(ArrayExpr):
  """Create a new view on already allocated underlying data"""
  def __init__(self, data, shape, strides, offset, size, type = None, source_info = None):
    self.data = data 
    self.shape = shape 
    self.strides = strides 
    self.offset = offset 
    self.size = size 
    self.type = type 
    self.source_info = source_info

  def children(self):
    yield self.data
    yield self.shape
    yield self.strides
    yield self.offset
    yield self.size

class Ravel(ArrayExpr):
  def children(self):
    return (self.array,)

  def __str__(self):
    return "Ravel(%s)" % self.array 

class Reshape(ArrayExpr):
  def __init__(self, array, shape, type = None, source_info = None):
    self.array = array 
    self.shape = shape 
    self.type = type 
    self.source_info = source_info

  
  def children(self):
    yield self.array 
    yield self.shape
    
  def __str__(self):
    return "Reshape(%s, %s)" % (self.array, self.shape)

class Shape(ArrayExpr):
  
  def __str__(self):
    return "Shape(%s)" % self.array 
  
class Strides(ArrayExpr):
  def __str__(self):
    return "Strides(%s)" % self.array 
  
    
class Transpose(ArrayExpr):
  def children(self):
    yield self.array
  
  def __str__(self):
    return "%s.T" % self.array 
  
class Tile(ArrayExpr):
  def __init__(self, array, reps, type = None, source_info = None):
    self.array = array 
    self.reps = reps 
    self.type = type 
    self.source_info = source_info 

  def children(self):
    yield self.array 
    yield self.reps 
    
  def __str__(self):
    return "Tile(%s, %s)" % (self.array, self.reps)
    
