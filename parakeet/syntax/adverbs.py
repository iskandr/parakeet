from dsltools import Node 
from expr import Expr
  


class Adverb(Expr, Node):
  _members = ['fn', 'output', 'type', 'source_info']
  
  def node_init(self):
    assert self.fn, "Can't construct adverb %s without a function argument" % self 
  def functions(self):
    yield self.fn 
    

      
class Accumulative(Expr):
  """
  Adverbs such as Reduce and Scan which carry an accumulated value and require a
  'combine' function to merge the accumulators resulting from parallel
  sub-computations.
  """
  _members = ['combine', 'init']
  
class HasEmit(Expr):
  """
  Common base class for Scan, IndexScan, and whatever other sorts of scans can be dreamed up
  """
  _members = ['emit']
  
  

class IndexAdverb(Adverb):
  _members = ['shape', 'start_index']
  
  
class IndexMap(IndexAdverb):
  """
  Map from each distinct index in the shape to a value 
  """
  pass 


class IndexAccumulative(IndexAdverb, Accumulative):
  pass 
  
class IndexReduce(IndexAccumulative):
  """
  Expect the 'fn' field to take indices and produces 
  element values, whereas 'combine' takes pairs of element 
  values and combines them. 
  """
  pass 

    
class IndexScan(IndexAccumulative, HasEmit):
  pass 
  
class DataAdverb(Adverb):
  _members = ['args', 'axis']

  def fn_to_str(self, fn):
    if hasattr(fn, 'name'):
      return fn.name
    else:
      return str(fn)

  def args_to_str(self):
    if isinstance(self.args, (list, tuple)):
      return ", ".join([str(arg) + ":" + str(arg.type) for arg in self.args])
    else:
      return str(self.args)

  def __repr__(self):
    s = "%s(axis = %s, args = (%s), type=%s, fn = %s)" % \
          (self.node_type(), self.axis,
           self.args_to_str(),
           self.type,
           self.fn_to_str(self.fn),)
    return s

  def __str__(self):
    return repr(self)

class Map(DataAdverb):
  pass 

class OuterMap(DataAdverb):
  pass 



class DataAccumulative(DataAdverb, Accumulative):
  pass 


class Reduce(DataAccumulative):
  def __repr__(self):
    return "Reduce(axis = %s, args = (%s), type = %s, init = %s, map_fn = %s, combine = %s)" % \
        (self.axis,
         self.args_to_str(),
         self.type,
         self.init,
         self.fn_to_str(self.fn),
         self.fn_to_str(self.combine))

class Scan(DataAccumulative, HasEmit):
  def __repr__(self):
    s = "%s(axis = %s, args = {%s}, type = %s, " \
        % (self.node_type(), self.axis,
           self.args_to_str(),
           self.type
          )
    s += "init = %s, map_fn = %s, combine = %s, emit = %s)" % \
         (self.init,
          self.fn_to_str(self.fn),
          self.fn_to_str(self.combine),
          self.fn_to_str(self.emit))
    return s




class Filter(Adverb):
  """
  Filters its arguments using the boolean predicate field 'fn'
  """
  pass 

class HasPred(Filter):
  _members = ['pred']


class Where(DataAdverb):
  """
  Applies 'fn' to each element of the arguments and 
  returns indices where 'pred' is True for the resulting
  element values
  """
  _members = ['pred']

class IndexFilter(IndexAdverb, Filter):
  pass 


class FilterReduce(Reduce, Filter):
  """
  Like a normal reduce but skips some elements if they don't pass
  the predicate 'pred'
  """
  pass  
  
  
class IndexFilterReduce(FilterReduce):
  pass 

class Tiled(object):
  _members = ['axes', 'fixed_tile_size']

  def __repr__(self):
    s = "%s(axes = %s, args = (%s), type=%s, fn = %s)" % \
          (self.node_type(), self.axes,
           self.args_to_str(),
           self.type,
           self.fn_to_str(self.fn),)
    return s

class TiledMap(Tiled, Map):
  pass

class TiledOuterMap(Tiled, OuterMap):
  pass

class TiledReduce(Tiled, Reduce):
  def __repr__(self):
    s = ("%s(axes = %s, args = {%s}, type = %s, init = %s, map_fn = %s, combine = %s)") % \
        (self.node_type(), self.axes,
         self.args_to_str(),
         self.type,
         self.init,
         self.fn_to_str(self.fn),
         self.fn_to_str(self.combine),
        )
    return s

class TiledScan(Tiled, Scan):
  def __repr__(self):
    s = "%s(axes = %s, args = {%s}, type = %s, " \
        % (self.node_type(), self.axes,
           self.args_to_str(),
           self.type
          )
    s += "init = %s, map_fn = %s, combine = %s, emit = %s)" % \
         (self.init,
          self.fn_to_str(self.fn),
          self.fn_to_str(self.combine),
          self.fn_to_str(self.emit))
    return s

  
class Conv(Adverb):
  _members = ['x', 'window_shape']
  
  def __repr__(self):
    return "Conv(fn = %s, x = %s, window_shape=%s)" % \
      (self.fn, self.x, self.window_shape)
      
  def __str__(self):
      return repr(self)

class ConvBorderFn(Conv):
  _members = ['border_fn']
  
class ConvBorderValue(Conv):
  _members = ['border_value']
  
class ConvPadding(Conv):
  _members = ['fill_value']
  
  