from expr import Expr
from adverb_eval import AdverbEvalHelpers  


class AdverbEvalNotImplemented(Exception):
  def __init__(self, obj, context):
    self.obj = obj 
    self.context = context 
    
  def __str__(self):
    return "Semantics for adverb %s not implemented (called from %s)" % \
        (self.obj.__class__.__name__, self.context.__class__.__name__)
 
class Adverb(Expr, AdverbEvalHelpers):
  _members = ['fn']
    
  def transform(self, transformer):
    raise AdverbEvalNotImplemented(self,transformer)
  
class IndexAdverb(Adverb):
  _members = ['shape']
  
class IndexMap(IndexAdverb):
  """
  Map from each distinct index in the shape to a value 
  """
  pass 

 

class IndexAccumulative(IndexAdverb):
  _members = ['combine', 'init']
  
class IndexReduce(IndexAccumulative):
  """
  Expect the 'fn' field to take indices and produces 
  element values, whereas 'combine' takes pairs of element 
  values and combines them. 
  """

  pass 

    
class IndexScan(IndexAccumulative):
  _members = ['emit']
  
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

class Accumulative(DataAdverb):
  """
  Adverbs such as Reduce and Scan which carry an accumulated value and require a
  'combine' function to merge the accumulators resulting from parallel
  sub-computations.
  """
  _members = ['combine', 'init']



class Reduce(Accumulative):
  def __repr__(self):
    return "Reduce(axis = %s, args = (%s), type = %s, init = %s, map_fn = %s, combine = %s)" % \
        (self.axis,
         self.args_to_str(),
         self.type,
         self.init,
         self.fn_to_str(self.fn),
         self.fn_to_str(self.combine))

class Scan(Accumulative):
  _members = ['emit']

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

class IndexFilter(IndexAdverb, Filter):
  pass 

class FilterReduce(Reduce, Filter):
  """
  Like a normal reduce but skips some elements if they don't pass
  the predicate 'pred'
  """
  _members = ['pred'] 

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
  
  