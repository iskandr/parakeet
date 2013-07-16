from expr import Expr
 
class AdverbNotImplemented(Exception):
  def __init__(self, obj):
    self.obj = obj 
    
  def __str__(self):
    return "Adverb %s not implemented" % (self.obj.__class__.__name__)
 
class Adverb(Expr):
  _members = ['fn']
  
  def eval(self, context):
    raise AdverbNotImplemented(self)
  
class IndexAdverb(Adverb):
  _members = ['shape']
  
  
  
class IndexMap(IndexAdverb):
  """
  Map from each distinct index in the shape to a value 
  """
  
  def eval(self, context):
    """
    Experiment in attaching sequential adverb semantics to the syntax itself
    """
    fn = context.eval(self.fn)
    shape = context.eval(self.shape)
    dims = self.tuple_elts(shape)
    if len(dims) == 1:
      shape = dims[0]
      
    if self.output is None:
      output = self.create_output_array(self.fn, [shape], shape)
    else:
      output = context.eval(self.output)

    n_loops = len(dims)
    def build_loops(index_vars = ()):
      n_indices = len(index_vars)
      if n_indices == n_loops:
        if n_indices > 1:
          idx_tuple = self.tuple(index_vars)
        else:
          idx_tuple = index_vars[0]
        elt_result =  self.invoke(fn, (idx_tuple,))
        self.setidx(output, index_vars, elt_result)
      else:
        def loop_body(idx):
          build_loops(index_vars + (idx,))
        self.loop(self.int(0), dims[n_indices], loop_body)
    build_loops()
    return output 
 

class ParFor(IndexAdverb):
  """
  Not really an adverb, since it doesn't return an array but only evaluates 
  its function argument for side effects
  """
  pass   
  
class IndexReduce(IndexAdverb):
  """
  Expect the 'fn' field to take indices and produces 
  element values, whereas 'combine' takes pairs of element 
  values and combines them. 
  """
  _members = ['combine', 'init']
  

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

class AllPairs(DataAdverb):
  pass 

class Accumulative(DataAdverb):
  """
  Adverbs such as Reduce and Scan which carry an accumulated value and require a
  'combine' function to merge the accumulators resulting from parallel
  sub-computations.
  """
  _members = ['combine', 'init']


  def node_init(self):
    # assert self.init is not None
    # assert self.combine is not None
    pass

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

class TiledAllPairs(Tiled, AllPairs):
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
  
  