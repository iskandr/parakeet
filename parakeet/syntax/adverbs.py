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
  
  
  def eval(self, context):
    """
    Experiment in attaching sequential adverb semantics to the syntax itself
    """
    dims = self.tuple_elts(self.shape)
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
        elt_result =  self.invoke(self.fn, (idx_tuple,))
        context.setidx(output, index_vars, elt_result)
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
  
  def eval(self, context):
    shape = context.transform_expr(self.shape)
    init = context.transform_if_expr(self.init)
    fn = context.transform_expr(self.fn)
    combine = context.transform_expr(self.combine)
    
    dims = self.tuple_elts(shape)
    n_loops = len(dims)
    
    zero = self.int(0)
    
    if init is not None or not self.is_none(init):
      if n_loops > 1:
        zeros = self.tuple([zero for _ in xrange(n_loops)])
        init = self.call(fn, [zeros])
      else:
        init = self.call(fn, [zero])
        
    def build_loops(index_vars, acc):
    
      n_indices = len(index_vars)
      if n_indices > 0:
        acc_value = acc.get()
      else:
        acc_value = acc 
      if n_indices == n_loops:
        
        idx_tuple = self.tuple(index_vars) if n_indices > 1 else index_vars[0] 
        elt_result =  self.call(fn, (idx_tuple,))
        acc.update(self.call(combine, (acc_value, elt_result)))
        return acc.get()
      
      def loop_body(acc, idx):
        new_value = build_loops(index_vars + (idx,), acc = acc)
        acc.update(new_value)
        return new_value
      return self.accumulate_loop(self.int(0), dims[n_indices], loop_body, acc_value)
    return build_loops(index_vars = (), acc = init)
   
  
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
  def eval(self, context, output = None):
    f = context.transform_expr(self.fn)
    values = context.transform_expr_list(self.args)
    axis = context.transform_if_expr(self.axis)
    niters, delayed_elts = self.map_prelude(f, values, axis)
    zero = self.int(0)
    first_elts = self.force_list(delayed_elts, zero)
    if output is None:
      output = self.create_output_array(f, first_elts, niters)
    def loop_body(idx):
      output_indices = self.build_slice_indices(self.rank(output), 0, idx)
      elt_result = self.call(f, [elt(idx) for elt in delayed_elts])
      self.setidx(output, output_indices, elt_result)
    self.parfor(niters, loop_body)
    return output

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

class FilterReduce(Reduce, Filter):
  """
  Like a normal reduce but skips some elements if they don't pass
  the predicate 'pred'
  """
  _members = ['pred'] 



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
  
  