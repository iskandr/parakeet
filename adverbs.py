import syntax

class Adverb(syntax.Expr):
  _members = ['fn', 'args', 'axis', 'out']

  def fn_to_str(self, fn):
#    if isinstance(fn, (syntax.Fn, syntax.TypedFn)):
#      return fn.name
#    else:
    return str(fn)

  def args_to_str(self):
    if isinstance(self.args, (list, tuple)):
      return ", ".join([str(arg) + ":" + str(arg.type) for arg in self.args])
    else:
      return str(self.args)

  def __repr__(self):
    return "%s(axis = %s, fn = %s, %s, type=%s)" % \
        (self.node_type(), self.axis,
         self.fn_to_str(self.fn),
         self.args_to_str(),
         self.type)

  def __str__(self):
    return repr(self)

class Map(Adverb):
  pass

class AllPairs(Adverb):
  def node_init(self):
    if self.axis is None:
      self.axis = 0

class Accumulative(Adverb):
  """
  Adverbs such as Reduce and Scan which carry an accumulated value and require a
  'combine' function to merge the accumulators resulting from parallel
  sub-computations.
  """
  _members = ['combine', 'init']

  def __repr__(self):
    return "%s(axis = %s, map_fn = %s, combine = %s, init = %s, %s)" % \
        (self.node_type(), self.axis,
         self.fn_to_str(self.fn),
         self.fn_to_str(self.combine),
         self.init,
         self.args_to_str())

  def node_init(self):
    # assert self.init is not None
    # assert self.combine is not None
    pass

class Reduce(Accumulative):
  pass

class Scan(Accumulative):
  _members = ['emit']

  def __repr__(self):
    return "%s(axis = %s, map_fn = %s, combine = %s, emit = %s, init = %s, %s)"\
        % (self.node_type(), self.axis,
           self.fn_to_str(self.fn),
           self.fn_to_str(self.combine),
           self.fn_to_str(self.emit),
           self.init,
           self.args_to_str())

class Tiled(object):
  pass

class TiledMap(Map, Tiled):
  pass

class TiledAllPairs(AllPairs, Tiled):
  pass

class TiledReduce(Reduce, Tiled):
  pass

class TiledScan(Scan, Tiled):
  pass
