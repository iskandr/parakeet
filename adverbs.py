import syntax

class Adverb(syntax.Expr):
  _members = ['fn', 'args', 'axis']

  def node_init(self):
    assert self.fn is not None
    assert self.args is not None

  def __repr__(self):
    args_str = ", ".join([str(arg) for arg in self.args])
    return "%s(axis = %s, fn = %s, %s)" % \
        (self.node_type(), self.axis, self.fn, args_str)

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
  Adverbs such as Reduce and Scan
  which carry an accumulated value
  and require a 'combine' function
  to merge the accumulators resulting
  from parallel sub-computations.
  """
  _members = ['combine', 'init']

  def __repr__(self):
    args_str = ", ".join([str(x) for x in self.args])
    return "%s(axis = %s, map_fn = %s, combine = %s, init = %s, %s)" % \
        (self.node_type(), self.axis, self.fn, self.combine,
         self.init, args_str)

  def node_init(self):
    # assert self.init is not None
    # assert self.combine is not None
    pass

class Reduce(Accumulative):
  pass

class Scan(Accumulative):
  _members = ['emit']

  def __repr__(self):
    args_str = ", ".join([str(x) for x in self.args])
    return "%s(axis = %s, map_fn = %s, combine = %s, emit = %s, init = %s, %s)"\
        % (self.node_type(), self.axis, self.fn, self.combine, self.emit,
           self.init, args_str)

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
