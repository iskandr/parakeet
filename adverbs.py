import syntax

class Adverb(syntax.Expr):
  _members = ['fn', 'args', 'axis']

  def node_init(self):
    assert self.fn is not None
    assert self.args is not None

  def __repr__(self):
    args_str = ", ".join([str(arg) for arg in self.args])
    return "%s(%s, %s, axis = %s)" % \
      (self.node_type(), self.fn, args_str, self.axis)

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
    args_str = ", ".join(self.args)
    return "%s(%s, %s, axis = %s, init = %s, combine = %s)" % \
      (self.node_type(), self.fn, args_str, self.axis, self.init, self.combine)

  def node_init(self):
    # assert self.init is not None
    # assert self.combine is not None
    pass 
  
  
class Reduce(Adverb):
  pass

class Scan(Accumulative):
  pass

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

