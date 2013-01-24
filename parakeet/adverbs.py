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
    s = "%s(axis = %s, args = (%s), type=%s, fn = %s" % \
          (self.node_type(), self.axis,
           self.args_to_str(),
           self.type,
           self.fn_to_str(self.fn),)
    if self.out:
      s += ", out=%s" % self.out
    s += ")"
    return s

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
    s = ("%s(axis = %s, args = (%s), type = %s, init = %s, map_fn = %s, " + \
         "combine = %s") % \
        (self.node_type(), self.axis,
         self.args_to_str(),
         self.type,
         self.init,
         self.fn_to_str(self.fn),
         self.fn_to_str(self.combine))
    if self.out:
      s += ", out = %s" % self.out
    s += ")"
    return s

  def node_init(self):
    # assert self.init is not None
    # assert self.combine is not None
    pass

class Reduce(Accumulative):
  pass

class Scan(Accumulative):
  _members = ['emit']

  def __repr__(self):
    s = "%s(axis = %s, args = {%s}, type = %s, " \
        % (self.node_type(), self.axis,
           self.args_to_str(),
           self.type
          )
    s += "init = %s, map_fn = %s, combine = %s, emit = %s" % \
         (self.init,
          self.fn_to_str(self.fn),
          self.fn_to_str(self.combine),
          self.fn_to_str(self.emit))
    if self.out:
      s += ", out = %s" % self.out
    s += ")"
    return s

class Tiled(object):
  _members = ['axes', 'fixed_tile_size']

  def __repr__(self):
    s = "%s(axes = %s, args = (%s), type=%s, fn = %s" % \
          (self.node_type(), self.axes,
           self.args_to_str(),
           self.type,
           self.fn_to_str(self.fn),)
    if self.out:
      s += ", out=%s" % self.out
    s += ")"
    return s

class TiledMap(Tiled, Map):
  pass

class TiledAllPairs(Tiled, AllPairs):
  pass

class TiledReduce(Tiled, Reduce):
  def __repr__(self):
    s = ("%s(axes = %s, args = {%s}, type = %s, init = %s, map_fn = %s, " + \
         "combine = %s") % \
        (self.node_type(), self.axes,
         self.args_to_str(),
         self.type,
         self.init,
         self.fn_to_str(self.fn),
         self.fn_to_str(self.combine),
        )
    if self.out:
      s += ", out = %s" % self.out
    s += ")"
    return s

class TiledScan(Tiled, Scan):
  def __repr__(self):
    s = "%s(axes = %s, args = {%s}, type = %s, " \
        % (self.node_type(), self.axes,
           self.args_to_str(),
           self.type
          )
    s += "init = %s, map_fn = %s, combine = %s, emit = %s" % \
         (self.init,
          self.fn_to_str(self.fn),
          self.fn_to_str(self.combine),
          self.fn_to_str(self.emit))
    if self.out:
      s += ", out = %s" % self.out
    s += ")"
    return s
