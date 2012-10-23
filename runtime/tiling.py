import copy, sys
from syntax import *

class Tiler():
  def __init__(self):
    self.env = []
    self.cur_id = 0

  def clear(self):
    self.env = []

  def add(self, f):
    self.env.append(f)

  def pretty_print(self):
    for f in self.env:
      f.pretty_print()

  def flatten(self, fs):
    self.cur_id += 1
    def create_f(f):
      new_f = copy.deepcopy(f)
      new_f.name += "_" + str(self.cur_id)
      for stmt in new_f.stmts:
        if isinstance(stmt, Adverb) and not stmt.f.is_scalar():
          stmt.f.name += "_" + str(self.cur_id)
      return new_f

    first = create_f(fs[0])
    self.env.append(first)
    for f in fs[1:]:
      self.env.append(create_f(f))
    return first

  def tile_adverb(self, adverb, f, running_nest):
    if isinstance(adverb, AllPairs):
      adverb.f.args[0].expanded = True
      adverb.f.args[1].expanded = True
      tiled_f = self._tile_function(adverb.f, running_nest)
      adverb.f.args[0].expanded = False
      adverb.f.args[0].expanded = False
      return TiledAllPairs(adverb.args, tiled_f)
    elif isinstance(adverb, Map):
      for i in range(len(adverb.args)):
        adverb.f.args[i].expanded = True
      tiled_f = self._tile_function(adverb.f, running_nest)
      for i in range(len(adverb.args)):
        adverb.f.args[i].expanded = False
      return TiledMap(adverb.args, tiled_f, adverb.axes)
    elif isinstance(adverb, Reduce):
      expanded = []
      for arg in f.args:
        if arg.expanded:
          for i in range(len(adverb.args)):
            if arg.name == adverb.args[i].name:
              expanded.append(i)
      if len(expanded) != 0 and len(expanded) != len(adverb.args):
        print "Need to promote an argument of Reduction to match expanded" + \
              "other arguments."
        sys.exit()
      if adverb.f.is_scalar():
        nested = self.flatten(running_nest)
      else:
        nested = self._tile_function(adverb.f, running_nest)
      group_axes = copy.copy(adverb.axes)
      for i in range(len(adverb.args)):
        if i in expanded:
          group_axes[i] += 1
      return TiledReduce(adverb.args, nested, adverb.combiner, adverb.init,
                         adverb.axes, group_axes)

  def _tile_function(self, f, running_nest):
    if f.is_scalar():
      return f

    nest_f = running_nest[-1]
    tiled_stmts = []
    for stmt in f.stmts:
      nest_f.stmts.append(copy.deepcopy(stmt))
      if isinstance(stmt, Adverb):
        if not stmt.f.is_scalar():
          nested_f = copy.deepcopy(stmt.f)
          nested_f.stmts = []
          running_nest.append(nested_f)
          nest_f.stmts[-1].f = nested_f
        tiled_stmts.append(self.tile_adverb(stmt, f, running_nest))
      else:
        tiled_stmts.append(copy.deepcopy(stmt))
    new_f = TypedFn(f.name + "_tiled_L1", f.args, tiled_stmts)
    self.env.append(new_f)
    return new_f

  def tile_function(self, f):
    if f.is_scalar():
      return f

    nest_f = copy.deepcopy(f)
    running_nest = [nest_f]
    nest_f.stmts = []
    self._tile_function(f, running_nest)
