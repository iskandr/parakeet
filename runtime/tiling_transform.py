#!/usr/bin/python

class Adverb():
  pass

class AllPairs(Adverb):
  def __init__(self, args, f, axes=[0,0]):
    self.f = f
    self.args = args
    self.axes = axes

  def pretty_print(self, indent=0):
    print "  " * indent + ("AllPairs([%s, %s], %s, axes=[%d,%d])" %
        (self.args[0].to_string(), self.args[1].to_string(), self.f.name,
         self.axes[0], self.axes[1]))

class Map(Adverb):
  def __init__(self, args, f, axes=[]):
    self.f = f
    self.args = args
    if len(axes) == 0:
      self.axes = [0 for _ in args]
    else:
      self.axes = axes

  def pretty_print(self, indent=0):
    output = "  " * indent + "Map(["
    if len(self.args) > 0:
      output += self.args[0].to_string()
    if len(self.args) > 1:
      for arg in self.args[1:]:
        output += ", " + arg.to_string()
    output += ("], %s, axes=%s)" %
               (self.f.name, str(self.axes)))
    print output

class Reduce(Adverb):
  def __init__(self, args, f, combiner, init, axes=[]):
    self.args = args
    self.f = f
    self.combiner = combiner
    self.init = init
    if len(axes) == 0:
      self.axes = [0 for _ in args]
    else:
      self.axes = axes

  def pretty_print(self, indent=0):
    output = "  " * indent + "Reduce(["
    if len(self.args) > 0:
      output += self.args[0].to_string()
    if len(self.args) > 1:
      for arg in self.args[1:]:
        output += ", " + arg.to_string()
    output += ("], %s, combiner=%s, init=%d, axes=%s)" %
               (self.f.name, self.combiner.name, self.init, str(self.axes)))
    print output

class Function():
  def __init__(self, name, args, stmts, rankchange):
    self.name = name
    self.args = args
    self.stmts = stmts
    self.rankchange = rankchange
  
  def is_scalar(self):
    return len(self.stmts) == 0

  def pretty_print(self, indent=0):
    output = ("  " * indent) + self.name + "("
    if len(self.args) > 0:
      output = output + self.args[0].to_string()
    if len(self.args) > 1:
      for arg in self.args[1:]:
        output = output + ", " + arg.to_string()
    print output + ") {"
    indent += 1
    for stmt in self.stmts:
      if isinstance(stmt, Call):
        print stmt.to_string(indent)
      else:
        stmt.pretty_print(indent)
    indent -= 1
    print "  " * indent + "}"

class Group():
  def __init__(self, var, axis=0):
    self.var = var
    self.axis = axis

  def to_string(self):
    return "Group(" + self.var.name + ", axis=" + str(self.axis) + ")"

class Var():
  def __init__(self, name):
    self.name = name
    self.expanded = False

  def to_string(self):
    return self.name

class Call():
  def __init__(self, f, args):
    self.f = f
    self.args = args
  
  def to_string(self, indent=0):
    ret = self.f.name + "("
    if len(self.args) > 0:
      ret = ret + self.args[0].to_string()
    if len(self.args) > 1:
      for arg in self.args[1:]:
        ret = ret + ", " + arg.to_string()
    return ("  " * indent) + ret + ")"

class Set():
  def __init__(self, var, rhs):
    self.var = var
    self.rhs = rhs
  
  def pretty_print(self, indent=0):
    print ("  " * indent) + self.var.to_string() + " = " + \
        self.rhs.to_string()

class Broadcast():
  def __init__(self, var, rank):
    self.var = var
    self.rank = rank
  
class Tiler():
  def __init__(self):
    self.env = []
  
  def clear(self):
    self.env = []
  
  def add(self, f):
    self.env.append(f)
  
  def pretty_print(self):
    for f in self.env:
      f.pretty_print()

  def tile_adverb(self, adverb, f, top_f):
    if isinstance(adverb, AllPairs):
      adverb.f.args[0].expanded = True
      adverb.f.args[1].expanded = True
      tiled_f = self._tile_function(adverb.f, top_f)
      adverb.f.args[0].expanded = False
      adverb.f.args[0].expanded = False
      return [AllPairs([Group(adverb.args[0], adverb.axes[0]),
                        Group(adverb.args[1], adverb.axes[1])], tiled_f)]
    elif isinstance(adverb, Map):
      for i in range(len(adverb.args)):
        adverb.f.args[i].expanded = True
      tiled_f = self._tile_function(adverb.f, top_f)
      for i in range(len(adverb.args)):
        adverb.f.args[i].expanded = False
      return [Map([Group(arg, axis)
                   for (arg, axis) in zip(adverb.args, adverb.axes)], tiled_f)]
    elif isinstance(adverb, Reduce):
      expanded = []
      for arg in f.args:
        if arg.expanded:
          for i in range(len(adverb.args)):
            if arg.name == adverb.args[i].name:
              expanded.append(i)
      tiled_partials = Function(f.name + "_Nested", adverb.args,
                                [Map(adverb.args,
                                     self._tile_function(adverb.f, top_f))], 0)
      self.add(tiled_partials)
      func_name = f.name + "_ReducePartials_"
      final_reduce_partials = \
          Function(func_name + "0", [Var("Partials0")],
                   [Reduce([Var("Partials0")],
                            adverb.combiner, adverb.combiner,
                            adverb.init, adverb.axes)], adverb.f.rankchange)
      self.add(final_reduce_partials)
      last = final_reduce_partials
      for i in range(1, len(expanded) + 1):
        map_red_partials = \
            Function(func_name + str(i), [Var("Partials" + str(i))],
                     [Map([Var("Partials" + str(i))], last)],
                     adverb.f.rankchange)
        self.add(map_red_partials)
        last = map_red_partials
      tiled_args = []
      for i in range(len(adverb.args)):
        axis = adverb.axes[i]
        if i in expanded:
          axis += 1
        tiled_args.append(Group(adverb.args[i], axis))
      return [Set(Var(f.name + "_Partials"), Call(tiled_partials, tiled_args)),
              Call(last, [Var(f.name + "_Partials")])]

  def _tile_function(self, f, top_f):
    if f.is_scalar():
      return top_f
    stmts = []
    for stmt in f.stmts:
      if isinstance(stmt, Adverb):
        stmts.extend(self.tile_adverb(stmt, f, top_f))
      else:
        stmts.extend(stmt)
    new_f = Function(f.name + "_tiled_L1", f.args, stmts, f.rankchange)
    self.env.append(new_f)
    return new_f
  
  def tile_function(self, f):
    return self._tile_function(f, f)

def ap4DRedto3D():
  tiler = Tiler()
  oneDplus = Function("1Dplus", [Var("x"), Var("y")], [], 0)
  tiler.add(oneDplus)
  f2DtoScalar = Function("f2DtoScalar", [Var("X2D"), Var("Y2D")], [], -2)
  tiler.add(f2DtoScalar)
  reduce3DfTo1D = Function("reduce3DfTo1D", [Var("X3D"), Var("Y3D")],
                           [Reduce([Var("X3D"), Var("Y3D")], f2DtoScalar,
                                   oneDplus, 0)], -2)
  tiler.add(reduce3DfTo1D)
  ap4DReduceTo3D = Function("ap4DReduceTo3D", [Var("X4D"), Var("Y3D")],
                            [AllPairs([Var("X4D"), Var("Y4D")],
                                      reduce3DfTo1D)], -1)
  tiler.add(ap4DReduceTo3D)
  print "All Pairs Reduce 2 4D matrices to a 3D"
  print "-----------"
  print
  print "Environment before tiling:"
  print "--------------------------"
  tiler.pretty_print()
  tiler.tile_function(ap4DReduceTo3D)
  print
  print "Environment after tiling:"
  print "-------------------------"
  tiler.pretty_print()
  print
  print

def map2DRedto1D():
  tiler = Tiler()
  oneDplus = Function("1Dplus", [Var("x"), Var("y")], [], 0)
  tiler.add(oneDplus)
  redPlus = Function("redPlus", [Var("x")],
                     [Reduce([Var("x")], oneDplus, oneDplus, 0)], -1)
  tiler.add(redPlus)
  mapRedPlus = Function("mapRedPlus", [Var("X")], [Map([Var("X")], redPlus)],
                        0)
  tiler.add(mapRedPlus)
  print "Reduce Rows of 2D matrix"
  print "-----------"
  print
  print "Environment before tiling:"
  print "--------------------------"
  tiler.pretty_print()
  tiler.tile_function(mapRedPlus)
  print
  print "Environment after tiling:"
  print "-------------------------"
  tiler.pretty_print()
  print
  print

def timestable():
  tiler = Tiler()
  oneDtimes = Function("1Dtimes", [Var("x"), Var("y")], [], 0)
  tiler.add(oneDtimes)
  aptimes = Function("aptimes", [Var("X")],
                     [AllPairs([Var("X"), Var("X")], oneDtimes)], 1)
  tiler.add(aptimes)
  print "Times Table"
  print "-----------"
  print
  print "Environment before tiling:"
  print "--------------------------"
  tiler.pretty_print()
  tiler.tile_function(aptimes)
  print
  print "Environment after tiling:"
  print "-------------------------"
  tiler.pretty_print()
  print
  print

timestable()
map2DRedto1D()
ap4DRedto3D()