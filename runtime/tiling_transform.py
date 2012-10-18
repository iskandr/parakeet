#!/usr/bin/python

import copy, sys

class Adverb():
  pass

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

class AllPairs(Adverb):
  def __init__(self, args, f, axes=[0,0]):
    self.f = f
    self.args = args
    self.axes = axes

  def pretty_print(self, indent=0):
    print "  " * indent + ("AllPairs([%s, %s], %s, axes=[%d,%d])" %
        (self.args[0].to_string(), self.args[1].to_string(), self.f.name,
         self.axes[0], self.axes[1]))

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
  def __init__(self, name, args, stmts):
    self.name = name
    self.args = args
    self.stmts = stmts

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

class TiledMap(Map):
  def __init__(self, args, f, axes=[], group_axes=[]):
    if len(group_axes) == 0:
      self.group_axes = [0 for _ in args]
    else:
      self.group_axes = group_axes
    Map.__init__(self, args, f, axes)

  def pretty_print(self, indent=0):
    output = "  " * indent + "TiledMap(["
    if len(self.args) > 0:
      output += self.args[0].to_string()
    if len(self.args) > 1:
      for arg in self.args[1:]:
        output += ", " + arg.to_string()
    output += ("], %s, axes=%s, group_axes=%s)" %
               (self.f.name, str(self.axes), str(self.group_axes)))
    print output

class TiledAllPairs(AllPairs):
  def __init__(self, args, f, axes=[0,0], group_axes=[0,0]):
    self.group_axes = group_axes
    AllPairs.__init__(self, args, f, axes)

  def pretty_print(self, indent=0):
    print "  " * indent + \
        ("TiledAllPairs([%s, %s], %s, axes=[%d,%d], group_axes=[%d,%d])" %
        (self.args[0].to_string(), self.args[1].to_string(), self.f.name,
         self.axes[0], self.axes[1], self.group_axes[0], self.group_axes[1]))

class TiledReduce(Reduce):
  def __init__(self, args, f, combiner, init, axes=[], group_axes=[]):
    if len(group_axes) == 0:
      self.group_axes = [0 for _ in args]
    else:
      self.group_axes = group_axes
    Reduce.__init__(self, args, f, combiner, init, axes)

  def pretty_print(self, indent=0):
    output = "  " * indent + "TiledReduce(["
    if len(self.args) > 0:
      output += self.args[0].to_string()
    if len(self.args) > 1:
      for arg in self.args[1:]:
        output += ", " + arg.to_string()
    output += ("], %s, combiner=%s, init=%d, axes=%s, group_axes=%s)" %
               (self.f.name, self.combiner.name, self.init, str(self.axes),
                str(self.group_axes)))
    print output

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

class ScalarOp():
  def __init__(self, name):
    self.name = name

  def pretty_print(self, indent=0):
    print ("  " * indent) + self.name

class Broadcast():
  def __init__(self, var, rank):
    self.var = var
    self.rank = rank

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
    new_f = Function(f.name + "_tiled_L1", f.args, tiled_stmts)
    self.env.append(new_f)
    return new_f

  def tile_function(self, f):
    if f.is_scalar():
      return f

    nest_f = copy.deepcopy(f)
    running_nest = [nest_f]
    nest_f.stmts = []
    self._tile_function(f, running_nest)

def ap4DRedto3D():
  tiler = Tiler()
  scalarplus = Function("scalarplus", [Var("x"), Var("y")], [])
  tiler.add(scalarplus)
  f2DtoScalar = Function("f2DtoScalar", [Var("X2D"), Var("Y2D")], [])
  tiler.add(f2DtoScalar)
  reduce3DfTo1D = Function("reduce3DfTo1D", [Var("X3D"), Var("Y3D")],
                           [Reduce([Var("X3D"), Var("Y3D")], f2DtoScalar,
                                   scalarplus, 0)])
  tiler.add(reduce3DfTo1D)
  ap4DReduceTo3D = Function("ap4DReduceTo3D", [Var("X4D"), Var("Y3D")],
                            [AllPairs([Var("X4D"), Var("Y4D")],
                                      reduce3DfTo1D)])
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

def mm():
  tiler = Tiler()
  madd = Function("madd", [Var("x"), Var("y")], [])
  tiler.add(madd)
  dot = Function("Dot", [Var("x"), Var("y")],
                 [ScalarOp("Foo()"),
                  Reduce([Var("x"), Var("y")], madd, madd, 0),
                  ScalarOp("Bar()")])
  tiler.add(dot)
  gemm = Function("APDot", [Var("X"), Var("Y")],
                  [AllPairs([Var("X"), Var("Y")], dot)])
  tiler.add(gemm)
  print "2D Gemm"
  print "-----------"
  print
  print "Environment before tiling:"
  print "--------------------------"
  tiler.pretty_print()
  tiler.tile_function(gemm)
  print
  print "Environment after tiling:"
  print "-------------------------"
  tiler.pretty_print()
  print
  print

def timestable():
  tiler = Tiler()
  scalartimes = Function("scalartimes", [Var("x"), Var("y")], [])
  tiler.add(scalartimes)
  aptimes = Function("aptimes", [Var("X")],
                     [AllPairs([Var("X"), Var("X")], scalartimes)])
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

mm()