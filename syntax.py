from node import Node

class Stmt(Node):
  pass

class Assign(Stmt):
  _members = ['lhs', 'rhs']

class Return(Stmt):
  _members = ['value']

class If(Stmt):
  _members = ['cond', 'true', 'false', 'merge']

class While(Stmt):
  """A loop consists of a header, which runs
     before each iteration, a condition for
     continuing, the body of the loop,
     and optionally (if we're in SSA) merge
     nodes for incoming and outgoing variables
     of the form [(new_var1, (old_var1,old_var2)]
   """
  _members = ['cond', 'body', 'merge_before', 'merge_after']

class Expr(Node):
  _members = ['type']

class Adverb(Expr):
  _members = ['fn', 'args', 'axes']

class Map(Adverb):
  def __init__(self, args, fn, axes=[]):
    self.fn = fn
    self.args = args
    if len(axes) == 0:
      self.axes = [0 for _ in args]
    else:
      self.axes = axes

  def __repr__(self):
    output = "map(["
    if len(self.args) > 0:
      output += str(self.args[0])
    if len(self.args) > 1:
      for arg in self.args[1:]:
        output += ", " + str(arg)
    output += ("], %s, axes=%s)" %
               (self.fn.name, str(self.axes)))
    return output

  def __str__(self):
    return repr(self)

class AllPairs(Adverb):
  def __init__(self, args, fn, axes=[0,0]):
    self.fn = fn
    self.args = args
    self.axes = axes

  def __repr__(self):
    return ("allpairs([%s, %s], %s, axes=[%d,%d])" %
           (self.args[0], self.args[1], self.fn.name,
            self.axes[0], self.axes[1]))

  def __str__(self):
    return repr(self)

class Reduce(Adverb):
  _members = ['combiner', 'init']
  def __init__(self, args, fn, combiner, init, axes=[]):
    self.args = args
    self.fn = fn
    self.combiner = combiner
    self.init = init
    if len(axes) == 0:
      self.axes = [0 for _ in args]
    else:
      self.axes = axes

  def __repr__(self):
    output = "reduce(["
    if len(self.args) > 0:
      output += str(self.args[0])
    if len(self.args) > 1:
      for arg in self.args[1:]:
        output += ", " + str(arg)
    output += ("], %s, combiner=%s, init=%d, axes=%s)" %
               (self.fn.name, self.combiner.name, self.init, str(self.axes)))
    return output

  def __str__(self):
    return repr(self)

class TiledMap(Map):
  def __init__(self, args, fn, axes=[], group_axes=[]):
    if len(group_axes) == 0:
      self.group_axes = [0 for _ in args]
    else:
      self.group_axes = group_axes
    Map.__init__(self, args, fn, axes)

  def __repr__(self):
    output = "tiledmap(["
    if len(self.args) > 0:
      output += str(self.args[0])
    if len(self.args) > 1:
      for arg in self.args[1:]:
        output += ", " + str(arg)
    output += ("], %s, axes=%s, group_axes=%s)" %
               (self.fn.name, str(self.axes), str(self.group_axes)))
    return output

class TiledAllPairs(AllPairs):
  def __init__(self, args, fn, axes=[0,0], group_axes=[0,0]):
    self.group_axes = group_axes
    AllPairs.__init__(self, args, fn, axes)

  def __repr__(self):
    return ("tiledallpairs([%s, %s], %s, axes=[%d,%d], group_axes=[%d,%d])" %
            (str(self.args[0]), str(self.args[1]), self.fn.name,
             self.axes[0], self.axes[1],
             self.group_axes[0], self.group_axes[1]))

  def __str__(self):
    return repr(self)

class TiledReduce(Reduce):
  def __init__(self, args, fn, combiner, init, axes=[], group_axes=[]):
    if len(group_axes) == 0:
      self.group_axes = [0 for _ in args]
    else:
      self.group_axes = group_axes
    Reduce.__init__(self, args, fn, combiner, init, axes)

  def __repr__(self):
    output = "tiledreduce(["
    if len(self.args) > 0:
      output += str(self.args[0])
    if len(self.args) > 1:
      for arg in self.args[1:]:
        output += ", " + str(arg)
    output += ("], %s, combiner=%s, init=%d, axes=%s, group_axes=%s)" %
               (self.fn.name, self.combiner.name, self.init, str(self.axes),
                str(self.group_axes)))
    return output

  def __str__(self):
    return repr(self)

class Const(Expr):
  _members = ['value']

  def __repr__(self):
    return "const(%s : %s)" % (self.value, self.type)

  def __str__(self):
    return repr(self)

class Var(Expr):
  _members = ['name']

  def __repr__(self):
    if hasattr(self, 'type'):
      return "var(%s, type=%s)" % (self.name, self.type)
    else:
      return "var(%s)" % self.name

  def __str__(self):
    return self.name

class Attribute(Expr):
  _members = ['value', 'name']

class Index(Expr):
  _members = ['value', 'index']

class Tuple(Expr):
  _members = ['elts']

class TupleProj(Expr):
  _members = ['tuple', 'index']

class Array(Expr):
  _members = ['elts']

class Alloc(Expr):
  """
  Allocates a block of data, returns a pointer
  """
  _members = ['elt_type', 'count']

class Closure(Expr):
  """
  Create a closure which points to a global fn
  with a list of partial args
  """
  _members = ['fn', 'args']

class Invoke(Expr):
  """
  Invoke a closure with extra args
  """
  _members = ['closure', 'args']

class Slice(Expr):
  _members = ['lower', 'upper', 'step']

class PrimCall(Expr):
  """
  Call a primitive function, the "prim" field should be a
  prims.Prim object
  """
  _members = ['prim', 'args']

  def __repr__(self):
    return "primcall[%s](%s)" % (self.prim, ", ".join(map(str, self.args)))

  def __str__(self):
    return repr(self)

class Fn(Node):
  """
  Function definition
  """
  _members = ['name',  'args', 'body', 'nonlocals']

class Struct(Expr):
  """
  Eventually all non-scalar data should be transformed to be created
  with this syntax node, signifying explicit struct allocation
  """
  _members = ['args']

class Cast(Expr):
  # inherits the member 'type' from Expr, but for Cast nodes it is mandatory
  _members = ['value']

class TypedFn(Node):
  """The body of a TypedFn should contain Expr nodes
  which have been extended with a 'type' attribute
  """
  _members = ['name', 'args', 'body', 'input_types', 'return_type', 'type_env']

class Traversal:
  def visit_block(self, block, *args, **kwds):
    for stmt in block:
      self.visit_stmt(stmt, *args, **kwds)

  def visit_stmt(self, stmt, *args, **kwds):
    method_name = "stmt_" + stmt.__class__.__name__
    method = getattr(self, method_name)
    method(stmt, *args, **kwds)

  def visit_expr(self, expr, *args, **kwds):
    method_name = "expr_" + expr.__class__.__name__
    method = getattr(self, method_name)
    return method(expr, *args, **kwds)
