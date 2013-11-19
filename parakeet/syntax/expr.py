from dsltools import Node 

from .. ndtypes import NoneT

class Expr(object):
  # _members = ['type', 'source_info']
  
    
  @classmethod
  def node_type(cls):
    return cls.__name__
  
  def children(self):
    for v in self.itervalues():
      if v and isinstance(v, Expr):
        yield v
      elif isinstance(v, (list,tuple)):
        for child in v:
          if isinstance(child, Expr):
            yield child

  def short_str(self):
    return str(self)

  def __eq__(self, other):
    return (self.__class__ is other.__class__) and \
      self.type == other.type and \
      all(c1 == c2 for (c1,c2) in zip(self.children(), other.children()))
    
  def __ne__(self, other):
    return not (self == other)
  
  def hash(self):
    elts = tuple(self.children())
    return hash(elts)
   
class Const(Expr):
  def __init__(self, value, type = None, source_info = None):
    self.value = value 
    self.type = type 
    self.source_info = source_info 
  
  def children(self):
    return ()

  def short_str(self):
    return str(self.value)

  def __repr__(self):
    if self.type and not isinstance(self.type, NoneT):
      return "%s : %s" % (self.value, self.type)
    else:
      return str(self.value)

  def __str__(self):
    return repr(self)

  def __hash__(self):
    return hash(self.value)

  def __eq__(self, other):
    return other.__class__ is Const and \
           self.value == other.value and \
           self.type == other.type
    
  def __ne__(self, other):
    return other.__class__ is not Const or \
           self.value != other.value or \
           self.type != other.type

class Var(Expr):
  def __init__(self, name, type = None, source_info = None):
    assert name is not None 
    self.name = name
    self.type = type 
    self.source_info = source_info 

  # it's possible to speed things up a little
  # by by-passing the Node construct 
  #_members = ['name']
  #def node_init(self):
  #  assert self.name is not None
    
  def short_str(self):
    return self.name

  def __repr__(self):
    if hasattr(self, 'type'):
      return "%s : %s" % (self.name, self.type)
    else:
      return "%s" % self.name

  def __str__(self):
    return self.name

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other):
    return other.__class__  is Var and \
           self.name == other.name and \
           self.type == other.type


  def __ne__(self, other):
    if other.__class__ is not Var:
      return True 
    if self.name != other.name:
      return True
    return self.type != other.type 
  
  def children(self):
    return ()

class Attribute(Expr):
  def __init__(self, value, name, type = None, source_info = None):
    self.value = value 
    self.name = name 
    self.type = type 
    self.source_info = source_info 

  def children(self):
    yield self.value

  def __str__(self):
    return "attr(%s, '%s')" % (self.value, self.name)

  def __hash__(self):
    return hash ((self.value, self.name))

  def __eq__(self, other):
    return other.__class__ is Attribute and \
           self.name == other.name and \
           self.value == other.value


class Closure(Expr):
  """Create a closure which points to a global fn with a list of partial args"""
  def __init__(self, fn, args, type = None, source_info = None):
    self.fn = fn 
    self.args = args 
    self.type = type 
    self.source_info = source_info 

  def __str__(self):
    fn_str = str(self.fn) #self.fn.name if hasattr(self.fn, 'name') else str(self.fn)
    args_str = ",".join(str(arg) + ":" + str(arg.type) for arg in self.args)
    return "Closure(%s, fixed_args = {%s})" % (fn_str, args_str)

  def node_init(self):
    self.args = tuple(self.args)
    #self.return_type = self.fn.return_type
    #self.input_types = self.fn.input_types[len(self.args):]

  def children(self):
    if isinstance(self.fn, Expr):
      yield self.fn
    for arg in self.args:
      yield arg

  def __hash__(self):
    return hash((self.fn, tuple(self.args)))

class Call(Expr):
  def __init__(self, fn, args, type = None, source_info = None):
    self.fn = fn 
    self.args = args 
    self.type = type 
    self.source_info = source_info 
    
  def __str__(self):
    #if isinstance(self.fn, (Fn, TypedFn)):
    #  fn_name = self.fn.name
    #else:
    fn_name = str(self.fn)
    if self.args.__class__.__name__ == 'ActualArgs':
      arg_str = str(self.args)
    else:
      arg_str = ", ".join(str(arg) for arg in self.args)
    return "%s(%s)" % (fn_name, arg_str)

  def __repr__(self):
    return str(self)

  def children(self):
    yield self.fn
    for arg in self.args:
      yield arg

  def __hash__(self):
    return hash((self.fn, tuple(self.args)))


class PrimCall(Expr):
  """
  Call a primitive function, the "prim" field should be a prims.Prim object
  """
  def __init__(self, prim, args, type = None, source_info = None):
    self.prim = prim 
    self.args = args 
    self.type = type 
    self.source_info = source_info 
    
  #_members = ['prim', 'args']
  
  def _arg_str(self, i):
    arg = self.args[i]
    if arg.__class__ is PrimCall:
      return "(%s)" % arg
    else:
      return str(arg)
  
  def __hash__(self):
    self.args = tuple(self.args)
    return hash((self.prim, self.args))
    
  def __eq__(self, other):
    if other.__class__ is not PrimCall:
      return False 
    
    if other.prim != self.prim:
      return False
    
    my_args = self.args
    other_args = other.args
    n = len(my_args)
    if n != len(other_args):
      return False
    
    for (i,arg) in enumerate(my_args):
      other_arg = other_args[i]
      if arg != other_arg:
        return False
    return True
  
  def __ne__(self, other):
    return not self==other
  
  def __repr__(self):
    if self.prim.symbol:
      if len(self.args) == 1:
        return "%s%s" % (self.prim.symbol, self._arg_str(0))
      else:
        assert len(self.args) == 2
        return "%s %s %s" % (self._arg_str(0), self.prim.symbol, self._arg_str(1))
    else:
      arg_strings = [self._arg_str(i) for i in xrange(len(self.args))]
      combined = ", ".join(arg_strings)
      return "prim<%s>(%s)" % (self.prim.name, combined)

  def __str__(self):
    return repr(self)

  def node_init(self):
    self.args = tuple(self.args)

  def children(self):
    return self.args
  


class ClosureElt(Expr):
  def __init__(self, closure, index, type = None, source_info = None):
    self.closure = closure 
    self.index = index 
    self.type = type 
    self.source_info = source_info 
    
  def __str__(self):
    return "ClosureElt(%s, %d)" % (self.closure, self.index)

  def children(self):
    return (self.closure,)

  def __hash__(self):
    return hash((self.closure, self.index))

class Cast(Expr):
  def __init__(self, value, type, source_info = None):
    self.value = value 
    self.type = type 
    self.source_info = source_info 

  def children(self):
    yield self.value 

  def __hash__(self):
    return hash(self.value)
  
  def __str__(self):
    return "Cast(%s : %s)" % (self.value, self.type) 

class Select(Expr):
  def __init__(self, cond, true_value, false_value, type = None, source_info = None):
    self.cond = cond 
    self.true_value = true_value 
    self.false_value = false_value 
    self.type = type 
    self.source_info = source_info 
    
  # _members = ['cond', 'true_value', 'false_value']
  
  def __hash__(self):
    return hash((self.cond, self.true_value, self.false_value))
  
  def __str__(self):
    return "Select(%s, %s, %s)" % (self.cond, self.true_value, self.false_value)
  
  def children(self):
    yield self.cond 
    yield self.true_value 
    yield self.false_value 
  
  
