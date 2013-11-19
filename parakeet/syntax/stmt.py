from expr import Expr 
from dsltools import Node

class Stmt(Node):
  _members = ['source_info']

def block_to_str(stmts):
  body_str = '\n'
  body_str += '\n'.join([str(stmt) for stmt in stmts])

  return body_str.replace('\n', '\n    ')

def phi_nodes_to_str(phi_nodes):
  parts = ["%s <- phi(%s, %s)" %
           (var, left, right) for (var, (left, right)) in phi_nodes.items()]
  whole = "\n" + "\n".join(parts)
  # add tabs
  return whole.replace("\n", "\n    ")

class Assign(Stmt):
  _members = ['lhs', 'rhs']

  def __str__(self):
    if hasattr(self.lhs, 'type') and self.lhs.type:
      return "%s : %s = %s" % (self.lhs, self.lhs.type, self.rhs)
    else:
      return "%s = %s" % (self.lhs, self.rhs)

class ExprStmt(Stmt):
  """Run an expression without binding any new variables"""

  _members = ['value']

  def __str__(self):
    assert self.value is not None
    return "ExprStmt(%s)" % self.value

class Comment(Stmt):
  _members = ['text']

  def __str__(self):
    s = "#"
    for (i, c) in enumerate(self.text):
      if i % 78 == 0:
        s += "\n# "
      s += c
    s += "\n#"
    return s


class Return(Stmt):
  _members = ['value']

  def __str__(self):
    return "Return %s" % self.value

class If(Stmt):
  _members = ['cond', 'true', 'false', 'merge']

  def __str__(self):
    s = "if %s:" % self.cond
    if (len(self.true) + len(self.false)) > 0:
      s += "%s\n" % block_to_str(self.true)
    else:
      s += "\n"
    if len(self.false) > 0:
      s += "else:%s\n" % block_to_str(self.false)
    if len(self.merge) > 0:
      s += "(merge-if)%s" % phi_nodes_to_str(self.merge)
    return s

  def __repr__(self):
    return str(self)

class While(Stmt):
  """
  A loop consists of a header, which runs before each iteration, a condition for
  continuing, the body of the loop, and optionally (if we're in SSA) merge nodes
  for incoming and outgoing variables of the form
  [(new_var1, (old_var1,old_var2)]
  """

  _members = ['cond', 'body', 'merge']

  def __repr__(self):
    s = "while %s:\n  "  % self.cond
    if len(self.merge) > 0:
      s += "(header)%s\n  " % phi_nodes_to_str(self.merge)
    if len(self.body) > 0:
      s +=  "(body)%s" % block_to_str(self.body)
    return s

  def __str__(self):
    return repr(self)

class ForLoop(Stmt):
  """
  Having only one loop construct started to become cumbersome, especially now
  that we're playing with loop optimizations.

  So, here we have the stately and ancient for loop.  All hail its glory.
  """

  _members = ['var', 'start', 'stop', 'step', 'body', 'merge']
    
  def __str__(self):
    s = "for %s in range(%s, %s, %s):" % \
      (self.var,
       self.start.short_str(),
       self.stop.short_str(),
       self.step.short_str())

    if self.merge and len(self.merge) > 0:
      s += "\n  (header)%s\n  (body)" % phi_nodes_to_str(self.merge)
    s += block_to_str(self.body)
    return s
  
  
class ParFor(Stmt):
  _members = ['fn', 'bounds']

  def __str__(self):
    return "ParFor(fn = %s, bounds = %s)" % (self.fn, self.bounds)
  
#  
# Consider using these in the future
# instead of having a structured LHS for Assign
#

class SetIndex(Stmt):
  _members = ['array', 'index', 'value']
  
  def __str__(self):
    return "%s[%s] = %s" % (self.array, self.index, self.value)
  

class SetAttr(Stmt):
  _members = ['struct', 'attr', 'value']

  def node_init(self):
    assert isinstance(self.struct, Expr)
    assert isinstance(self.attr, str)
    assert isinstance(self.value, Expr)
    
  def __str__(self):
    return "%s.%s = %s" % (self.struct, self.attr, self.value)
  
class PrintString(Stmt):
  _members = ['text']
  