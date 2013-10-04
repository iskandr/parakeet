from expr import Expr 

class SeqExpr(Expr):
  pass 

class Enumerate(SeqExpr):
  _members = ['value']
  
  def children(self):
    yield self.value 
    
class Zip(SeqExpr):
  _members = ['values']
  
  def children(self):
    yield self.values

class Len(SeqExpr):
  _members = ['value']
  
class Index(SeqExpr):
  """
  TODO: 
    - make all user-defined indexing check_negative=True by default 
    - implement backend logic for lowering check_negative 
  """
  _members = ['value', 'index', 'check_negative']
  
  def __eq__(self, other):
    return other.__class__ is Index and \
           other.value == self.value and \
           other.index == self.index
  
  def __hash__(self):
    return hash((self.value, self.index))
  
  def children(self):
    yield self.value
    yield self.index

  def __str__(self):
    return "%s[%s]" % (self.value, self.index)
  
