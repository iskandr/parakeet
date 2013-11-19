from expr import Expr 

class SeqExpr(Expr):
  def __init__(self, value, type = None, source_info = None):
    self.value = value 
    self.type = type 
    self.source_info = source_info 

  def children(self):
    yield self.value 

class Enumerate(SeqExpr):
  pass 
  
class Zip(SeqExpr):
  def __init__(self, values, type = None, source_info = None):
    self.values = tuple(values) 
    self.type = type 
    self.source_info = source_info
    
  def children(self):
    return self.values

class Len(SeqExpr):
  pass 
  
class Index(SeqExpr):
  """
  TODO: 
    - make all user-defined indexing check_negative=True by default 
    - implement backend logic for lowering check_negative 
  """
  def __init__(self, value, index, check_negative = None, type = None, source_info = None):
    self.value = value 
    self.index = index 
    self.check_negative = check_negative 
    self.type = type 
    self.source_info = source_info
  
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
  
