from expr import Expr 
from stmt import Stmt 

class Struct(Expr):
  """
  Eventually all non-scalar data should be transformed to be created with this
  syntax node, signifying explicit struct allocation
  """

  def __init__(self, args, type = None, source_info = None):
    self.args = tuple(args)
    self.type = type 
    self.source_info = source_info 
    
  def __str__(self):
    return "Struct(%s) : %s" % \
           (", ".join(str(arg) for arg in self.args), self.type)

  def children(self):
    return self.args

  def __hash__(self):
    self.args = tuple(self.args)
    return hash(self.args)

class Alloc(Expr):
  """Allocates a block of data, returns a pointer"""
  
  def __init__(self, elt_type, count, type = None, source_info = None):
    self.elt_type = elt_type 
    self.count = count 
    self.type = type 
    self.source_info = source_info
  
  def __str__(self):
    return "alloc<%s>[%s] : %s" % (self.elt_type, self.count, self.type)

  def children(self):
    return (self.count,)

  def __hash__(self):
    return hash((self.elt_type, self.count))

class Free(Expr):
  """Free a manually allocated block of memory"""
  def __init__(self, value, type = None, source_info = None):
    self.value = value 
    self.type = type 
    self.source_info = source_info
    
  def __str__(self):
    return "free(%s)" % self.value 
  
  def children(self):
    yield self.value 
    
  def __hash__(self):
    return hash(self.value)
  
class NumCores(Expr):
  
  """
  Degree of available parallelism, 
  varies depending on backend and how
  ParFor is actually being mapped 
  to executing threads/thread blocks/etc..
  """
  
  def __str__(self):
    return "NUM_CORES"
  
  def __eq__(self, other):
    return other.__class__ is NumCores 
  
  def node_init(self):
    from ..ndtypes import Int64
    self.type = Int64 
  
  def __hash__(self):
    return 0
  
  def children(self):
    return ()

class SourceExpr(Expr):
  """
  Splice this code directly into the low-level representation, 
  should only be used from within a backend that knows what
  the target code should look like 
  """
  def __init__(self, text, type = None, source_info = None):
    self.text = text 
    self.type = type 
    self.source_info = source_info 
  
  def __str__(self):
    return "SourceExpr(%s)" % self.text 

class SourceStmt(Stmt):
  """
  Splice this code directly into the low-level representation, 
  should only be used from within a backend that knows what
  the target code should look like 
  """
  def __init__(self, text, type = None, source_info = None):
    self.text = text 
    self.type = type 
    self.source_info = source_info 
    
  def __str__(self):
    return "SourceStmt(%s)" % self.text 
  