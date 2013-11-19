from ..ndtypes import make_tuple_type

from expr import Const 
from seq_expr import SeqExpr

class Tuple(SeqExpr):
  def __init__(self, elts, type = None, source_info = None):
    self.elts = tuple(elts)
    self.type = type 
    self.source_info = source_info 
    
    if self.type is None and all(e.type is not None for e in self.elts):
      self.type = make_tuple_type(tuple(e.type for e in self.elts))
  
  def __iter__(self):
    return iter(self.elts)
  
  def __len__(self):
    return len(self.elts)
  
  def __getitem__(self, idx):
    return self.elts[idx]
  
  def __str__(self):
    if len(self.elts) > 0:
      return ", ".join([str(e) for e in self.elts])
    else:
      return "()"
    
  def children(self):
    return self.elts

  def __hash__(self):
    return hash(tuple(self.elts))


class TupleProj(SeqExpr):
  def __init__(self, tuple, index, type = None, source_info = None):
    self.tuple = tuple 
    self.index = index 
    self.type = type 
    self.source_info = source_info 

    if self.type is None:
      if self.tuple.type is not None and self.index.__class__ is Const:
        self.type = self.tuple.type.elt_types[self.index.value]
            
  def __str__(self):
    return "TupleProj(%s, %d)" % (self.tuple, self.index)

  def children(self):
    return (self.tuple,)

  def __hash__(self):
    return hash((self.tuple, self.index))