from seq_expr import SeqExpr 

class List(SeqExpr):
  def __init__(self, elts, type = None, source_info = None):
    self.elts = tuple(elts)
    self.type = type 
    self.source_info = source_info
  
  def children(self):
    return self.elts 
  