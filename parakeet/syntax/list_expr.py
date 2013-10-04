from seq_expr import SeqExpr 

class List(SeqExpr):
  _members = ['elts']
  
  def children(self):
    return self.elts 
  