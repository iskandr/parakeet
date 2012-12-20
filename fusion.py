
from syntax import Var, Assign  
 
from adverbs import Adverb, Scan, Reduce, Map, AllPairs 

from transform import Transform 
from syntax_visitor import SyntaxVisitor
from use_analysis import use_count
from collect_vars import collect_var_names

def fuse(prev_fn, next_fn):
  assert len()

class Fusion(Transform):
  def __init__(self, fn):
    Transform.__init__(self, fn)
    # name of variable -> Map or Scan adverb  
    self.adverb_bindings = {}
    
    # map each variable to 
    self.use_counts = use_count(fn)
      
  def transform_Assign(self, stmt):
    rhs = stmt.rhs
     
    if stmt.lhs.__class__ is Var and isinstance(rhs, Adverb) and \
        rhs.__class__ is not AllPairs:
      args = rhs.args 
      if len(args) == 1 and args[0].__class__ is Var:
        arg_name = args[0].name 
        if self.use_counts[arg_name] == 1 and arg_name in self.adverb_bindings:
          prev_adverb = self.adverb_bindings[arg_name]
          if prev_adverb.__class__ is Map and rhs.axis == prev_adverb.axis:
            # since we're modifying the RHS of the assignment
            # we better make sure the caller doesn't expect us 
            # to return a fresh copy of the AST 
            assert not self.copy
            rhs.fn = fuse(prev_adverb.fn, rhs.fn)
            rhs.args = prev_adverb.args
           
      lhs_name = stmt.lhs.name    
      self.adverb_bindings[lhs_name] = rhs 
    return stmt 
  
def var_names(expr_list):
  return [e.name for e in expr_list if isinstance(e, syntax.Var)]

