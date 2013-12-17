from itertools import izip 
from ..syntax import If, PrimCall, Var, Assign, AllocArray
from transform import Transform

class ExtractParallelLoops(Transform):
  
  def __init__(self):
    Transform.__init__(self)
    
  def extract_map(self, alloc_stmt, loop_stmt):
    pass 
  
  def extract_parfor(self, stmt):
    pass 
  
  def extract_reduce(self, stmt):
    last_stmt = stmt.body[-1]
    # Check if the accumulator variables 
    # get conditionally updated based on 
    # a comparison with some acc vars  
    if last_stmt.__class__ is not If:
      return stmt
    
    cond = stmt.cond
    if cond.__class__ is Var:
      pass 

    other_stmts = stmt.body[:-1]
    
  def transform_block(self, stmts):
    self.last_stmt = None 
    return Transform.transform_block(stmts)
  
  def transform_stmt(self, stmt):
    result = Transform.transform_stmt(stmt)
    self.last_stmt = result 
    return result
   
  def transform_ForLoop(self, stmt):
    #
    # Collection of heuristic patterns
    # of loops which can be lifted into 
    # higher order array operators  
    #
    
    if len(stmt.body) == 0: 
      # why is an empty loop even still here? 
      return stmt 
    
    phi_nodes = stmt.merge
    
    # if there aren't any values being 
    # accumulated between loop iterations
    # then this can't be a Reduction 
    if len(phi_nodes) == 0:
      # currently only the simplest case gets lifted into a map
      # if we allocate the result array immediately before using it
      # in a simple loop 
      if self.last_stmt and \
         self.last_stmt.__class__ is Assign and \
         self.last_stmt.rhs.__class__ is AllocArray:
        map_expr = self.extract_map(self.last_stmt, stmt)
        if map_expr is not None:
          new_stmt = Assign(self.last_stmt.lhs, map_expr)
          # can't delete the last stmt since it's already in the AST 
          # so just make it a null assignment () = ()
          self.last_stmt.lhs = self.tuple([])
          self.last_stmt.rhs = self.tuple([])
          return new_stmt 
        
      # if nothing is being accumulated then try to infer a simple map
      parfor = self.extract_parfor(stmt)
      if parfor is None: return stmt 
      else: return parfor 
    else:
      reduce_expr = self.extract_reduce(stmt)
      if reduce_expr is None: 
        return stmt 
      else:
        phi_var_names = phi_nodes.keys()
        phi_var_types = [left_val.type for (left_val, _) in phi_nodes.itervalues()]
        phi_vars = [Var(name = name, type = t) 
                    for name, t in 
                    izip(phi_var_names, phi_var_types)]
        lhs_tuple = self.tuple(phi_vars)
        return self.assign(lhs_tuple, reduce_expr)    
    
