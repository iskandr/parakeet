from dependence_graph import DependenceGraph
from transform import Transform 

class LoopFusion(Transform):
  def pre_apply(self, fn):    
    self.graph = DependenceGraph()
    self.graph.visit_fn(fn)
    self.waiting = self.graph.nodes
    self.added = set([])
    self.scopes = []
    
  def post_apply(self, fn):
    if len(self.waiting) != 0:
      print "Statement nodes not added:"
      for stmt_node in self.waiting:
        print " -- ", stmt_node 
      assert False, "Not all statements added back to program!"
    
  def transform_block(self, old_stmts):
    """
    Do an extremely slow and inefficient topological sort 
    (change this later, now just prototyping)
    """
    first_iter = True 
    n_added = 0
    scope = id(old_stmts)
    self.scopes.append(scope)
    new_stmts = []
    while first_iter or n_added > 0:
      n_added = 0
      first_iter = False 
      for node in sorted(list(self.waiting)):
        if node.id not in self.added and \
           node.scope == scope and \
           all(node_id in self.added or node_id == node.id 
               for node_id in node.depends_on):
          self.waiting.remove(node)
          stmt = self.transform_stmt(node.stmt)
          new_stmts.append(stmt)
          self.added.add(node.id)  
          n_added += 1
    self.scopes.pop()
    return new_stmts 
  
