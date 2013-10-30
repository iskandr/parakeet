from .. syntax import Assign, Index, Var

from loop_transform import LoopTransform 


class ScalarReplacement(LoopTransform):
  """
  When a loop reads and writes to a non-varying memory location, 
  we can keep the value of that location in a register and write 
  it to the heap after the loop completes. 
  
  Transform code like this:
      for i in low .. high: 
        z = x[const]
        q = z ** 2
        x[const] = q + i 
  into 
      z_in = x[const]
      for i in low .. high:
        (header)
          z_loop = phi(z_in, z_out)
        (body) 
          q = z_loop ** 2
          z_out = q + i 
      x[const] = z_loop   
  """
  
  
  
  def preload_reads(self, reads):
    scalar_vars = {}
    for (array_name, index_set) in reads.iteritems():
      t = self.type_env[array_name]
      for index_expr in index_set:
        scalar = self.index(Var(array_name, type = t), index_expr, 
                            temp = True, name = "scalar_repl_input")
       
        scalar_vars[(array_name, index_expr)] = scalar
    return scalar_vars

  def replace_indexing(self, loop_body, loop_scalars):
    """
    Given a map from (array_name, index_expr) pairs to 
    scalar variables, replace reads/writes with scalars
    """
    final_scalars = loop_scalars.copy()
    for stmt in loop_body:
      if stmt.__class__ is not Assign: 
        continue 
      if stmt.rhs.__class__ is Index:
        key = stmt.rhs.value.name, stmt.rhs.index
        if key in final_scalars:
          stmt.rhs = final_scalars[key]  
      if stmt.lhs.__class__ is Index:
        key = stmt.lhs.value.name, stmt.lhs.index
        if key in final_scalars:
          new_var = self.fresh_var(stmt.lhs.type, "scalar_repl_out")
          stmt.lhs = new_var
          final_scalars[key] = new_var  
    return final_scalars
  
  
  def transform_ForLoop(self, stmt):
    
    if self.is_simple_block(stmt.body):
      loop_vars = set([stmt.var.name])
      loop_vars.update(stmt.merge.keys())
      # gather all the variables whose values change between loop iters
      self.collect_loop_vars(loop_vars, stmt.body)
      # full set of array reads & writes 
      reads, writes = self.collect_memory_accesses(stmt.body)
      
      # which arrays are written to at loop-dependent locations? 
      unsafe = set([])
      for (name, write_indices) in writes.iteritems():
        if self.any_loop_vars(loop_vars, write_indices):
          unsafe.add(name)
      safe_writes = dict([(k,v) for (k,v) in writes.items() 
                          if k not in unsafe])
      
      safe_reads = dict([(k,v) for (k,v) in reads.items() 
                         if k not in unsafe and not self.any_loop_vars(loop_vars, v)])

      
      safe_locations = {}
      all_keys = set(safe_writes.keys()).union(set(safe_reads.keys()))
      for name in all_keys:
        index_set = safe_reads.get(name, set([])).union(safe_writes.get(name, set([])))
        safe_locations[name] = index_set 
      # move all safe/loop-invariant reads into registers at the top of the loop
      # I'm also including the writes (by passing safe_locations instead of safe_reads)
      # so that they become loop-carried accumulator values, otherwise where else would
      # they get their initial values?
      input_scalars = self.preload_reads(safe_locations)

      
      # need to rename the scalars so they have an SSA variable for the beginning of the loop
      loop_scalars = {}
      for ( (name,index_expr), input_var) in input_scalars.iteritems():
        loop_var = self.fresh_var(input_var.type, "scalar_repl_acc")
        loop_scalars[(name, index_expr)] = loop_var
      
      
      # propagate register names for all writes 
      final_scalars = self.replace_indexing(stmt.body, loop_scalars)
      
      for (key, final_var) in final_scalars.iteritems():
        loop_var = loop_scalars[key]
        input_var = input_scalars[key]
        stmt.merge[loop_var.name] = (input_var, final_var)
      
      self.blocks.append(stmt)
      # write out the results back to memeory 
      for ( (array_name, index_expr), loop_var) in loop_scalars.iteritems():
        array_type = self.type_env[array_name]
        lhs = self.index(Var(array_name, type = array_type), index_expr, temp = False)
        self.assign(lhs, loop_var) 
      return None
    else:     
      stmt.body = self.transform_block(stmt.body)
      return stmt 