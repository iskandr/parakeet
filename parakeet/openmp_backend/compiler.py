from ..c_backend import PyModuleCompiler, FlatFnCompiler

class MulticoreFlatFnCompiler(FlatFnCompiler):
  def visit_ParFor(self, stmt):
    bounds = self.tuple_to_var_list(stmt.bounds)
    loop_var_names = ["i", "j", "k", "l", "ii", "jj", "kk", "ll"]
    n_vars = len(bounds)
    assert n_vars <= len(loop_var_names)
    loop_vars = [self.fresh_var("int64_t", loop_var_names[i]) for i in xrange(n_vars)]
    
    omp = "#pragma omp parallel for" 
    fn_name = self.get_fn(stmt.fn)
    closure_args = self.get_closure_args(stmt.fn)
    combined_args = tuple(closure_args) + tuple(loop_vars)
    arg_str = ", ".join(combined_args)
    body = "%s(%s);" % (fn_name, arg_str)    
    return omp + self.build_loops(loop_vars, bounds, body) 
  
  def visit_IndexMap(self, expr):
    pass
  
  def visit_IndexReduce(self, expr):
    pass 
  
  def visit_IndexScan(self, expr):
    pass 

class MulticoreModuleCompiler(PyModuleCompiler, MulticoreFlatFnCompiler):
  pass  

def compile_entry(fn):
  pass 