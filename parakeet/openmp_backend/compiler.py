from ..c_backend import PyModuleCompiler, FlatFnCompiler

class MulticoreFlatFnCompiler(FlatFnCompiler):
  def visit_ParFor(self, stmt):
    bounds = self.tuple_to_var_list(stmt.bounds)
    loop_var_names = ["i", "j", "k", "l", "ii", "jj", "kk", "ll"]
    n_vars = len(bounds)
    assert n_vars <= len(loop_var_names)
    loop_vars = [self.fresh_var("int64_t", loop_var_names[i]) for i in xrange(n_vars)]
    
    omp = "#pragma omp parallel for collapse(%d)" % len(bounds) 
    fn_name = self.get_fn(stmt.fn)
    closure_args = self.get_closure_args(stmt.fn)
    combined_args = tuple(closure_args) + tuple(loop_vars)
    arg_str = ", ".join(combined_args)
    body = "%s(%s);" % (fn_name, arg_str)    
    return omp + self.build_loops(loop_vars, bounds, body) 
  
  def visit_IndexReduce(self, expr):
    pass 
  
  def visit_IndexScan(self, expr):
    pass
  
  def visit_Map(self, expr):
    assert False, "Map should have been lowered into ParFor by now: %s" % expr 
  
  def visit_OuterMap(self, expr):
    assert False, "OuterMap should have been lowered into ParFor by now: %s" % expr 
  
  def visit_Reduce(self, expr):
    assert False, "Reduce should have been lowered into ParFor by now: %s" % expr 
  
  def visit_Scan(self, expr):
    assert False, "Scan should have been lowered into ParFor by now: %s" % expr
    
  def visit_IndexMap(self, expr):
    assert False, "IndexMap should have been lowered into ParFor by now: %s" % expr 
  

class MulticoreModuleCompiler(PyModuleCompiler, MulticoreFlatFnCompiler):
  pass


from ..analysis import contains_adverbs
from .. import c_backend 
def compile_entry(fn):
  
  return c_backend.compile_entry(fn, compiler_class = MulticoreModuleCompiler)


