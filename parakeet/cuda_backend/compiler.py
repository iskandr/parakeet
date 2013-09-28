from ..c_backend import PyModuleCompiler, FlatFnCompiler

class CudaFlatFnCompiler(FlatFnCompiler):
  def visit_ParFor(self, stmt):
    pass 
   
  def visit_IndexMap(self, expr):
    pass
  
  def visit_IndexReduce(self, expr):
    pass 
  
  def visit_IndexScan(self, expr):
    pass  
  

class CudaModuleCompiler(PyModuleCompiler, CudaFlatFnCompiler):
  pass  

def compile_entry(fn):
  pass 