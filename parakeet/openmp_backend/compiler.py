import multiprocessing 

from ..syntax import Expr, Tuple
from ..syntax.helpers import get_fn 
from ..ndtypes import ScalarT, TupleT
from ..c_backend import PyModuleCompiler, FlatFnCompiler

class MulticoreCompiler(PyModuleCompiler):
  
  def __init__(self, parfor_depth = 0, *args, **kwargs):
    self.parfor_depth = parfor_depth
    PyModuleCompiler.__init__(self, *args, **kwargs)
    self.add_compile_flag("-fopenmp")
    self.add_link_flag("-fopenmp")

  
  def visit_NumCores(self, expr):
    # by default we're running sequentially 
    return "%d" % multiprocessing.cpu_count()
  
  def tuple_to_var_list(self, expr):
    assert isinstance(expr, Expr)
    if isinstance(expr, Tuple):
      return self.visit_expr_list(expr)
    elif isinstance(expr.type, TupleT):
      c_val = self.visit_expr(expr)
      return ["%s.elt%d" % (c_val, i) for i in xrange(len(expr.type.elt_types))]
    else:
      assert isinstance(expr.type, ScalarT), "Unexpected expr %s : %s" % (expr, expr.type)
      return [self.visit_expr(expr)]
  
  
  def visit_ParFor(self, stmt):
    
    bounds = self.tuple_to_var_list(stmt.bounds)
    loop_var_names = ["i", "j", "k", "l", "ii", "jj", "kk", "ll"]
    n_vars = len(bounds)
    assert n_vars <= len(loop_var_names)
    loop_vars = [self.fresh_var("int64_t", loop_var_names[i]) for i in xrange(n_vars)]
    
    self.parfor_depth += 1  
    
    fn_name = self.get_fn(stmt.fn, compiler_kwargs = {'parfor_depth':self.parfor_depth})
    closure_args = self.get_closure_args(stmt.fn)
    private_vars = [loop_var for loop_var in loop_vars] 
    fn = get_fn(stmt.fn)
    last_input_type = fn.input_types[-1]
    if isinstance(last_input_type, TupleT):
      # make a tupke out of the loop variables
      
      c_tuple_t = self.to_ctype(last_input_type)
      index_tuple = self.fresh_var(c_tuple_t, "index_tuple")
      private_vars.append(index_tuple)
      body = ""
      for i, loop_var in enumerate(loop_vars):
        body += "\n%s.elt%d = %s;\n" % (index_tuple, i, loop_var)
      combined_args = tuple(closure_args) + (index_tuple,)
      body += "%s(%s);\n" % (fn_name, ", ".join(combined_args))
    else:
      
      combined_args = tuple(closure_args) + tuple(loop_vars)
      body = "%s(%s);\n" % (fn_name, ", ".join(combined_args))
 
    loops = self.build_loops(loop_vars, bounds, body)
    self.parfor_depth -= 1 
    if self.parfor_depth == 0:  
      release_gil = "\nPy_BEGIN_ALLOW_THREADS\n"
      acquire_gil = "\nPy_END_ALLOW_THREADS\n"  
      omp = "#pragma omp parallel for collapse(%d) private(%s)" % \
        (len(bounds), ", ".join(private_vars))
      return release_gil + omp + loops + acquire_gil    
    else:
      return loops 
     
  def visit_IndexReduce(self, expr):
    assert False, "IndexReduce needs impl" 
  
  def visit_IndexScan(self, expr):
    assert False, "IndexScan needs impl"
  
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
  
