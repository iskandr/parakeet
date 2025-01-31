import multiprocessing 

from .. import prims 
from ..syntax import Expr, Tuple, Assign, Return, Var, PrimCall 
from ..syntax.helpers import get_fn, return_type
from ..ndtypes import ScalarT, TupleT, ArrayT
from ..c_backend import PyModuleCompiler

import config 

class MulticoreCompiler(PyModuleCompiler):
  
  def __init__(self, depth = 0, *args, **kwargs):
    self.depth = depth
    self.seen_parfor = None 
    PyModuleCompiler.__init__(self, *args, **kwargs)
  
  @property 
  def cache_key(self):
    return self.__class__, self.depth > 0
  
  _loop_var_names = ["i","j","k","l","a","b","c","ii","jj","kk","ll","aa","bb","cc"] 
  def loop_vars(self, count, init_value = "0"):
    assert count <= len(self._loop_var_names)
    return [self.fresh_var("int64_t", self._loop_var_names[i], init_value) 
            for i in xrange(count)]
       
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
  
  
  def get_fn_name(self, fn_expr, attributes = [], inline = True):
    return PyModuleCompiler.get_fn_name(self, fn_expr, 
                                        compiler_kwargs = {'depth':self.depth}, 
                                        attributes = attributes, 
                                        inline = inline)
  
  def get_fn_info(self, fn_expr, attributes = [], inline = True):
    """
    Given a function expression, return:
      - the name of its C representation
      - C expressions representing its closure arguments
      - Parakeet input types 
    """

    fn_name = self.get_fn_name(fn_expr, attributes = attributes, inline = inline)
    closure_args = self.get_closure_args(fn_expr)
    root_fn = get_fn(fn_expr)
    input_types = root_fn.input_types 
    return fn_name, closure_args, input_types
    
  
  def build_loop_body(self, fn_expr, loop_vars, target_name = None):
    """
    Inside a loop nest, construct an index tuple, call the given function, 
    and optionally assign its result to the target variable. 
    Returns the string representation of the loop body and the set of
    private variables it uses. 
    """
    fn_name, closure_args, input_types = self.get_fn_info(fn_expr)

    private_vars = [loop_var for loop_var in loop_vars] 
    last_input_type = input_types[-1]
    body = ""
    if isinstance(last_input_type, TupleT):
      # make a tuple out of the loop variables
      
      c_tuple_t = self.to_ctype(last_input_type)
      index_tuple = self.fresh_var(c_tuple_t, "index_tuple")
      private_vars.append(index_tuple)
      for i, loop_var in enumerate(loop_vars):
        body += "\n%s.elt%d = %s;\n" % (index_tuple, i, loop_var)
      combined_args = tuple(closure_args) + (index_tuple,)
      call = "%s(%s)" % (fn_name, ", ".join(combined_args))
    else:
      combined_args = tuple(closure_args) + tuple(loop_vars)
      call = "%s(%s)" % (fn_name, ", ".join(combined_args))
    if target_name:
      body += "%s = %s;\n" % (target_name, call)
    else:
      body += "\n%s;\n" % call
    return body, private_vars
  
  
  def enter_parfor(self):
    self.depth += 1
    if not self.seen_parfor:
      self.seen_parfor = True 
      self.add_compile_flag("-fopenmp")
      self.add_link_flag("-fopenmp")
    
  def exit_parfor(self):
    self.depth -= 1

  def omp_pragma(self, n_loops, private_vars, reduce_op = None, reduce_vars = None):
    if config.collapse_nested_loops:
      omp = "#pragma omp parallel for private(%s) schedule(%s)" % \
        (", ".join(private_vars), config.schedule)
      if n_loops > 1:
        omp += " collapse(%d)" % n_loops
    else:
      omp = "#pragma omp parallel for private(%s) schedule(%s)" % \
          (private_vars[0], config.schedule)
    
    if reduce_op:
      omp += " reduction (%s:%s)" % (reduce_op, ", ".join(reduce_vars))
    return omp 
  
  def visit_ParFor(self, stmt):
    bounds = self.tuple_to_var_list(stmt.bounds)
    n_vars = len(bounds)
    loop_vars = self.loop_vars(n_vars)
    
    self.enter_parfor()
    body, private_vars = self.build_loop_body(stmt.fn, loop_vars)
    loops = self.build_loops(loop_vars, bounds, body)
    self.exit_parfor()
    
    if self.depth == 0:  
      release_gil = "\nPy_BEGIN_ALLOW_THREADS\n"
      acquire_gil = "\nPy_END_ALLOW_THREADS\n" 
      omp = self.omp_pragma(len(loop_vars), private_vars)
      return release_gil + omp + loops + acquire_gil    
    else:
      return loops 
     
  
  
  def get_binop_prim(self, fn):
    """
    If function is a simple binary operator, then return its prim, 
    otherwise return None
    """

    if len(fn.input_types) != 2:
      return None
      
    t1, t2 = fn.input_types
    if t1 != t2:
      return None 
    if not isinstance(t1, ScalarT):
      return None 
    if not isinstance(t2, ScalarT):
      return None 
    body = fn.body

    if len(body) == 2:
      s1 = body[0]
      s2 = body[1]
      if s1.__class__ is not Assign:
        return None
      rhs = s1.rhs
      if rhs.__class__ is not PrimCall:
        return None 
      args = rhs.args 
      if len(args) != 2:
        return None
      x,y = args 
      if x.__class__ is not Var or y.__class__ is not Var:
        return None 
      if x.name != fn.arg_names[0] or y.name != fn.arg_names[1]:
        return None 
      if s2.__class__ is not Return:
        return None 
      if s2.value.__class__ is not Var:
        return None 
      if s2.value != s1.lhs:
        return None
      return s1.rhs.prim
    elif len(body) == 1:
      s = body[0]
      if s.__class__ is not Return:
        return None
      v = s.value 
      if v.__class__ is not PrimCall:
        return None 
      args = v.args 
      if len(args) != 2:
        return None
      x,y = args 
      if x.__class__ is not Var or y.__class__ is not Var:
        return None 
      if x.name != fn.arg_names[0] or y.name != fn.arg_names[1]:
        return None 
      return v.prim 
       
    
      
  def visit_IndexReduce(self, expr):
    """
    For now, just use a sequential implementation for reductions
    """
    bounds = self.tuple_to_var_list(expr.shape)
    n_vars = len(bounds)
    acc = self.fresh_var(expr.type, "acc", self.visit_expr(expr.init))
    # try to get a simple primitive to use as the OpenMP combiner
    # if this isn't possible then run the loops sequentially
    #
    # TODO: instead of falling back on sequential execution 
    # use complex combiners by generating one 
    # accumulator value per thread and then combining those
    combine_prim = self.get_binop_prim(expr.combine)
    if combine_prim is prims.add:
      omp_reduce_op = "+"
    elif combine_prim is prims.multiply:
      omp_reduce_op = "*"
    elif combine_prim is prims.logical_and:
      omp_reduce_op = "&&"
    elif combine_prim is prims.logical_or:
      omp_reduce_op = "||"
    else:
      omp_reduce_op = None 
 
    loop_vars = self.loop_vars(n_vars)
    assert expr.init is not None, "Accumulator required but not given"
    elt = self.fresh_var(return_type(expr.fn), "elt")
    if omp_reduce_op: self.enter_parfor()
    body, private_vars = self.build_loop_body(expr.fn, loop_vars, target_name = elt)
    if omp_reduce_op:
      body += "\n%s = %s %s %s;" % (acc, acc, omp_reduce_op, elt)
    else:
      combine_name, combine_closure_args, _ = self.get_fn_info(expr.combine)
      combine_arg_str = ", ".join(tuple(combine_closure_args) + (acc, elt))
      body += "\n%s = %s(%s);\n" % (acc, combine_name, combine_arg_str)
    loops = self.build_loops(loop_vars, bounds, body)
    if omp_reduce_op: 
      self.exit_parfor()

    if omp_reduce_op and self.depth == 0:
      release_gil = "\nPy_BEGIN_ALLOW_THREADS\n"
      acquire_gil = "\nPy_END_ALLOW_THREADS\n" 
      omp = self.omp_pragma(len(loop_vars), private_vars, 
                            reduce_op = omp_reduce_op, 
                            reduce_vars = [acc])
      loops = release_gil + omp + loops + acquire_gil    
    self.append(loops)
    return acc 
    
  def visit_IndexScan(self, expr):
    """
    For now, just use a sequential implementation for scans
    """
    assert isinstance(expr.type, ArrayT), "Expected output of Scan to be an array"
    
    bounds = self.tuple_to_var_list(expr.shape)
    n_vars = len(bounds)
    
    combine_name, combine_closure_args, _ = self.get_fn_info(expr.combine)
    loop_vars = self.loop_vars(n_vars)
    
    
    result = self.alloc_array(expr.type, expr.shape)
    
    assert expr.init is not None, "Accumulator required but not given"
    
    elt_t = return_type(expr.fn) 
    assert isinstance(elt_t, ScalarT), "Scans of non-scalar values (%s) not yet implemented" % elt_t
    elt = self.fresh_var(elt_t, "elt")
    body, _ = self.build_loop_body(expr.fn, loop_vars, target_name = elt)
    acc = self.fresh_var(expr.init.type, "acc", self.visit_expr(expr.init))
    combine_arg_str = ", ".join(tuple(combine_closure_args) + (acc, elt))
    body += "\n%s = %s(%s);\n" % (acc, combine_name, combine_arg_str)
    emit_name, emit_closure_args, _ = self.get_fn_info(expr.emit)
    body += "\n"
    emit_args = tuple(emit_closure_args) + (acc,)
    emit_args_str = ", ".join(emit_args)
    body += self.setidx(result, 
                        loop_vars, 
                        "%s(%s)" % (emit_name, emit_args_str), 
                        full_array = True, 
                        return_stmt = True)
    self.append(self.build_loops(loop_vars, bounds, body))
    return result
    
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
  
