import collections 
import ctypes

from treelike import NestedBlocks

from .. import names, prims 
from ..analysis import SyntaxVisitor
from ..syntax import Var, Const, TypedFn 
from ..ndtypes import (TupleT, ScalarT, ArrayT, ClosureT, 
                       elt_type, FloatT, IntT, BoolT) 

from boxing import box_scalar, unbox_scalar
from c_types import to_ctype, to_dtype
from compile_util import compile_module 
from config import debug 

CompiledFn = collections.namedtuple("CompiledFn",("c_fn", "src", "name"))

def compile(fn, _compile_cache = {}):
  key = fn.name, fn.copied_by 
  if key in _compile_cache:
    return _compile_cache[key]
  name, src = Translator().visit_fn(fn)
  
  print src 
  
  
  c_fn = compile_module(src, name)
  #fn_ptr = getattr(dll, name)
  #fn_ptr.argtypes = (ctypes.py_object,) * len(fn.input_types)
  #fn_ptr.restype = ctypes.py_object
  
  compiled_fn = CompiledFn(c_fn  = c_fn, src = src, name = name)
  _compile_cache[key]  = compiled_fn
  return compiled_fn


def function_source(fn):
  return compile(fn).src 

def function_name(fn):
  return compile(fn).name 

class Translator(object):
   
  def __init__(self):
    self.blocks = []
    self.name_versions = {}
    self.name_mappings = {}

  
  def visit_expr(self, expr):
    expr_class_name = expr.__class__.__name__
    method_name = "visit_" + expr_class_name
    assert hasattr(self, method_name), "Unsupported expression %s" % expr_class_name  
    result = getattr(self, method_name)(expr)
    assert result is not None, "Compilation method for expression %s return None" % expr_class_name
    return result 
  
      
  def visit_expr_list(self, exprs):
    return [self.visit_expr(e) for e in exprs]
  
  
  def as_pyobj(self, expr):
    """
    Compile the expression and if necessary box it up as a PyObject
    """
    result = self.visit_expr(expr)
    if isinstance(expr.type, ScalarT):
      return box_scalar(result, expr.type)
    else:
      return result
  
  def as_pyobj_list(self, exprs):
    return [self.as_pyobj(expr) for expr in exprs]
  
  def visit_stmt(self, stmt):
    stmt_class_name = stmt.__class__.__name__
    method_name = "visit_" + stmt_class_name
    assert hasattr(self, method_name), "Unsupported statemet %s" % stmt_class_name  
    result = getattr(self, method_name)(stmt)
    assert result is not None, "Compilation method for statement %s return None" % stmt_class_name
    return result 
  
  def push(self):
    self.blocks.append([])
  
  def pop(self):
    stmts = self.blocks.pop()
    return "\n".join("  " + stmt for stmt in stmts)
  
  def indent(self, block_str):
    return block_str.replace("\n", "\n  ")
  
  def append(self, stmt):
    self.blocks[-1].append(stmt)
  
  def printf(self, str):
    self.append('printf("%s\\n");' % str)
    
  def fresh_name(self, prefix):
    prefix = names.original(prefix)
    prefix = prefix.replace(".", "_")
    version = self.name_versions.get(prefix, 1)
    self.name_versions[prefix] = version + 1
    if version == 1:
      return prefix 
    else:
      return "%s_%d" % (prefix, version)
  
  def name(self, ssa_name, overwrite = False):
    """
    Convert from ssa names, which might have large version numbers and contain 
    syntactically invalid characters to valid local C names
    """
    if ssa_name in self.name_mappings and not overwrite:
      return self.name_mappings[ssa_name]
    prefix = names.original(ssa_name)
    prefix = prefix.replace(".", "_")
    name = self.fresh_name(prefix)
    self.name_mappings[ssa_name] = name 
    return name 

  def check_tuple(self, tup):
    self.printf("Checking tuple type for %s" % tup)
    self.append('printf("Tuple OK? %%d\\n", PyTuple_Check(%s));' % tup)
 
  def check_array(self, arr):
    self.append('printf("Checking array type for %s @ %%p\\n", %s);' % (arr, arr,))
    self.append('printf("Array OK? %%d\\n", PyArray_Check(%s));' % (arr,))
 
  def tuple_to_stack_array(self, expr):
    if debug:
      assert expr.type.__class__ is TupleT 
      assert all(t == t0 for t in expr.type.elt_types[1:])
    t0 = expr.type.elt_types[0]
    tup = self.visit_expr(expr)
    if debug: self.check_tuple(tup)
    array_name = self.fresh_name("array_from_tuple")
    n = len(expr.type.elt_types)
    self.append("%s %s[%d];" % (to_ctype(t0), array_name, n))
    for i, elt in enumerate(self.tuple_elts(tup, expr.type.elt_types)):
      self.append("%s[%d] = %s;" % array_name, i, elt )
    return array_name
    
  def array_to_tuple(self, arr, n, elt_t):
    elts = [box_scalar("%s[%d]" % (arr,i), elt_t) for i in xrange(n)]
    elt_str = ", ".join(elts)
    return "PyTuple_Pack(%d, %s)" % (n, elt_str)
    
  def visit_Alloc(self, expr):
    elt_size = expr.elt_type.dtype.itemsize
    elt_t = to_ctype(expr.elt_type)
    nelts = self.visit_expr(expr.count)
    return "(%s*) malloc(%d * %s)" % (elt_t, elt_size, nelts)
  
  def visit_AllocArray(self, expr):
    shape = self.tuple_to_stack_array(expr.shape)
    t = to_dtype(elt_type(expr.type))
    return "PyArray_SimpleNew(%d, %s, %s)" % (expr.type.rank, shape, t)
  
  def visit_Const(self, expr):
    return "%s" % expr.value 
  
  def visit_Var(self, expr):
    return self.name(expr.name)
  
  def visit_Cast(self, expr):
    x = self.visit_expr(expr.value)
    ct = to_ctype(expr.type)
    if isinstance(expr, (Const, Var)):
      return "(%s) %s" % (ct, x)
    else:
      return "((%s) (%s))" % (ct, x)
  
  def mk_tuple(self, elts):
    elt_str = ", ".join(self.as_pyobj_list(elts)) 
    n = len(elts)
    return "PyTuple_Pack(%d, %s)" % (n, elt_str)
    
  def visit_Tuple(self, expr):
    return self.mk_tuple(expr.elts)
  
  def visit_Closure(self, expr):
    return self.mk_tuple(expr.args)
 
  def tuple_elt(self, tup, idx, t):
    if debug: self.check_tuple(tup)
    proj_str = "PyTuple_GetItem(%s, %d)" % (tup, idx)
    if isinstance(t, ScalarT):
      return unbox_scalar(proj_str, t)
    else:
      return proj_str 
  
  def tuple_elts(self, tup, ts):
    result = []
    for i,t in enumerate(ts):
      result.append(self.tuple_elt(tup, i, t))
    return result
  
  def visit_TupleProj(self, expr):
    tup = self.visit_expr(expr.tuple)
    return self.tuple_elt(tup, expr.index, expr.type)
  
  def visit_ClosureElt(self, expr):
    clos = self.visit_expr(expr.closure)
    return self.tuple_elt(clos, expr.index, expr.type)
  
  def visit_Index(self, expr):
    arr = self.visit_expr(expr.value)
    idx = self.visit_expr(expr.index)
    return "%s[%s]" % (arr, idx)
  
  
  def visit_ArrayView(self, expr):
    data = self.visit_expr(expr.data)
    shape = self.visit_expr(expr.shape)
    strides = self.visit_expr(expr.strides)
    count = self.visit_expr(expr.size)
    offset = self.visit_expr(expr.offset)
    buffer_name = self.fresh_name("array_buffer")
    bytes_per_elt = expr.type.elt_type.dtype.itemsize
    self.append("PyObject* %s = PyBuffer_FromReadWriteMemory(%s, %s * %d);" % (buffer_name,  data, count, bytes_per_elt))
    dtype = "PyArray_DescrFromType(%s)" % to_dtype(expr.type.elt_type)
    
    vec_name = self.fresh_name("linear_array")
    #   _members = ['data', 'shape', 'strides', 'offset', 'size']
    self.append("PyObject* %s = PyArray_FromBuffer(%s, %s, %s, %s);" % (vec_name, buffer_name, dtype, count, offset))
    # TODO: Assign PyArray_STRIDES[i] = PyTuple_GetItem(strides, i)
    return "PyArray_Reshape(( PyArrayObject*) %s, %s)" % (vec_name, shape)
    
  def attribute(self, v, attr, t):
    if attr == "data":
      if debug: self.check_array(v)
      return "(%s) PyArray_DATA (%s)" % (to_ctype(t), v)
    elif attr == "shape":
      if debug: self.check_array(v)
      elt_types = t.elt_types
      n = len(elt_types)
      elt_t = elt_types[0]
      assert all(t == elt_t for t in elt_types)
      return self.array_to_tuple("PyArray_DIMS( (PyArrayObject*) %s)" % v, n, elt_t) 
    elif attr == "strides":
      if debug: self.check_array(v)
      elt_types = t.elt_types
      n = len(elt_types)
      elt_t = elt_types[0]
      assert all(t == elt_t for t in elt_types)
      strides_name = self.fresh_name("strides")
      self.printf("Getting strides of %s" % v)
      self.append("npy_intp* %s = PyArray_STRIDES(%s);" % (strides_name, v))
      strides_tuple = self.array_to_tuple(strides_name, n, elt_t)
      self.printf("Turning array to tuple")
      return strides_tuple
    elif attr == 'offset':
      return "0"
    else:
      assert False, "Unsupported attribute %s" % attr 
    
  def visit_Attribute(self, expr):
    attr = expr.name
    v = self.visit_expr(expr.value) 
    return self.attribute(v, attr, expr.type)
    
  def visit_PrimCall(self, expr):
    args = self.visit_expr_list(expr.args)
    p = expr.prim 
    if p == prims.add:
      return "%s + %s" % (args[0], args[1])
    if p == prims.subtract:
      return "%s + %s" % (args[0], args[1])
    elif p == prims.multiply:
      return "%s * %s" % (args[0], args[1])
    elif p == prims.divide:
      return "%s / %s" % (args[0], args[1])
    elif p == prims.abs:
      x  = args[0]
      return "%(x)s ? %(x)s >= 0 : -%(x)s" % {'x': x}
    elif p == prims.bitwise_and:
      return "%s & %s" % (args[0], args[1])
    elif p == prims.bitwise_or:
      return "%s | %s" % (args[0], args[1])
    elif p == prims.bitwise_or:
      return "%s | %s" % (args[0], args[1])
    elif p == prims.equal:
      return "%s == %s" % (args[0], args[1])
    elif p == prims.not_equal:
      return "%s != %s" % (args[0], args[1])
    elif p == prims.greater:
      return "%s > %s" % (args[0], args[1])
    elif p == prims.greater_equal:
      return "%s >= %s" % (args[0], args[1])
    elif p == prims.less:
      return "%s < %s" % (args[0], args[1])
    elif p == prims.less_equal:
      return "%s <= %s" % (args[0], args[1])
    
    else:
      assert False, "Prim not yet implemented: %s" % p
  
  def visit_Select(self, expr):
    cond = self.visit_expr(expr.cond)
    true = self.visit_expr(expr.true_value)
    false = self.visit_expr(expr.false_value)
    return "%s ? %s : %s" % (true, cond, false) 
  def visit_Assign(self, stmt):
    self.append('printf("Running %s\\n");' % stmt)
    
    #assert stmt.lhs.__class__ is Var
    #lhs_name = self.name(stmt.lhs.name)
    #rhs = self.visit_expr(stmt.rhs)
    #return "%s = %s;" % (lhs_name, rhs)
    lhs = self.visit_expr(stmt.lhs)
    rhs = self.visit_expr(stmt.rhs)
    if stmt.lhs.__class__ is Var:
      return "%s %s = %s;" % (to_ctype(stmt.lhs.type), lhs, rhs)
    else:
      return "%s = %s;" % (lhs, rhs)
    
  def visit_merge_left(self, merge):
    if len(merge) == 0:
      return ""
    
    stmts = ["\n"]
    for (name, (left, _)) in merge.iteritems():
      stmts.append("%s %s = %s;"  % (to_ctype(left.type), 
                                     self.name(name), 
                                     self.visit_expr(left)))
    return "\n".join(stmts)
      
  
  def visit_merge_right(self, merge):
    if len(merge) == 0:
      return ""
    stmts = ["\n"]
    for (name, (_, right)) in merge.iteritems():
      stmts.append("%s = %s;"  % (self.name(name), self.visit_expr(right)))
    return "\n".join(stmts)
    
  def visit_ForLoop(self, stmt):
    start = self.visit_expr(stmt.start)
    stop = self.visit_expr(stmt.stop)
    step = self.visit_expr(stmt.step)
    var = self.visit_expr(stmt.var)
    t = to_ctype(stmt.var.type)
    
    body =  self.visit_block(stmt.body) +  self.visit_merge_right(stmt.merge)
    body = self.indent("\n" + body)
    
    s = self.visit_merge_left(stmt.merge)
    s += "\nfor (%(t)s %(var)s = %(start)s; %(var)s < %(stop)s; %(var)s += %(step)s) {%(body)s}"
    return s % locals()
  
  def visit_Return(self, stmt):
    return "return %s;" % self.as_pyobj(stmt.value)
  
  def visit_block(self, stmts, push = True):
    if push: self.push()
    for stmt in stmts:
      s = self.visit_stmt(stmt)
      self.append(s)
    self.append("\n")
    return self.pop()
      
  def visit_TypedFn(self, expr):
    return function_name(expr)

  def visit_UntypedFn(self, expr):
    assert False, "Unexpected UntypedFn %s in C backend, should have been specialized" % expr.name
  
  def check_type(self, v, t):
    if isinstance(t, (ClosureT, TupleT)):
      self.printf("Checking tuple for %s" % v)
      self.append('printf("Is %s a tuple? %%d\\n", PyTuple_Check(%s));' % (v,v))
    elif isinstance(t, IntT):
      self.printf("Checking int for %s" % v)
      self.append('printf("Is %s an int? %%d\\n", PyInt_Check(%s));' % (v,v))
        
  def visit_fn(self, fn):
    c_fn_name = self.fresh_name(fn.name)
    dummy = self.fresh_name("dummy")
    args = self.fresh_name("args")
    
    c_args = "PyObject* %s, PyObject* %s" % (dummy, args) #", ".join("PyObject* %s" % self.name(n) for n in fn.arg_names)
    
    self.push()
    #self.printf("Eval init")
    #self.append("Py_Initialize();")
    #self.append("PyEval_InitThreads();")
    #self.printf("Create GILstate")
    #self.append("PyGILState_STATE gstate;")
    #self.printf("GILstate ensure")
    #self.append("gstate = PyGILState_Ensure();")
    
    for i, argname in enumerate(fn.arg_names):
      
      c_name = self.name(argname)
      self.append("PyObject* %s = PyTuple_GET_ITEM(%s, %d);" % (c_name, args, i))
      if debug:
        self.check_type(c_name, fn.type_env[argname])
      
      if debug:
        self.printf("Printing arg #%d %s" % (i,c_name))
        self.append('printf("Type: %%s\\n", PyString_AsString(PyObject_Str(PyObject_Type(%s))));' % c_name)
        self.append('printf("Value: %%s\\n", PyString_AsString(PyObject_Str(%s)));' % c_name)
      
      t = fn.type_env[argname]
      if isinstance(t, ScalarT):
        new_name = self.name(argname, overwrite = True)
        self.append("%s %s = %s;" % (to_ctype(t), new_name, unbox_scalar(c_name, t)))
    c_body = self.visit_block(fn.body, push=False)
    c_body = self.indent("\n" + c_body )#+ "\nPyGILState_Release(gstate);")
    fndef = "PyObject* %(c_fn_name)s (%(c_args)s) {%(c_body)s}" % locals()
    return c_fn_name, fndef 

  
    
    
    
    
    