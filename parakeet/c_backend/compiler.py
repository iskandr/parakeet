import collections 
import ctypes

from treelike import NestedBlocks

from .. import names, prims 
from ..analysis import SyntaxVisitor
from ..syntax import Var, Const, TypedFn 
from ..ndtypes import (TupleT, ScalarT, ArrayT, ClosureT, 
                       elt_type, FloatT, IntT, BoolT) 


from c_types import to_ctype, to_dtype
from compile_util import compile_module 
from config import debug, print_function_source, print_module_source 
from reserved_names import is_reserved

CompiledFn = collections.namedtuple("CompiledFn",("c_fn", "module", "filename", "src", "name"))

def compile(fn, _compile_cache = {}):
  key = fn.name, fn.copied_by 
  if key in _compile_cache:
    return _compile_cache[key]
  name, src = Translator().visit_fn(fn)
  if print_function_source:
    print "Generated C source for %s:" %(name, src)
  module = compile_module(src, name, print_source = print_module_source)
  c_fn = getattr(module,name)
  compiled_fn = CompiledFn(c_fn = c_fn, 
                           module = module, 
                           filename= module.__file__,
                           src = src, 
                           name = name)
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
  

  def unbox_scalar(self, x, t):
    assert isinstance(t, ScalarT), "Expected scalar type, got %s" % t
    temp = self.fresh_name("scalar_temp")
    self.append("%s %s;" % (to_ctype(t), temp))
    self.append("PyArray_ScalarAsCtype(%s, &%s);" % (x, temp))
    return temp 
      
  def box_scalar(self, x, t):
    scalar = self.fresh_name("scalar");
    self.append("%s %s = %s;" % (to_ctype(t), scalar, x))
    return "PyArray_Scalar(&%s, PyArray_DescrFromType(%s), NULL)" % (scalar, to_dtype(t) )
    
  def as_pyobj(self, expr):
    """
    Compile the expression and if necessary box it up as a PyObject
    """
    x = self.visit_expr(expr)
    if isinstance(expr.type, ScalarT):
      return self.box_scalar(x, expr.type)
    else:
      return x
  
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
  
  def printf(self, s):
    self.append('printf("%s\\n");' % s)
  
  def c_str(self, obj):
    return "PyString_AsString(PyObject_Str(%s))" % obj
  
  def c_type_str(self, obj):
    return self.c_str("PyObject_Type(%s)" % obj)
   
  def print_pyobj(self, obj, text = ""):
    text = '"%s"' % text
    self.append('printf("%%s%%s\\n", %s, %s);' % (text, self.c_str(obj)))
  
  def print_pyobj_type(self, obj, text=""):
    text = '"%s"' % text
    self.append('printf("%%s%%s\\n", %s, %s);' % (text, self.c_type_str(obj)))
    
    
  def fresh_name(self, prefix):
    prefix = names.original(prefix)
    prefix = prefix.replace(".", "_")
    version = self.name_versions.get(prefix, 1)
    self.name_versions[prefix] = version + 1
    if version == 1 and not is_reserved(prefix):
      return prefix 
    else:
      return "%s_%d" % (prefix, version)
  
  def fresh_var(self, t, prefix = None, init = None):
    if prefix is None:
      prefix = "temp"
    name = self.fresh_name(prefix)
    if isinstance(t, str):
      t_str = t
    else:
      t_str = to_ctype(t)
    if init is None:
      self.append("%s %s;" % (t_str, name))
    else:
      self.append("%s %s = %s;" % (t_str, name, init))
    return name
  
  def assign(self, name, rhs):
    self.append("%s = %s;" % (name, rhs))
  
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

  

 
 
  def tuple_to_stack_array(self, expr):
    t0 = expr.type.elt_types[0]
    
    if debug:
      assert expr.type.__class__ is TupleT 
      assert all(t == t0 for t in expr.type.elt_types[1:])
      
    tup = self.visit_expr(expr)
    if debug: self.check_tuple(tup)
    array_name = self.fresh_name("array_from_tuple")
    n = len(expr.type.elt_types)
    self.append("%s %s[%d];" % (to_ctype(t0), array_name, n))
    for i, elt in enumerate(self.tuple_elts(tup, expr.type.elt_types)):
      self.append("%s[%d] = %s;" % array_name, i, elt )
    return array_name
    
  def array_to_tuple(self, arr, n, elt_t):
    elts = [self.box_scalar("%s[%d]" % (arr,i), elt_t) for i in xrange(n)]
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
    if isinstance(expr.type, BoolT):
      return "Py_True == Py_%s" % expr.value
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
      return self.unbox_scalar(proj_str, t)
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
      shape_name = self.fresh_name("strides")
      shape_array = "PyArray_DIMS( (PyArrayObject*) %s)" % v
      self.append("npy_intp* %s = %s;" % (shape_name, shape_array))
      return self.array_to_tuple(shape_name, n, elt_t)
      
    elif attr == "strides":
      if debug: self.check_array(v)
      elt_types = t.elt_types
      n = len(elt_types)
      elt_t = elt_types[0]
      assert all(t == elt_t for t in elt_types)
      strides_name = self.fresh_name("strides")
      # if debug: self.printf("Getting strides of %s" % v)
      self.append("npy_intp* %s = PyArray_STRIDES(%s);" % (strides_name, v))
      strides_tuple = self.array_to_tuple(strides_name, n, elt_t)
      # if debug: self.printf("Turning array to tuple")
      return strides_tuple
    elif attr == 'offset':
      return "0"
    elif attr in ('size', 'nelts'):
      return "PyArray_Size(%s)" % v
    else:
      assert False, "Unsupported attribute %s" % attr 
    
  def visit_Attribute(self, expr):
    attr = expr.name
    v = self.visit_expr(expr.value) 
    return self.attribute(v, attr, expr.type)
  
  def visit_PrimCall(self, expr):
    t = expr.type
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
    elif p == prims.remainder:
      x,y = args
      assert isinstance(t, IntT), "Modulo not yet implemented for floats"
      rem = self.fresh_var(t, "rem", "%s %% %s" % (x,y))
      y_is_negative = self.fresh_var(t, "y_is_negative", "%s < 0" % y)
      rem_is_negative = self.fresh_var(t, "rem_is_negative", "%s < 0" % rem)
      y_nonzero = self.fresh_var(t, "y_nonzero", "%s != 0" % y)
      rem_nonzero = self.fresh_var(t, "rem_nonzero", "%s != 0" % rem)
      neither_zero = self.fresh_var(t, "neither_zero", "%s && %s" % (y_nonzero, rem_nonzero))
      diff_signs = self.fresh_var(t, "diff_signs", "%s ^ %s" % (y_is_negative, rem_is_negative))
      should_flip = self.fresh_var(t, "should_flip", "%s && %s" % (neither_zero, diff_signs))
      flipped_rem = self.fresh_var(t, "flipped_rem", "%s + %s" % (y, rem))
      return "%s ? %s : %s" % (should_flip, flipped_rem, rem)
    else:
      assert False, "Prim not yet implemented: %s" % p
  
  def visit_Select(self, expr):
    cond = self.visit_expr(expr.cond)
    true = self.visit_expr(expr.true_value)
    false = self.visit_expr(expr.false_value)
    return "%s ? %s : %s" % (cond, true, false) 
  
  def visit_Assign(self, stmt):
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
  
  
  
  def visit_If(self, stmt):
    cond = self.visit_expr(stmt.cond)
    true = self.visit_block(stmt.true) + self.visit_merge_left(stmt.merge)
    false = self.visit_block(stmt.false) + self.visit_merge_right(stmt.merge)
    return "if(%s) {%s} else {%s}" % (cond, true, false) 
  
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
    v = self.as_pyobj(stmt.value)
    self.print_pyobj_type(v, "Return type: ")
    self.print_pyobj(v, "Return value: ")
    return "return %s;" % v
  
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
  
  def check_tuple(self, tup):
    self.printf("Checking tuple type for %s" % tup)
    self.append("""
      if (!PyTuple_Check(%s)) { 
        PyErr_Format(PyExc_AssertionError, 
                    "Expected %s to be tuple, got %%s", 
                    %s); 
        return NULL;
      }""" % (tup, tup, self.c_type_str(tup)))
 
  def check_array(self, arr):
    self.append('printf("Checking array type for %s @ %%p\\n", %s);' % (arr, arr,))
    self.append("""
      if (!PyArray_Check(%s)) { 
        PyErr_Format(PyExc_AssertionError, 
                    "Expected %s to be array, got %%s", 
                    %s); 
        return NULL;
      }""" % (arr, arr, self.c_type_str(arr)))
  
  
  def check_int(self, x):
    self.append('printf("Checking int type for %s @ %%p\\n", %s);' % (x,x))
    self.append("""
      if (!PyArray_IsIntegerScalar(%s)) { 
        PyErr_Format(PyExc_AssertionError, 
                     "Expected %s to be int, got %%s", 
                     %s); 
        return NULL;
      }""" % (x, x, self.c_type_str(x)))
  
  def check_type(self, v, t):
    if not debug: return
    if isinstance(t, (ClosureT, TupleT)):
      self.check_tuple(v)
    elif isinstance(t, IntT):
      self.check_int(v)
    elif isinstance(t, ArrayT):
      self.check_array(v)
        
  def visit_fn(self, fn):
    c_fn_name = self.fresh_name(fn.name)
    dummy = self.fresh_name("dummy")
    args = self.fresh_name("args")
    
    c_args = "PyObject* %s, PyObject* %s" % (dummy, args) #", ".join("PyObject* %s" % self.name(n) for n in fn.arg_names)
    
    self.push()
    for i, argname in enumerate(fn.arg_names):
      c_name = self.name(argname)
      self.append("PyObject* %s = PyTuple_GetItem(%s, %d);" % (c_name, args, i))
      if debug:
        self.check_type(c_name, fn.type_env[argname])
         
      if debug:
        self.printf("Printing arg #%d %s" % (i,c_name))
        self.print_pyobj_type(c_name, text = "Type: ")
        self.print_pyobj(c_name, text = "Value: ")
      
      t = fn.type_env[argname]
      if isinstance(t, ScalarT):
        new_name = self.name(argname, overwrite = True)
        self.append("%s %s = %s;" % (to_ctype(t), new_name, self.unbox_scalar(c_name, t)))
        
    c_body = self.visit_block(fn.body, push=False)
    c_body = self.indent("\n" + c_body )#+ "\nPyGILState_Release(gstate);")
    fndef = "PyObject* %(c_fn_name)s (%(c_args)s) {%(c_body)s}" % locals()
    return c_fn_name, fndef 

  
    
    
    
    
    