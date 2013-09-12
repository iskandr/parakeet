import collections 
import ctypes

from treelike import NestedBlocks

from .. import names, prims 
from ..analysis import use_count
from ..syntax import Var, Const, TypedFn 
from ..ndtypes import (TupleT,  ArrayT, ClosureT, NoneT, 
                       elt_type, ScalarT, 
                       FloatT, Float32, Float64, 
                       IntT, BoolT, Int64, SignedT,) 


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
  

  def breakpoint(self):
    self.append("raise(SIGINT);")
    
  def unbox_scalar(self, x, t):
    assert isinstance(t, ScalarT), "Expected scalar type, got %s" % t
    
    result = self.fresh_var(t, "scalar_value")
    if isinstance(t, IntT):
      check = "PyInt_Check"
      if isinstance(t, SignedT):
        get = "PyInt_AsLong"
      else:
        get = "PyInt_AsUnsignedLongMask"
    elif isinstance(t, FloatT):
      check = "PyFloat_Check"
      get = "PyFloat_AsDouble"
    else:
      assert isinstance(t, BoolT), "Unexpected type %s" % t 
      check = "PyBool_Check"
      get = "PyObject_IsTrue"
    
    self.append("""
      if (%(check)s(%(x)s)) { %(result)s = %(get)s(%(x)s); }
      else { PyArray_ScalarAsCtype(%(x)s, &%(result)s); }
    """ % locals())
    
    return result 
      
  def box_scalar(self, x, t):
    
    if isinstance(t, BoolT):
      return "PyBool_FromLong(%s)" % x
    elif isinstance(t, NoneT):
      self.append("Py_INCREF(Py_None);")
      return "Py_None"
    if x.replace("_", "").isalpha():
      scalar = x
    else:
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
    stripped = stmt.strip()
    
    assert len(stripped) == 0 or \
      ";" in stripped or \
      stripped.startswith("//") or \
      stripped.startswith("/*"), "Invalid statement: %s" % stmt
    self.blocks[-1].append(stmt)
  
  def newline(self):
    self.append("\n")
    
  def comment(self, text):
    self.append("// %s" % text)
  
  def printf(self, fmt, *args):
    
    result = 'printf("%s\\n"' % fmt
    if len(args) > 0:
      result = result + ", " + ", ".join(str(arg) for arg in args)
    self.append( result + ");" )
    
  
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
    prefix = prefix.replace(".", "")
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
    prefix = prefix.replace(".", "")

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
    elif isinstance(expr.type, NoneT):
      return "0"
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
      elt_obj = self.fresh_var("PyObject*", "%s_elt" % tup, proj_str)
      result = self.unbox_scalar(elt_obj, t)
      if debug and t == Int64:
        self.append(""" printf("tupleproj %s[%d] = %%ld\\n", %s);""" % (tup, idx, result))
      return result
    else:
      return proj_str 
  
  def tuple_elts(self, tup, ts):
    result = []
    for i,t in enumerate(ts):
      result.append(self.tuple_elt(tup, i, t))
    return result
  
  def visit_TupleProj(self, expr):
    tup = self.visit_expr(expr.tuple)
    result = self.tuple_elt(tup, expr.index, expr.type)
    return result
  
  def visit_ClosureElt(self, expr):
    clos = self.visit_expr(expr.closure)
    return self.tuple_elt(clos, expr.index, expr.type)
  
  def visit_Index(self, expr):
    arr = self.visit_expr(expr.value)
    idx = self.visit_expr(expr.index)
    #self.breakpoint()
    return "%s[%s]" % (arr, idx)
  
  
  def visit_ArrayView(self, expr):
    data = self.visit_expr(expr.data)
    shape = self.visit_expr(expr.shape)
    strides = self.visit_expr(expr.strides)
    count = self.visit_expr(expr.size)
    offset = self.visit_expr(expr.offset)
    bytes_per_elt = expr.type.elt_type.dtype.itemsize
    offset_bytes = self.fresh_var("npy_intp", "offset_bytes", "%s * %d" % (offset, bytes_per_elt))
    buffer_name = self.fresh_name("array_buffer")
    bytes_per_elt = expr.type.elt_type.dtype.itemsize
    self.append("PyObject* %s = PyBuffer_FromReadWriteMemory(%s, %s * %d);" % (buffer_name,  data, count, bytes_per_elt))
    dtype = "PyArray_DescrFromType(%s)" % to_dtype(expr.type.elt_type)
    
    vec_name = self.fresh_name("linear_array")
    #   _members = ['data', 'shape', 'strides', 'offset', 'size']
    self.append("PyObject* %s = PyArray_FromBuffer(%s, %s, %s, %s);" % \
                (vec_name, buffer_name, dtype, count, offset_bytes))
    reshaped  = self.fresh_name("reshaped")
    self.append("PyObject* %s = PyArray_Reshape(( PyArrayObject*) %s, %s);" % (reshaped, vec_name, shape))
    strides_array = self.fresh_name("strides_array")
    self.append("npy_intp* %s = PyArray_STRIDES(%s);" % (strides_array, reshaped))
    for i, stride_t in enumerate(expr.strides.type.elt_types):
      stride_value = self.tuple_elt(strides, i, stride_t)
      self.append("%s[%d] = %s * %d;" % (strides_array, i, stride_value, bytes_per_elt) )
    return reshaped
  
  def strides(self, array_expr):
    arr_t = array_expr.type
    assert isinstance(arr_t, ArrayT), \
      "Can only get strides of array, not %s : %s" % (array_expr, arr_t)
    elt_t = arr_t.elt_type
    arr = self.visit_expr(array_expr)
    # if debug: self.check_array(arr)
    
    
    bytes_per_elt = elt_t.dtype.itemsize

    strides_tuple_t = arr_t.strides_t
    stride_t = strides_tuple_t.elt_types[0]
    
    assert all(t == stride_t for t in strides_tuple_t)
    n = len(strides_tuple_t.elt_types)
    strides_bytes = self.fresh_name("strides_bytes")
    self.append("npy_intp* %s = PyArray_STRIDES(%s);" % (strides_bytes, arr))
    strides_elts = self.fresh_name("strides_elts")
    self.append("npy_intp %s[%d];" % (strides_elts, n))
    for i in xrange(n):
      if debug:
        self.printf("converting strides %s[%d] = %%ld to %%ld" % (strides_bytes, i), 
                    "%s[%d]" % (strides_bytes, i), "%s[%d] / %d" % (strides_bytes, i, bytes_per_elt))   
      self.append("%s[%d] = %s[%d] / %d;" % (strides_elts, i, strides_bytes, i, bytes_per_elt))
    strides_tuple = self.array_to_tuple(strides_elts, n, stride_t)
    return strides_tuple
    
        
  def attribute(self, v, attr, t):
    if attr == "data":
      # if debug: self.check_array(v)
      return "(%s) PyArray_DATA (%s)" % (to_ctype(t), v)
    elif attr == "shape":
      # if debug: self.check_array(v)
      elt_types = t.elt_types
      n = len(elt_types)
      elt_t = elt_types[0]
      assert all(t == elt_t for t in elt_types)
      shape_name = self.fresh_name("strides")
      shape_array = "PyArray_DIMS( (PyArrayObject*) %s)" % v
      self.append("npy_intp* %s = %s;" % (shape_name, shape_array))
      return self.array_to_tuple(shape_name, n, elt_t)
      
    elif attr == "strides":
      assert False, "Can't directly use NumPy strides without dividing by itemsize"
      
    elif attr == 'offset':
      return "0"
    elif attr in ('size', 'nelts'):
      return "PyArray_Size(%s)" % v
    else:
      assert False, "Unsupported attribute %s" % attr 
    
  def visit_Attribute(self, expr):
    attr = expr.name
    if attr == 'strides':
      return self.strides(expr.value)
    v = self.visit_expr(expr.value) 
    return self.attribute(v, attr, expr.type)
  
  def visit_PrimCall(self, expr):
    t = expr.type
    args = self.visit_expr_list(expr.args)
    p = expr.prim 
    if p == prims.add:
      return "%s + %s" % (args[0], args[1])
    if p == prims.subtract:
      return "%s - %s" % (args[0], args[1])
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
      if t == Float32:
        return "fmodf(%s, %s)" % (x,y)
      elif t == Float64:
        return "fmod(%s, %s)" % (x,y)
      
      assert isinstance(t, (BoolT, IntT)), "Modulo not implemented for %s" % t
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
    if debug: 
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
    self.newline()
    self.comment("Checking tuple type for %s" % tup)
    self.append("""
      if (!PyTuple_Check(%s)) { 
        PyErr_Format(PyExc_AssertionError, 
                    "Expected %s to be tuple, got %%s", 
                    %s); 
        return NULL;
      }""" % (tup, tup, self.c_type_str(tup)))
 
  def check_array(self, arr):
    self.newline()
    self.comment("Checking array type for %s" % arr)
    self.append("""
      if (!PyArray_Check(%s)) { 
        PyErr_Format(PyExc_AssertionError, 
                    "Expected %s to be array, got %%s", 
                    %s); 
        return NULL;
      }""" % (arr, arr, self.c_type_str(arr)))
  
  
  def check_int(self, x):
    self.newline()
    self.comment("Checking int type for %s" % x)
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
    uses = use_count(fn)
    c_fn_name = self.fresh_name(fn.name)
    
    dummy = self.fresh_name("dummy")
    args = self.fresh_name("args")
    
    c_args = "PyObject* %s, PyObject* %s" % (dummy, args) #", ".join("PyObject* %s" % self.name(n) for n in fn.arg_names)
    
    self.push()
    for i, argname in enumerate(fn.arg_names):
      assert argname in uses, "Couldn't find arg %s in use-counts" % argname
      if uses[argname] <= 1:
        self.comment("Skipping unused argument %s" % argname)
        continue
      
      self.comment("Unpacking argument %s"  % argname)
      c_name = self.name(argname)
      self.append("PyObject* %s = PyTuple_GetItem(%s, %d);" % (c_name, args, i))
      
      
      t = fn.type_env[argname]
      if debug:
        self.check_type(c_name, t)
        self.printf("Printing arg #%d %s" % (i,c_name))
        self.print_pyobj_type(c_name, text = "Type: ")
        self.print_pyobj(c_name, text = "Value: ")
      
      if isinstance(t, ScalarT):
        new_name = self.name(argname, overwrite = True)
        self.append("%s %s = %s;" % (to_ctype(t), new_name, self.unbox_scalar(c_name, t)))
        
    c_body = self.visit_block(fn.body, push=False)
    c_body = self.indent("\n" + c_body )#+ "\nPyGILState_Release(gstate);")
    fndef = "PyObject* %(c_fn_name)s (%(c_args)s) {%(c_body)s}" % locals()
    return c_fn_name, fndef 

  
    
    
    
    
    