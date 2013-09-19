 
import ctypes

from treelike import NestedBlocks

from .. import names, prims 
from ..analysis import use_count
from ..syntax import Const, Tuple, TypedFn, Var, TupleProj, ArrayView, PrimCall, Attribute 
from ..ndtypes import (TupleT,  ArrayT, ClosureT, NoneT, 
                       elt_type, ScalarT, 
                       FloatT, Float32, Float64, 
                       IntT, BoolT, Int64, SignedT,
                       PtrT, NoneType) 


from c_types import to_ctype, to_dtype
from base_compiler import BaseCompiler

from compile_util import compile_module, compile_object
from config import (debug, 
                    check_pyobj_types, 
                    print_function_source, 
                    print_module_source, 
                    print_input_ir, 
                    print_commands) 
from parakeet.c_backend.config import use_openmp
 

def compile_flat(fn, _compile_cache = {}):

  key = fn.name, fn.copied_by 
  if key in _compile_cache:
    return _compile_cache[key]
  compiler = FlatFnCompiler()
  name, sig, src = compiler.visit_fn(fn)
  if print_function_source: print "Generated C source for %s: %s" %(name, src)
  obj = compile_object(src, 
                       fn_name = name, 
                       fn_signature = sig, 
                       extra_objects = compiler.extra_objects,
                       forward_declarations =  compiler.forward_declarations, 
                       print_source = print_module_source, 
                       print_commands = print_commands)
  return obj
  

def flat_function_source(fn):
  return compile_flat(fn).src 

def flat_function_name(fn):
  return compile_flat(fn).fn_name 

def flat_function_signature(fn):
  return compile_flat(fn).fn_signature


def compile_entry(fn, _compile_cache = {}):
  key = fn.name, fn.copied_by 
  if key in _compile_cache:
    return _compile_cache[key]
  compiler = PyModuleCompiler()
  name, sig, src = compiler.visit_fn(fn)
  if print_function_source: print "Generated C source for %s: %s" %(name, src)
  compiled_fn = compile_module(src, 
                               fn_name = name,
                               fn_signature = sig, 
                               extra_objects = set(compiler.extra_objects),
                               forward_declarations =  compiler.forward_declarations, 
                               print_source = print_module_source)
  _compile_cache[key]  = compiled_fn
  return compiled_fn


def entry_function_source(fn):
  return compile_entry(fn).src 

def entry_function_name(fn):
  return compile_entry(fn).fn_name 

def entry_function_signature(fn):
  return compile_entry(fn).fn_signature 

class FlatFnCompiler(BaseCompiler):
  
  def __init__(self):
    BaseCompiler.__init__(self)
    
    self.forward_declarations = set([])
    # depends on these .o files
    self.extra_objects = set([]) 
    
  def visit_Alloc(self, expr):
    elt_size = expr.elt_type.dtype.itemsize
    elt_t =  expr.elt_type
    elt_t_str = to_ctype(elt_t)
    nelts = self.fresh_var("npy_intp", "nelts", self.visit_expr(expr.count))
    nbytes = "%d * %s" % (elt_size, nelts)
    ptr  =  "malloc(%s)" % (nbytes,)
    return "PyArray_SimpleNewFromData(1, &%s, %s, %s )" % (nelts, to_dtype(elt_t), ptr) 
  
  
  
  def visit_Const(self, expr):
    if isinstance(expr.type, BoolT):
      return "1" if expr.value else "0"
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
  
  
  def visit_PrimCall(self, expr):
    t = expr.type
    args = self.visit_expr_list(expr.args)
    
    # parenthesize any compound expressions 
    for i, arg_expr in enumerate(expr.args):
      if not isinstance(arg_expr, (Var, Const)):
        args[i] = "(" + args[i] + ")"
        
    p = expr.prim 
    if p == prims.add:
      return "%s + %s" % (args[0], args[1])
    if p == prims.subtract:
      return "%s - %s" % (args[0], args[1])
    elif p == prims.multiply:
      return "%s * %s" % (args[0], args[1])
    elif p == prims.divide:
      return "%s / %s" % (args[0], args[1])
    elif p == prims.negative:
      return "-%s" % args[0]
    elif p == prims.abs:
      x  = args[0]
      return " %s >= 0 ? %s  : -%s" % (x,x,x)
    
    elif p == prims.bitwise_and:
      return "%s & %s" % (args[0], args[1])
    elif p == prims.bitwise_or:
      return "%s | %s" % (args[0], args[1])
    elif p == prims.bitwise_not:
      return "~%s" % args[0]
    
    elif p == prims.logical_and:
      return "%s && %s" % (args[0], args[1])
    elif p == prims.logical_or:
      return "%s || %s" % (args[0], args[1])
    elif p == prims.logical_not:
      return "!%s" % args[0]
    
    
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
      if t == Float32: return "remainderf(%s, %s)" % (x,y)
      elif t == Float64: return "remainder(%s, %s)" % (x,y)
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
    elif p == prims.fmod:
      if t == Float32: return "fmodf(%s, %s)" % (args[0], args[1])
      elif t == Float64: return "fmod(%s, %s)" % (args[0], args[1])
      return "%s %% %s" % (args[0], args[1])
    elif p == prims.maximum:
      x,y = args
      return "(%s > %s) ? %s : %s" % (x,y,x,y)
    elif p == prims.minimum:
      x,y = args
      return "(%s < %s) ? %s : %s" % (x,y,x,y)
    
    elif p == prims.power:
      if t == Float32: 
        return "powf(%s, %s)" % (args[0], args[1])
      else:
        return "pow(%s, %s)" % (args[0], args[1])
    
    elif isinstance(t, FloatT):
      # many float prims implemented using the same name in math.h
      name = p.name
      if name.startswith("arc"):
        # arccos -> acos
        name = "a" + name[3:]
      if t == Float32: name = name + "f" 
      if len(args) == 1:
        return "%s(%s)" % (name, args[0])
      else:
        assert len(args) == 2, "Unexpected prim %s with %d args (%s)" % (p, len(args), args)
        return "%s(%s, %s)" % (name, args[0], args[1])
  
    else:
      assert False, "Prim not yet implemented: %s" % p
  
  def visit_Index(self, expr):
    arr = self.visit_expr(expr.value)
    idx = self.visit_expr(expr.index)
    elt_t = expr.value.type.elt_type
    ptr_t = "%s*" % to_ctype(elt_t)
    return "( (%s) (PyArray_DATA(%s)))[%s]" % (ptr_t, arr, idx)
  
  def visit_Call(self, expr):
    fn = expr.fn
    assert isinstance(fn, TypedFn), "Expected TypedFn, got %s : %s" % (fn, fn.type)
    compiled_fn = compile_flat(fn)
    args = self.visit_expr_list(expr.args)
    self.extra_objects.add(compiled_fn.object_filename)
    self.forward_declarations.add(compiled_fn.fn_signature)
    return "%s(%s)" % (compiled_fn.fn_name, ", ".join(args))
  
  def visit_Select(self, expr):
    cond = self.visit_expr(expr.cond)
    true = self.visit_expr(expr.true_value)
    false = self.visit_expr(expr.false_value)
    return "%s ? %s : %s" % (cond, true, false) 
  
  def is_pure(self, expr):
    return expr.__class__ in (Var, Const, PrimCall, Attribute, TupleProj, Tuple, ArrayView)
  
  def visit_Assign(self, stmt):
    lhs = self.visit_expr(stmt.lhs)
    rhs = self.visit_expr(stmt.rhs)
    
    if stmt.lhs.__class__ is Var:
      return "%s %s = %s;" % (to_ctype(stmt.lhs.type), lhs, rhs)
    else:
      return "%s = %s;" % (lhs, rhs)
  
  def declare_merge_vars(self, merge):
    """ 
    Declare but don't initialize
    """
    for (name, (left, _)) in merge.iteritems():
      c_name = self.name(name)
      t = to_ctype(left.type)
      self.append("%s %s;" % (t, c_name))
   
  def visit_merge_left(self, merge, declare = False):
    
    if len(merge) == 0:
      return ""
    
    self.push()
    self.comment("Merge Phi Nodes (left side) " + str(merge))
    
    
    for (name, (left, _)) in merge.iteritems():
      c_left = self.visit_expr(left)
      c_name = self.name(name)
      if declare:
        self.append("%s %s = %s;"  % (to_ctype(left.type), c_name, c_left))
      else:
        self.append("%s = %s;" % (c_name, c_left))
    return self.pop()
  
  def visit_merge_right(self, merge):
    
    if len(merge) == 0:
      return ""
    self.push()
    self.comment("Merge Phi Nodes (right side) " + str(merge))
    
    for (name, (_, right)) in merge.iteritems():
      self.append("%s = %s;"  % (self.name(name), self.visit_expr(right)))
    return self.pop()
  
  def visit_If(self, stmt):
    self.declare_merge_vars(stmt.merge)
    cond = self.visit_expr(stmt.cond)
    true = self.visit_block(stmt.true) + self.visit_merge_left(stmt.merge, declare = False)
    false = self.visit_block(stmt.false) + self.visit_merge_right(stmt.merge)
    return self.indent("if(%s) {\n%s\n} else {\n%s\n}" % (cond, self.indent(true), self.indent(false))) 
  
  def visit_While(self, stmt):
    decls = self.visit_merge_left(stmt.merge, declare = True)
    cond = self.visit_expr(stmt.cond)
    body = self.visit_block(stmt.body) + self.visit_merge_right(stmt.merge)
    return decls + "while (%s) {%s}" % (cond, body)
  
  def visit_ForLoop(self, stmt):
    s = self.visit_merge_left(stmt.merge, declare=True)
    start = self.visit_expr(stmt.start)
    stop = self.visit_expr(stmt.stop)
    step = self.visit_expr(stmt.step)
    var = self.visit_expr(stmt.var)
    t = to_ctype(stmt.var.type)
    
    body =  self.visit_block(stmt.body)
    body += self.visit_merge_right(stmt.merge)
    body = self.indent("\n" + body)
    
    s += "\n %(t)s %(var)s;"
    s += "\nfor (%(var)s = %(start)s; %(var)s < %(stop)s; %(var)s += %(step)s) {%(body)s}"
    return s % locals()

  def visit_Return(self, stmt):
    assert not self.return_by_ref, "Returning multiple values not yet implemented: %s" % stmt 
    if self.return_void:
      return "return;"
    else:
      v = self.visit_expr(stmt.value)
      return "return %s;" % v
      
  def visit_block(self, stmts, push = True):
    if push: self.push()
    for stmt in stmts:
      s = self.visit_stmt(stmt)
      self.append(s)
    self.append("\n")
    return self.indent("\n" + self.pop())
  
  def visit_ParFor(self, stmt):
    bounds = self.visit_expr(stmt.bounds)
    print stmt.bounds, bounds 
    i = self.fresh_var("int64_t", "i")
    omp = "#pragma omp parallel for" if use_openmp else ""
    body = "" 
    return """
      %(omp)s
      for(%(i)s = 0; %(i)s < 10; ++%(i)s) {
        %(body)s
      }
    """ % locals()
      
  def visit_TypedFn(self, expr):
    
    return flat_function_name(expr)

  def visit_UntypedFn(self, expr):
    assert False, "Unexpected UntypedFn %s in C backend, should have been specialized" % expr.name
  
  
  def return_types(self, fn):
    if isinstance(fn.return_type, TupleT):
      return fn.return_type.elt_types
    else:
      assert isinstance(fn.return_type, (PtrT, ScalarT))
      return [fn.return_type]
    
  
  def visit_fn(self, fn):
    c_fn_name = self.fresh_name(fn.name)
    arg_types = [to_ctype(t) for t in fn.input_types]
    arg_names = [self.name(old_arg) for old_arg in fn.arg_names]
    return_types = self.return_types(fn)
    n_return = len(return_types)
    
    if n_return == 1:
      return_type = to_ctype(return_types[0])
      self.return_void = (return_type == NoneType)
      self.return_by_ref = False
    elif n_return == 0:
      return_type = "void"
      self.return_void = True
      self.return_by_ref = False
    else:
      return_type = "void"
      self.return_void = True
      self.return_by_ref = True
      self.return_var_types = [to_ctype(t) for t in return_types]
      self.return_var_names = [self.fresh_name("return_value%d" % i) for i in xrange(n_return)]
      arg_types = arg_types + ["%s*" % t for t in self.return_var_types] 
      arg_names = arg_names + self.return_var_names
      
    args_str = ", ".join("%s %s" % (t, name) for (t,name) in zip(arg_types,arg_names))
    body_str = self.visit_block(fn.body)
    sig = "%s %s(%s)" % (return_type, c_fn_name, args_str)
    src = "%s { %s }" % (sig, body_str) 
    return c_fn_name, sig, src
    

class PyModuleCompiler(FlatFnCompiler):
   
  def unbox_scalar(self, x, t, target = "scalar_value"):
    assert isinstance(t, ScalarT), "Expected scalar type, got %s" % t
    
    result = self.fresh_var(t, target)
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
    if isinstance(expr.type, (NoneT, ScalarT)):
      return self.box_scalar(x, expr.type)
    else:
      return x
  
  def as_pyobj_list(self, exprs):
    return [self.as_pyobj(expr) for expr in exprs]
  
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
    
  def tuple_to_stack_array(self, expr, name = "array_from_tuple", elt_type = None):
    t0 = expr.type.elt_types[0]
    
    assert expr.type.__class__ is TupleT 
    assert all(t == t0 for t in expr.type.elt_types[1:])
    
    if expr.__class__ is Tuple:
      elts = [self.visit_expr(elt_expr) for elt_expr in expr.elts]
    else:
      tup = self.visit_expr(expr)
      self.check_tuple(tup)
      elts = self.tuple_elts(tup, expr.type.elt_types)
    
    array_name = self.fresh_name(name)
    n = len(expr.type.elt_types)
    if elt_type is None:
      elt_type = to_ctype(t0)
    self.append("%s %s[%d];" % (elt_type, array_name, n))
    for i, elt in enumerate(elts):
      self.append("%s[%d] = %s;" % (array_name, i, elt))
    return array_name
    
  def array_to_tuple(self, arr, n, elt_t):
    if n == 0: return "PyTuple_Pack(0);"
    elts = [self.box_scalar("%s[%d]" % (arr,i), elt_t) for i in xrange(n)]
    elt_str = ", ".join(elts)
    return "PyTuple_Pack(%d, %s)" % (n, elt_str)
  
  def tuple_elts(self, tup, ts):
    result = []
    for i,t in enumerate(ts):
      result.append(self.tuple_elt(tup, i, t))
    return result
  
  def mk_tuple(self, elts):
    n = len(elts)
    if n == 0: return "PyTuple_Pack(0);"
    elt_str = ", ".join(self.as_pyobj_list(elts)) 
    return "PyTuple_Pack(%d, %s)" % (n, elt_str)
  
  
  def check_tuple(self, tup):
    if not check_pyobj_types: return 
    self.newline()
    self.comment("Checking tuple type for %s" % tup)
    self.append("""
      if (!PyTuple_Check(%s)) { 
        PyErr_Format(PyExc_AssertionError, 
                    "Expected %s to be tuple, got %%s", 
                    %s); 
        return NULL;
      }""" % (tup, tup, self.c_type_str(tup)))
 
  def check_slice(self, obj):
    if not check_pyobj_types: return 
    self.newline()
    self.comment("Checking slice type for %s" % obj)
    self.append("""
      if (!PySlice_Check(%s)) { 
        PyErr_Format(PyExc_AssertionError, 
                    "Expected %s to be slice, got %%s", 
                    %s); 
        return NULL;
      }""" % (obj, obj, self.c_type_str(obj)))
  
  def check_array(self, arr):
    if not check_pyobj_types: return 
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
    if not check_pyobj_types: return 
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
    if not check_pyobj_types: return 
    if isinstance(t, (ClosureT, TupleT)):
      self.check_tuple(v)
    elif isinstance(t, IntT):
      self.check_int(v)
    elif isinstance(t, ArrayT):
      self.check_array(v)
      
  def tuple_elt(self, tup, idx, t):
    self.check_tuple(tup)
    proj_str = "PyTuple_GetItem(%s, %d)" % (tup, idx)
    if isinstance(t, ScalarT):
      elt_obj = self.fresh_var("PyObject*", "%s_elt" % tup, proj_str)
      result = self.unbox_scalar(elt_obj, t)
      if debug and t == Int64:
        self.append(""" printf("tupleproj %s[%d] = %%" PRId64 "\\n", %s);""" % (tup, idx, result))
      return result
    else:
      return proj_str 
 
  
  def strides(self, array_expr):
    arr_t = array_expr.type
    assert isinstance(arr_t, ArrayT), \
      "Can only get strides of array, not %s : %s" % (array_expr, arr_t)
    elt_t = arr_t.elt_type
    arr = self.visit_expr(array_expr)
    
    bytes_per_elt = elt_t.dtype.itemsize

    strides_tuple_t = arr_t.strides_t
    stride_t = strides_tuple_t.elt_types[0]
    
    assert all(t == stride_t for t in strides_tuple_t)
    n = len(strides_tuple_t.elt_types)
    strides_bytes = self.fresh_name("strides_bytes")
    self.append("npy_intp* %s = PyArray_STRIDES( (PyArrayObject*) %s);" % (strides_bytes, arr))
    strides_elts = self.fresh_name("strides_elts")
    self.append("npy_intp %s[%d];" % (strides_elts, n))
    for i in xrange(n):
      if debug:
        self.printf("converting strides %s[%d] = %%ld to %%ld" % (strides_bytes, i), 
                    "%s[%d]" % (strides_bytes, i), "%s[%d] / %d" % (strides_bytes, i, bytes_per_elt))   
      self.append("%s[%d] = %s[%d] / %d;" % (strides_elts, i, strides_bytes, i, bytes_per_elt))
    strides_tuple = self.array_to_tuple(strides_elts, n, stride_t)
    return strides_tuple
    
  def decref(self, obj):
    self.append("Py_DECREF(%s);" % obj)
    
  def attribute(self, v, attr, t):
    if attr == "data":
      self.check_array(v)
      return "PyArray_Ravel((PyArrayObject*) %s, 0)" % (v,)  
      #c_expr = "(%s) PyArray_DATA (  (PyArrayObject*) %s)" % (to_ctype(t), v)
      #data_var = self.fresh_var(to_ctype(t), "data", c_expr)
      # pyint = self.fresh_var("PyObject*", "data_pyint", "PyInt_FromLong((int64_t) %s)" % data_var)
      # self.append("PySet_Add(%s, %s);" % (self.external_pointer_set, pyint))
      # self.decref(pyint)
      #return data_var
    
    elif attr == "shape":
      self.check_array(v)
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
    
    elif attr in ('start', 'stop', 'step'):
      self.check_slice(v)
      obj = "((PySliceObject*)%s)->%s" % (v, attr)
      return self.unbox_scalar(obj, t, attr)
    else:
      assert False, "Unsupported attribute %s" % attr 
   
    
  
  def visit_AllocArray(self, expr):
    shape = self.tuple_to_stack_array(expr.shape)
    t = to_dtype(elt_type(expr.type))
    return "PyArray_SimpleNew(%d, %s, %s)" % (expr.type.rank, shape, t)
    
  def visit_Tuple(self, expr):
    return self.mk_tuple(expr.elts)
  
  def visit_Closure(self, expr):
    return self.mk_tuple(expr.args)
  
  def visit_TupleProj(self, expr):
    tup = self.visit_expr(expr.tuple)
    result = self.tuple_elt(tup, expr.index, expr.type)
    return result
  
  def visit_ClosureElt(self, expr):
    clos = self.visit_expr(expr.closure)
    return self.tuple_elt(clos, expr.index, expr.type)
  
  
  
  def visit_ArrayView(self, expr):
    vec_name = self.visit_expr(expr.data)
    ndims = expr.type.rank
    offset = self.visit_expr(expr.offset)
    bytes_per_elt = expr.type.elt_type.dtype.itemsize
    dtype = to_dtype(expr.type.elt_type)
    
    count = self.visit_expr(expr.size)
    offset_bytes = self.fresh_var("npy_intp", "offset_bytes", "%s * %d" % (offset, bytes_per_elt))
    
   
    shape = self.visit_expr(expr.shape)
    reshaped  = self.fresh_var("PyObject*", "reshaped")
    self.append("%s = PyArray_Reshape(( PyArrayObject*) %s, %s);" % \
                (reshaped, vec_name, shape))
    
    strides_elts = self.tuple_to_stack_array(expr.strides, name = "strides", elt_type = "npy_intp")
    strides_bytes = self.fresh_name("strides_bytes")
    self.append("npy_intp* %s = PyArray_STRIDES(  (PyArrayObject*) %s);" % (strides_bytes, reshaped))
    for i, _ in enumerate(expr.strides.type.elt_types):
      self.append("%s[%d] = %s[%d] * %d;" % (strides_bytes, i, strides_elts, i, bytes_per_elt) )
    return reshaped
  
    
  def visit_Attribute(self, expr):
    attr = expr.name
    if attr == 'strides':
      return self.strides(expr.value)
    v = self.visit_expr(expr.value) 
    return self.attribute(v, attr, expr.type)
  
  
    
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
    return flat_function_name(expr)

  def visit_UntypedFn(self, expr):
    assert False, "Unexpected UntypedFn %s in C backend, should have been specialized" % expr.name
  
         
  def visit_fn(self, fn):
    if print_input_ir:
      print "=== Compiling to C (entry function) ==="
      print fn
    c_fn_name = self.fresh_name(fn.name)
    uses = use_count(fn)
    self.push()
    
    
    dummy = self.fresh_name("dummy")
    args = self.fresh_name("args")
    
    if debug: 
      self.newline()
      self.printf("\\nStarting %s : %s..." % (c_fn_name, fn.type))
      
    for i, argname in enumerate(fn.arg_names):
      assert argname in uses, "Couldn't find arg %s in use-counts" % argname
      if uses[argname] <= 1:
        self.comment("Skipping unused argument %s" % argname)
        continue
      self.comment("Unpacking argument %s"  % argname)
      c_name = self.name(argname)
      self.append("PyObject* %s = PyTuple_GetItem(%s, %d);" % (c_name, args, i))
      t = fn.type_env[argname]
      self.check_type(c_name, t)
      if debug:
        self.printf("Printing arg #%d %s" % (i,c_name))
        self.print_pyobj_type(c_name, text = "Type: ")
        self.print_pyobj(c_name, text = "Value: ")
      
      if isinstance(t, ScalarT):
        new_name = self.name(argname, overwrite = True)
        self.unbox_scalar(c_name, t, target = new_name)

    self.external_pointer_set = \
      self.fresh_var("PyObject*", "external_pointer_set", "PySet_New(NULL)")
        
    c_body = self.visit_block(fn.body, push=False)
    c_body = self.indent("\n" + c_body )#+ "\nPyGILState_Release(gstate);")
    c_args = "PyObject* %s, PyObject* %s" % (dummy, args) #", ".join("PyObject* %s" % self.name(n) for n in fn.arg_names)
    c_sig = "PyObject* %(c_fn_name)s (%(c_args)s)" % locals() 
    fndef = "%s {%s}" % (c_sig, c_body)
    return c_fn_name, c_sig, fndef 

  
    
    
    
    
    