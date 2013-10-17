from ..analysis import use_count
from ..syntax import Tuple 
 
from ..ndtypes import (TupleT,  ArrayT, NoneT, 
                       elt_type, ScalarT, 
                       FloatT, 
                       BoolT,  
                       IntT,  Int64, SignedT,
                       PtrT,  
                       ClosureT) 

import type_mappings
from flat_fn_compiler import FlatFnCompiler
from compile_util import compile_module
import config 


class PyModuleCompiler(FlatFnCompiler):
  """
  Compile a Parakeet function into a Python module with an 
  entry-point that unboxes all the PyObject inputs, 
  runs a flattened computations, and boxes the result as PyObjects
  """
  def __init__(self, module_entry = True, *args, **kwargs):
    FlatFnCompiler.__init__(self, module_entry = module_entry, *args, **kwargs)
    
  def unbox_scalar(self, x, t, target = None):
    assert isinstance(t, ScalarT), "Expected scalar type, got %s" % t
    if target is None:
      target = "scalar_value"
    
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
      self.append("%s %s = %s;" % (self.to_ctype(t), scalar, x))
    return "PyArray_Scalar(&%s, PyArray_DescrFromType(%s), NULL)" % (scalar, type_mappings.to_dtype(t) )
  
  def box_tuple(self, x, t):
    if isinstance(t, ClosureT):
      elt_types = t.arg_types 
    else:
      assert isinstance(t, TupleT)
      elt_types = t.elt_types
    unboxed_elts = self.tuple_elts(x, elt_types)
    boxed_elts = [self.box(elt, elt_t) for elt, elt_t in zip(unboxed_elts, elt_types)]
    n = len(boxed_elts)
    if n == 0:
      return "PyTuple_Pack(0)"
    else:
      return "PyTuple_Pack(%d, %s)" % (n, ", ".join(boxed_elts))
  
  def box(self, x, t):
    if isinstance(t, (NoneT, ScalarT)):
      return self.box_scalar(x, t)
    elif isinstance(t, (ClosureT, TupleT)):
      return self.box_tuple(x, t)
    else:
      # all other types already boxed 
      return x
      
      
  def as_pyobj(self, expr):
    """
    Compile the expression and if necessary box it up as a PyObject
    """
    x = self.visit_expr(expr)
    if isinstance(expr.type, (NoneT, ScalarT, TupleT, ClosureT)):
      return self.box(x, expr.type)
    elif isinstance(expr.type, ArrayT):
      return "(PyObject*) " + x 
    else:
      return x
  
  def as_pyobj_list(self, exprs):
    return [self.as_pyobj(expr) for expr in exprs]
  
  def c_str(self, obj):
    return "PyString_AsString(PyObject_Str(%s))" % obj
  
  def c_type_str(self, obj):
    return self.c_str("PyObject_Type((PyObject*) %s)" % obj)
   
  def print_pyobj(self, obj, text = ""):
    text = '"%s"' % text
    self.append('printf("%%s%%s\\n", %s, %s);' % (text, self.c_str(obj)))
  
  def print_pyobj_type(self, obj, text=""):
    text = '"%s"' % text
    self.append('printf("%%s%%s\\n", %s, %s);' % (text, self.c_type_str(obj)))
  
  def unbox(self, boxed, t, target = None):
    if isinstance(t, (NoneT, PtrT, ScalarT)):
      return self.unbox_scalar(boxed, t, target = target)
    elif isinstance(t, (ClosureT, TupleT)):
      return self.unbox_tuple(boxed, t, target = target)
    else:
      return boxed 
  
  def unbox_tuple(self, boxed_tuple, tuple_t, target = "tuple_value"):
    if isinstance(tuple_t, TupleT):
      elt_types = tuple(tuple_t.elt_types)
    elif isinstance(tuple_t, ClosureT):
      elt_types = tuple(tuple_t.arg_types)
    else:
      assert False, "Expected tuple type, got %s" % tuple_t
     
    c_struct_t = self.struct_type_from_fields(elt_types)
    tuple_elts = self.tuple_elts(boxed_tuple, elt_types, boxed=True)
    elts_str = ", ".join(tuple_elts)
    return self.fresh_var(c_struct_t, target, "{" + elts_str + "}")
     
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
      elt_type = self.to_ctype(t0)
    self.append("%s %s[%d];" % (elt_type, array_name, n))
    for i, elt in enumerate(elts):
      self.append("%s[%d] = %s;" % (array_name, i, elt))
    return array_name
    
  def array_to_tuple(self, arr, n, elt_t, boxed = False):
    raw_elts = ["%s[%d]" % (arr,i) for i in xrange(n)]
    if boxed:
      if n == 0: 
        return "PyTuple_Pack(0);"
      else:
        boxed_elts = [self.box_scalar(raw_elt, elt_t) for raw_elt in raw_elts]
        elt_str = ", ".join(boxed_elts)
        return "PyTuple_Pack(%d, %s)" % (n, elt_str)
    else:
      field_types = (self.to_ctype(elt_t),) * n 
      tuple_struct_t = self.struct_type_from_fields(field_types)
      init = "{%s}" % ", ".join(raw_elts)
      return self.fresh_var(tuple_struct_t, "tuple_value", init)
      
  
  def tuple_elts(self, tup, ts, boxed = False):
    result = []
    for i,t in enumerate(ts):
      result.append(self.tuple_elt(tup, i, t, boxed = boxed))
    return result
  
  def mk_tuple(self, elts, boxed = False):
    n = len(elts)
    if boxed:
      if n == 0: 
        return "PyTuple_Pack(0);"
      else:
        elt_str = ", ".join(self.as_pyobj_list(elts)) 
        return "PyTuple_Pack(%d, %s)" % (n, elt_str)
    else:
      field_types = tuple(elt.type for elt in elts)
      tuple_struct_t = self.struct_type_from_fields(field_types)
      elts_str = ", ".join(self.visit_expr(elt) for elt in elts)
      init = "{" + elts_str +  "}"
      return self.fresh_var(tuple_struct_t, "tuple_value", init)
    
  
  def check_tuple(self, tup):
    if not config.check_pyobj_types: return 
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
    if not config.check_pyobj_types: return 
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
    if not config.check_pyobj_types: return 
    self.newline()
    self.comment("Checking array type for %s" % arr)
    self.append("""
      if (!PyArray_Check(%s)) { 
        PyErr_Format(PyExc_AssertionError, 
                    "Expected %s to be array, got %%s : %%s", 
                    %s, %s); 
        return NULL;
      }""" % (arr, arr, self.c_str(arr), self.c_type_str(arr)))
  
  
  def check_bool(self, x):
    if not config.check_pyobj_types: return 
    self.newline()
    self.comment("Checking bool type for %s" % x)
    self.append("""
      if (!PyArray_IsScalar(%s, Bool)) { 
        PyErr_Format(PyExc_AssertionError, 
                     "Expected %s to be bool, got %%s", 
                     %s); 
        return NULL;
      }""" % (x, x, self.c_type_str(x)))
  
  def check_int(self, x):
    if not config.check_pyobj_types: return 
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
    if not config.check_pyobj_types: return 
    if isinstance(t, (ClosureT, TupleT)):
      self.check_tuple(v)
    elif isinstance(t, BoolT):
      self.check_bool(v)
    elif isinstance(t, IntT):
      self.check_int(v)
    elif isinstance(t, ArrayT):
      self.check_array(v)
      
  def tuple_elt(self, tup, idx, t, boxed = False):
    if boxed: 
      self.check_tuple(tup)
      proj_str = "PyTuple_GetItem(%s, %d)" % (tup, idx)
      if isinstance(t, (TupleT, ScalarT)):
        elt_obj = self.fresh_var("PyObject*", "%s_elt" % tup, proj_str)
        result = self.unbox(elt_obj, t)
        if config.debug and t == Int64:
          self.append(""" printf("tupleproj %s[%d] = %%" PRId64 "\\n", %s);""" % (tup, idx, result))
        return result
      
      else:
        return proj_str
    else:
      return "%s.elt%d" % (tup, idx)  
 
  
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
      if config.debug:
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
      struct_type = self.struct_type(t)
      
      ptr_type = type_mappings.to_ctype(t) 
      data_field = "(%s) (((PyArrayObject*) %s)->data)" % (ptr_type, v) 
      # get the data field but also fill the base object 
      return self.fresh_var(struct_type, "data", "{%s, %s}" % (data_field, v))
      
      
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
    t = type_mappings.to_dtype(elt_type(expr.type))
    return "(PyArrayObject*) PyArray_SimpleNew(%d, %s, %s)" % (expr.type.rank, shape, t)
    
  def visit_Tuple(self, expr):
    return self.mk_tuple(expr.elts, boxed = False)
  
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
    data = self.visit_expr(expr.data)
    ndims = expr.type.rank 
    offset = self.visit_expr(expr.offset)
    count = self.visit_expr(expr.size)
    
    if expr.strides.__class__ is Tuple:
      strides_elts = self.visit_expr_list(expr.strides.elts)  
    else:
      strides_array = self.tuple_to_stack_array(expr.strides, "strides_array", "npy_intp")
      strides_elts = ["%s[%d]" % (strides_array, i) for i in xrange(ndims)]

    shape_array = self.tuple_to_stack_array(expr.shape, "shape_array", "npy_intp")
    shape_elts = ["%s[%d]" % (shape_array, i) for i in xrange(ndims)]
    
    
    # PyObject* PyArray_SimpleNewFromData(int nd, npy_intp* dims, int typenum, void* data)
    typenum = type_mappings.to_dtype(expr.data.type.elt_type)
    array_alloc = \
      "PyArray_SimpleNewFromData(%d, %s, %s, %s.data)" % (ndims, shape_array, typenum, data)
    vec = self.fresh_var("PyArrayObject*", "fresh_array", array_alloc) 
    self.return_if_null(vec)
    
    # if the pointer had a PyObject reference, 
    # set that as the new array's base 
    self.append("if (%s.base) { %s->base = %s.base; %s->flags &= ~NPY_ARRAY_OWNDATA; }" % \
                (data, vec, data, vec))
        
    numpy_strides = self.fresh_var("npy_intp*", "numpy_strides")
    self.append("%s = PyArray_STRIDES(  (PyArrayObject*) %s);" % (numpy_strides, vec))
    bytes_per_elt = expr.type.elt_type.dtype.itemsize
    
    for i, _ in enumerate(expr.strides.type.elt_types):
      self.append("%s[%d] = %s * %d;" % (numpy_strides, i, strides_elts[i], bytes_per_elt) )
      
    
    self.append("""
      // clear both fortran and c layout flags 
      ((PyArrayObject*) %(vec)s)->flags &= ~NPY_F_CONTIGUOUS;
      ((PyArrayObject*) %(vec)s)->flags &= ~NPY_C_CONTIGUOUS;
    """ % locals())
    
    f_layout_strides = ["1"]
    for shape_elt in shape_elts[1:]:
      f_layout_strides.append(f_layout_strides[-1] + " * " + shape_elt)
    
    c_layout_strides = ["1"]
    for shape_elt in list(reversed(shape_elts))[:-1]:
      c_layout_strides = [c_layout_strides[-1] + " * " + shape_elt] + c_layout_strides
    
    
    
    is_c_layout = "&& ".join(self.eq(actual, ideal, Int64) 
                             for actual, ideal 
                             in zip(strides_elts, c_layout_strides))
    is_f_layout = " && ".join(self.eq(actual, ideal, Int64) 
                             for actual, ideal 
                             in zip(strides_elts, f_layout_strides))
    
       
    # make sure the contiguity flags are set correctly 
    self.append("""
      // it's possible that *neither* of the above flags should be on
      // which is why we enable them separately here 
      if (%(is_f_layout)s) { ((PyArrayObject*)%(vec)s)->flags |= NPY_F_CONTIGUOUS; }
      if (%(is_c_layout)s) { ((PyArrayObject*)%(vec)s)->flags |= NPY_C_CONTIGUOUS; }
    """ % locals())
    return vec
  
    
  def visit_Attribute(self, expr):
    attr = expr.name
    if attr == 'strides':
      return self.strides(expr.value)
    v = self.visit_expr(expr.value) 
    return self.attribute(v, attr, expr.type)
  
  def visit_Return(self, stmt):
    if self.module_entry:
      v = self.as_pyobj(stmt.value)
      if config.debug: 
        self.print_pyobj_type(v, "Return type: ")
        self.print_pyobj(v, "Return value: ")
      return "return %s;" % v
    else:
      return FlatFnCompiler.visit_Return(self, stmt)
  
  def visit_block(self, stmts, push = True):
    if push: self.push()
    for stmt in stmts:
      s = self.visit_stmt(stmt)
      self.append(s)
    self.append("\n")
    return self.pop()
      
  def visit_TypedFn(self, expr):
    return self.get_fn(expr)

  def visit_UntypedFn(self, expr):
    assert False, "Unexpected UntypedFn %s in C backend, should have been specialized" % expr.name
         
  def visit_fn(self, fn):
    if config.print_input_ir:
      print "=== Compiling to C (entry function) ==="
      print fn
    c_fn_name = self.fresh_name(fn.name)
    uses = use_count(fn)
    self.push()
    
    
    dummy = self.fresh_name("dummy")
    args = self.fresh_name("args")
    
    if config.debug: 
      self.newline()
      self.printf("\\nStarting %s : %s..." % (c_fn_name, fn.type))
      
    for i, argname in enumerate(fn.arg_names):
      assert argname in uses, "Couldn't find arg %s in use-counts" % argname
      if uses[argname] <= 1:
        self.comment("Skipping unused argument %s" % argname)
        continue
      self.comment("Unpacking argument %s"  % argname)
      c_name = self.name(argname)
      t = fn.type_env[argname]
      self.comment("Getting arg #%d: %s : %s => %s" % (i+1, argname,  t, c_name) )
      self.append("PyObject* %s = PyTuple_GetItem(%s, %d);" % (c_name, args, i))
      
      self.check_type(c_name, t)
      if config.debug:
        self.printf("Printing arg #%d %s" % (i,c_name))
        self.print_pyobj_type(c_name, text = "Type: ")
        self.print_pyobj(c_name, text = "Value: ")
      
      if isinstance(t, (TupleT, NoneT, PtrT, ClosureT, ScalarT)):
        #new_name = self.name(argname, overwrite = True)

        var = self.unbox(c_name, t, target = argname)

        self.name_mappings[argname] = var
      


    c_body = self.visit_block(fn.body, push=False)

    c_body = self.indent(c_body )
    c_args = "PyObject* %s, PyObject* %s" % (dummy, args) #", ".join("PyObject* %s" % self.name(n) for n in fn.arg_names)
    c_sig = "PyObject* %(c_fn_name)s (%(c_args)s)" % locals() 
    fndef = "%s {%s}" % (c_sig, c_body)
    return c_fn_name, c_sig, fndef 
  
  _entry_compile_cache = {} 
  def compile_entry(self, parakeet_fn):  
    # we include the compiler's class as part of the key
    # since this function might get reused by descendant backends like OpenMP and CUDA
    key = parakeet_fn.cache_key, self.__class__ 
    if key in self._entry_compile_cache:
      return self._entry_compile_cache[key]
    
    name, sig, src = self.visit_fn(parakeet_fn)
    
    if config.print_function_source: 
      print "Generated C source for %s: %s" %(name, src)
    ordered_function_sources = [self.extra_functions[extra_sig] for 
                                extra_sig in self.extra_function_signatures]

    compiled_fn = compile_module(src, 
                                 fn_name = name,
                                 fn_signature = sig, 
                                 extra_objects = set(self.extra_objects),
                                 extra_function_sources = ordered_function_sources, 
                                 declarations =  self.declarations, 
                                 extra_compile_flags = self.extra_compile_flags, 
                                 extra_link_flags = self.extra_link_flags, 
                                 print_source = config.print_module_source)
    self._entry_compile_cache[key]  = compiled_fn
    return compiled_fn

"""
def entry_function_source(fn):
  return compile_entry(fn).src 

def entry_function_name(fn):
  return compile_entry(fn).fn_name 

def entry_function_signature(fn):
  return compile_entry(fn).fn_signature 
"""