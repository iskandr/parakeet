from ..analysis import use_count
from ..syntax import Tuple,  Expr
 
from ..ndtypes import (TupleT,  ArrayT, 
                       NoneT, NoneType,  
                       elt_type, ScalarT, 
                       FloatT, 
                       BoolT,  
                       IntT,  Int64, SignedT,
                       PtrT,  
                       ClosureT, 
                       SliceT, ptr_type)
 

import type_mappings
from fn_compiler import FnCompiler
from compile_util import compile_module
from .. import config as root_config 
import config 

def attr_from_kwargs(obj, kwargs, attr, value = None):
  """
  If an attribute is in the kwargs dictionary, then assign it to the
  given object and remove it from the dictionary, otherwise assign 
  the default value
  """
  if attr in kwargs:
    setattr(obj, attr, kwargs[attr])
    del kwargs[attr]
  else:
    setattr(obj, attr, value)


class PyModuleCompiler(FnCompiler):
  """
  Compile a Parakeet function into a Python module with an 
  entry-point that unboxes all the PyObject inputs, 
  runs a flattened computations, and boxes the result as PyObjects
  """
  def __init__(self, module_entry = True, *args, **kwargs):
    attr_from_kwargs(self, kwargs, 'compiler_cmd')    
    attr_from_kwargs(self, kwargs, 'compiler_flag_prefix')
    attr_from_kwargs(self, kwargs, 'linker_flag_prefix')  
    attr_from_kwargs(self, kwargs, 'src_extension')
    FnCompiler.__init__(self, module_entry = module_entry, *args, **kwargs)
    
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
  
  def unbox_array(self, boxed_array, elt_type, ndims, target = "array_value"):
    shape_ptr = self.fresh_var("npy_intp*", "shape_ptr", "PyArray_DIMS(%s)" % boxed_array)
    strides_bytes = self.fresh_var("npy_intp*", "strides_bytes", 
                                   "PyArray_STRIDES( (PyArrayObject*) %s)" % boxed_array)

    #strides_elts = self.fresh_name("strides_elts")
    #self.append("npy_intp %s[%d];" % (strides_elts, ndims))
    bytes_per_elt = elt_type.dtype.itemsize 
    typename = self.array_struct_type(elt_type, ndims)
    result = self.fresh_var(typename, "unboxed_array")
    ptr = self.get_boxed_array_ptr(boxed_array, ptr_type(elt_type))
    self.setfield(result, 'data', ptr)
    for i in xrange(ndims):
      if config.debug:
        self.printf("converting strides %s[%d] = %%ld to %%ld" % (strides_bytes, i), 
                    "%s[%d]" % (strides_bytes, i), "%s[%d] / %d" % (strides_bytes, i, bytes_per_elt))   
      self.setidx("%s.strides" % result, i, "%s[%d] / %d" % (strides_bytes, i, bytes_per_elt))
      self.setidx("%s.shape" % result, i, "%s[%d]" % (shape_ptr, i))
    self.setfield(result, "offset", "0")
    self.setfield(result, "size", "PyArray_Size(%s)" % boxed_array)
    return result 
  
  def unbox_tuple(self, boxed_tuple, tuple_t, target = "tuple_value"):
    if isinstance(tuple_t, TupleT):
      elt_types = tuple(tuple_t.elt_types)
    elif isinstance(tuple_t, ClosureT):
      elt_types = tuple(tuple_t.arg_types)
    else:
      assert False, "Expected tuple type, got %s" % tuple_t
     
    c_struct_t = self.struct_type_from_fields(elt_types)
    unboxed_elts = []
    for i in xrange(len(elt_types)):
      elt_t = elt_types[i]
      elt_typename = self.to_ctype(elt_t)
      elt = self.fresh_var(elt_typename, "unboxed_elt%d" % i, 
                           self.tuple_elt(boxed_tuple, i, elt_types[i], boxed=True))
      unboxed_elts.append(elt)
    elts_str = ", ".join(unboxed_elts)
    return self.fresh_var(c_struct_t, target, "{" + elts_str + "}")
      
  def unbox_slice(self, boxed, t, target = None):
    if target is None: target = "slice"
    typename = self.to_ctype(t)
    start = self.unbox("((PySliceObject*)%s)->start" % boxed, t.start_type, "start")
    stop = self.unbox("((PySliceObject*)%s)->stop" % boxed, t.stop_type, "stop")
    step = self.unbox("((PySliceObject*)%s)->step" % boxed, t.step_type, "step")
    return self.fresh_var(typename, target, "{%s,%s,%s}" % (start,stop,step) )
    
  def unbox(self, boxed, t, target = None):
    if isinstance(t, NoneT):
      return "0"
    elif isinstance(t, PtrT):
      assert False, "Unexpected raw pointer passed as argument from Python %s : %s" % (boxed, t)
    if isinstance(t, ScalarT):
      return self.unbox_scalar(boxed, t, target = target)
    elif isinstance(t, (ClosureT, TupleT)):
      return self.unbox_tuple(boxed, t, target = target)
    elif isinstance(t, SliceT):
      return self.unbox_slice(boxed, t, target = target)
    elif isinstance(t, ArrayT):
      return self.unbox_array(boxed, 
                              elt_type = t.elt_type, 
                              ndims = t.rank, target = target)
    else:
      assert False, "Don't know how to unbox %s : %s" % (boxed, t)
        
  def box_none(self):
    self.append("Py_INCREF(Py_None);")
    return "Py_None"
  
  def box_scalar(self, x, t):  
    if isinstance(t, BoolT):
      return "PyBool_FromLong(%s)" % x
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
  
  def box_slice(self, x, t):
    start = self.box("%s.start" % x, t.start_type)
    stop = self.box("%s.stop" % x, t.stop_type)
    step = self.box("%s.step" % x, t.step_type)
    return "PySlice_New(%s, %s, %s)" % (start, stop, step)
  
  def box(self, x, t):
    if isinstance(t, ScalarT):
      return self.box_scalar(x, t)
    elif isinstance(t, NoneT):
      return self.box_none()
    elif isinstance(t, (ClosureT, TupleT)):
      return self.box_tuple(x, t)
    elif isinstance(t, SliceT):
      return self.box_sclie(x, t)
    elif isinstance(t, ArrayT):
      return self.box_array(x, t)
    else:
      # all other types already boxed 
      return x
      
      
  def as_pyobj(self, expr):
    """
    Compile the expression and if necessary box it up as a PyObject
    """
    x = self.visit_expr(expr)
    if isinstance(expr.type, (NoneT, ScalarT, TupleT, ClosureT, ArrayT, SliceT)):
      return self.box(x, expr.type)
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

    
  def setfield(self, base, fieldname, value):
    self.append("%s.%s = %s;" % (base, fieldname, value))
  
  def getidx(self, arr, idx, full_array = False):
    """
    Given either the compiled string representation or Parakeet IR
    for an array and index, construct a C indexing expression 
    """

    if isinstance(arr, Expr):
      if isinstance(arr.type, ArrayT):
        arr = self.visit_expr(arr)
        ptr = "%s.data.raw_ptr" % arr
      else:
        assert isinstance(arr.type, PtrT), \
          "Expected array or pointer but got %s : %s" % (arr, arr.type)
        arr = None
        ptr = self.visit_expr(arr)
    else:    
      assert isinstance(arr, str), "Expected string repr of array but got %s" % arr
      if full_array:
        ptr = "%s.data.raw_ptr" % arr 
      else:
        ptr = arr 
        arr = None  
        
    if isinstance(idx, Expr):
      if isinstance(idx.type, TupleT):
        nelts = len(idx.type.elt_types)
        tuple_value = self.visit_expr(idx)
        idx = ["%s.elt%d" % (tuple_value,i) for i in xrange(nelts)]
      else:
        assert isinstance(idx.type, ScalarT), "Expected index %s to be scalar or tuple" % idx
        offset = idx 
        
    if isinstance(idx, str):
      offset = idx 
    elif isinstance(idx, (int,long)):
      offset = "%s" % idx
    elif isinstance(idx, (list,tuple)):
      assert arr, "Expected full array but only pointer %s is available" % ptr 
      offset = "0"
      for i,idx_elt in enumerate(idx):
        if isinstance(idx_elt, Expr):
          idx_elt = self.visit_expr(idx_elt)
        offset = self.add(offset, self.mul("%s.strides[%d]" % (arr,i), idx_elt))
      
    return "%s[%s]" % (ptr, offset)
  
  def setidx(self, arr, idx, value, full_array = False, return_stmt = False):
    stmt = "%s = %s;" % (self.getidx(arr, idx, full_array = full_array), value)
    if return_stmt:
      return stmt 
    else:
      self.append(stmt)

     
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
      elt_obj = self.fresh_var("PyObject*", "%s_elt" % tup, proj_str)
      result = self.unbox(elt_obj, t)
      if config.debug and t == Int64:
        self.append(""" printf("tupleproj %s[%d] = %%" PRId64 "\\n", %s);""" % (tup, idx, result))
      return result
    else:
      return "%s.elt%d" % (tup, idx)  
 
    
  def decref(self, obj):
    self.append("Py_DECREF(%s);" % obj)
  
  def get_boxed_array_ptr(self, v, parakeet_ptr_t):
    self.check_array(v)
    c_struct_type = self.to_ctype(parakeet_ptr_t)
    c_ptr_type = type_mappings.to_ctype(parakeet_ptr_t) 
    data_field = "(%s) (((PyArrayObject*) %s)->data)" % (c_ptr_type, v) 
    # get the data field but also fill the base object 
    return self.fresh_var(c_struct_type, "data", "{%s, %s}" % (data_field, v))
 
  def attribute(self, v, attr, t, boxed = False):
    if attr == "data":
      if boxed:
        self.check_array(v)
        struct_type = self.to_ctype(t)
        ptr_type = type_mappings.to_ctype(t) 
        data_field = "(%s) (((PyArrayObject*) %s)->data)" % (ptr_type, v) 
        # get the data field but also fill the base object 
        return self.fresh_var(struct_type, "data", "{%s, %s}" % (data_field, v))
      else:
        return "%s.data" % v  
    elif attr == "shape":
      shape_name = self.fresh_name("shape")
      elt_types = t.elt_types
      n = len(elt_types)
      elt_t = elt_types[0]  
      if boxed:
        self.check_array(v)
        
        assert all(t == elt_t for t in elt_types)

        shape_array = "PyArray_DIMS( (PyArrayObject*) %s)" % v
        self.append("npy_intp* %s = %s;" % (shape_name, shape_array))
      else:
        self.append("npy_intp* %s = %s.shape;" % (shape_name, v))
      return self.array_to_tuple(shape_name, n, elt_t)
      
    elif attr == "strides":
      elt_types = t.elt_types
      n = len(elt_types)  
      elt_t = elt_types[0]
      if boxed:
        assert False, "Can't directly use NumPy strides without dividing by itemsize"
      else:
        strides_name = self.fresh_var("npy_intp*", "strides", "%s.strides" % v)
      return self.array_to_tuple(strides_name, n, elt_t)
    elif attr == 'offset':
      if boxed:
        return "0"
      else:
        return "%s.offset" % v
    elif attr in ('size', 'nelts'):
      if boxed:
        return "PyArray_Size(%s)" % v
      else:
        return "%s.size" % v
       
    elif attr in ('start', 'stop', 'step'):
      if boxed:
        self.check_slice(v)
        obj = "((PySliceObject*)%s)->%s" % (v, attr)
        return self.unbox_scalar(obj, t, attr)
      else:
        return "%s.%s" % (v, attr)
    else:
      assert False, "Unsupported attribute %s" % attr 
   
    
  
  def alloc_array(self, array_t, shape_expr):
    if isinstance(shape_expr.type, ScalarT):
      dim = self.visit_expr(shape_expr)
      nelts = dim 
      shape_elts = [dim]
      ndims = 1 
    else:
      shape = self.visit_expr(shape_expr)
      ndims = array_t.rank
      shape_elts = ["%s.elt%d" % (shape, i) for i in xrange(ndims)]
      nelts = self.fresh_var("int64_t", "nelts", " * ".join(shape_elts))
    bytes_per_elt = array_t.elt_type.dtype.itemsize
    typename = self.to_ctype(array_t)
    result = self.fresh_var(typename, "new_array")
    raw_ptr_t = self.to_ctype(array_t.elt_type) + "*"
    self.setfield(result, "data.raw_ptr", "(%s) malloc(%s * %s)" % (raw_ptr_t, nelts, bytes_per_elt) )
    self.setfield(result, "data.base", "(PyObject*) NULL")
    self.setfield(result, "offset", "0")
    self.setfield(result, "size", nelts)
    # assume C-order layout
    strides_elts = [" * ".join(["1"] + shape_elts[(i+1):]) for i in xrange(ndims)]
    for i in xrange(ndims):
      self.setidx("%s.shape" % result, i, shape_elts[i])
      # assume c major layout for now
      self.setidx("%s.strides" % result, i, strides_elts[i])
    if config.debug:
      self.printf("[Debug] Done allocating array")
    return result 
    
  
  def visit_AllocArray(self, expr, boxed=False):
    if boxed:
      shape = self.tuple_to_stack_array(expr.shape)
      t = type_mappings.to_dtype(elt_type(expr.type))
      return "(PyArrayObject*) PyArray_SimpleNew(%d, %s, %s)" % (expr.type.rank, shape, t)
    
    if config.debug:
      print "[Debug] Allocating array : %s " % expr.type  
    return self.alloc_array(expr.type, expr.shape)
     
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
    typename = self.to_ctype(expr.type)
    data = self.visit_expr(expr.data)
    ndims = expr.type.rank 
    offset = self.visit_expr(expr.offset)
    count = self.visit_expr(expr.size)
    shape = self.visit_expr(expr.shape)
    strides = self.visit_expr(expr.strides)
  
    strides_elts = ["%s.elt%d" % (strides, i) for i in xrange(ndims)]
    shape_elts = ["%s.elt%d" % (shape, i) for i in xrange(ndims)]

    result = self.fresh_var(typename, "array_result")
    self.setfield(result, "data", data)
    self.setfield(result, "offset", offset)
    self.setfield(result, 'size', count)
    for i in xrange(ndims):
      self.setidx("%s.shape" % result, i, shape_elts[i])
      self.setidx("%s.strides" % result, i, strides_elts[i])
    return result 
  
  """
  def box_ArrayView(self, expr):
    data = self.visit_expr(expr.data)
    ndims = expr.type.rank 
    offset = self.visit_expr(expr.offset)
    count = self.visit_expr(expr.size)
    shape = self.visit_expr(expr.shape)
    strides = self.visit_expr(expr.strides)
  
    strides_elts = ["%s.elt%d" % (strides, i) for i in xrange(ndims)]
    shape_elts = ["%s.elt%d" % (shape, i) for i in xrange(ndims)]
    shape_array = None 
    strides_array = None 
    elt_type = expr.type.elt_type 
    return self.make_boxed_array(elt_type, ndims, data, strides_array, shape_array, offset, count)
  """
  
  def box_array(self, arr, t):
    elt_t = t.elt_type
    ndims = t.rank 
    data_ptr = "%s.data.raw_ptr" % arr 
    base = "%s.data.base" % arr
    strides_array = "%s.strides" % arr 
    shape_array = "%s.shape" % arr 
    offset = "%s.offset" % arr 
    size = "%s.size" % arr 
    return self.make_boxed_array(elt_t, ndims, data_ptr, strides_array, shape_array, offset, size, base)
  
  def make_boxed_array(self, elt_type, ndims, data_ptr, strides_array, shape_array, offset, size, base = "NULL"):
    
    
    # PyObject* PyArray_SimpleNewFromData(int nd, npy_intp* dims, int typenum, void* data)
    typenum = type_mappings.to_dtype(elt_type)
    array_alloc = \
      "(PyArrayObject*) PyArray_SimpleNewFromData(%d, %s, %s, &%s[%s])" % \
        (ndims, shape_array, typenum, data_ptr, offset)
    vec = self.fresh_var("PyArrayObject*", "fresh_array", array_alloc) 
    self.return_if_null(vec)
    
    # if the pointer had a PyObject reference, 
    # set that as the new array's base 
    if base not in ("0", "NULL"):
      self.append("""
        if (%s) { 
          %s->base = %s;
          Py_INCREF(%s);  
          %s->flags &= ~NPY_ARRAY_OWNDATA; 
        }""" % (base, vec, base, base, vec))
        
    numpy_strides = self.fresh_var("npy_intp*", "numpy_strides")
    self.append("%s = PyArray_STRIDES(  (PyArrayObject*) %s);" % (numpy_strides, vec))
    bytes_per_elt = elt_type.dtype.itemsize
    
    for i in xrange(ndims):
      self.append("%s[%d] = %s[%d] * %d;" % (numpy_strides, i, strides_array, i, bytes_per_elt) )
      
    
    self.append("""
      // clear both fortran and c layout flags 
      ((PyArrayObject*) %(vec)s)->flags &= ~NPY_F_CONTIGUOUS;
      ((PyArrayObject*) %(vec)s)->flags &= ~NPY_C_CONTIGUOUS;
    """ % locals())
    
    f_layout_strides = ["1"]
    for i in xrange(1, ndims):
      shape_elt = "%s[%d]" % (shape_array, i)
      f_layout_strides.append(f_layout_strides[-1] + " * " + shape_elt)
    
    c_layout_strides = ["1"]
    for i in xrange(ndims-1,0,-1):
      shape_elt = "%s[%d]" % (shape_array, i)
      c_layout_strides = [c_layout_strides[-1] + " * " + shape_elt] + c_layout_strides
    
    
    strides_elts = ["%s[%d]" % (strides_array, i) for i in xrange(ndims)]
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
    v = self.visit_expr(expr.value) 
    return self.attribute(v, attr, expr.type)
  
  def visit_Return(self, stmt):
    if self.module_entry:
      v = self.as_pyobj(stmt.value)
      if config.debug: 
        self.print_pyobj_type(v, "Return type: ")
        self.print_pyobj(v, "Return value: ")
      return "return (PyObject*) %s;" % v
    else:
      return FnCompiler.visit_Return(self, stmt)
  
  def visit_block(self, stmts, push = True):
    if push: self.push()
    for stmt in stmts:
      s = self.visit_stmt(stmt)
      self.append(s)
    self.append("\n")
    return self.pop()
  
  
  def enter_module_body(self):
    """
    Some derived compiler classes might want to use this hook
    to generate special code when entering the function body of the module entry
    """
    pass 
  
  def exit_module_body(self):
    pass 
  
  def visit_fn(self, fn):
    if config.print_input_ir:
      print "=== Compiling to C with %s (entry function) ===" % self.__class__.__name__ 
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
      
      if isinstance(t, (TupleT, NoneT, PtrT, ClosureT, ScalarT, ArrayT, SliceT)):
        #new_name = self.name(argname, overwrite = True)
        self.comment("Unboxing %s : %s" % (argname, t))
        var = self.unbox(c_name, t, target = argname)

        self.name_mappings[argname] = var
      

    self.enter_module_body()
    c_body = self.visit_block(fn.body, push=False)
    self.exit_module_body()
    c_body = self.indent(c_body )
    c_args = "PyObject* %s, PyObject* %s" % (dummy, args) #", ".join("PyObject* %s" % self.name(n) for n in fn.arg_names)
    c_sig = "PyObject* %(c_fn_name)s (%(c_args)s)" % locals() 
    fndef = "%s {\n\n %s}" % (c_sig, c_body)
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
                                 src_extension = self.src_extension,
                                 extra_objects = set(self.extra_objects),
                                 extra_function_sources = ordered_function_sources, 
                                 declarations =  self.declarations, 
                                 extra_compile_flags = self.extra_compile_flags, 
                                 extra_link_flags = self.extra_link_flags, 
                                 print_source = root_config.print_generated_code, 
                                 compiler = self.compiler_cmd, 
                                 compiler_flag_prefix = self.compiler_flag_prefix, 
                                 linker_flag_prefix = self.linker_flag_prefix)
    self._entry_compile_cache[key]  = compiled_fn
    return compiled_fn

