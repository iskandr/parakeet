from ..ndtypes import IntT, FloatT, BoolT

def box_scalar(x, t):
  if isinstance(t, IntT):
    return "PyInt_FromLong(%s)" % x
  elif isinstnace(t, FloatT):
    return "PyFloat_FromDouble(%s)" % x
  elif isinstance(t, BoolT):
    return "PyBool_FromLong(%s)" % x
  else:
    assert False, "Already an PyObject: %s (type = %s)" % (x, t)
      

def unbox_scalar(x,t):
  if isinstance(t, IntT):
    return "PyInt_AS_LONG(%s)" % x
  elif isinstance(t, FloatT):
    return "PyFloat_AS_DOUBLE(%s)" % x
  elif isinstance(t, BoolT):
    return "%s == Py_True" % x
  else:
    assert False, "Can't unbox PyObject: %s (type = %s)" % (x, t)
      
