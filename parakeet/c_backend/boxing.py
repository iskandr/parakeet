from ..ndtypes import IntT, FloatT, BoolT

def box_scalar(x, t):
  if isinstance(t, BoolT):
    return "PyBool_FromLong(%s)" % x
  elif isinstance(t, IntT):
    return "PyInt_FromLong(%s)" % x
  elif isinstance(t, FloatT):
    return "PyFloat_FromDouble(%s)" % x
  else:
    assert False, "Already an PyObject: %s (type = %s)" % (x, t)
      

def unbox_scalar(x,t):
  if isinstance(t, IntT):
    return "PyInt_AsLong(%s)" % x
  elif isinstance(t, FloatT):
    return "PyFloat_AsDouble(%s)" % x
  elif isinstance(t, BoolT):
    return "%s == Py_True" % x
  else:
    assert False, "Can't unbox PyObject: %s (type = %s)" % (x, t)
      
