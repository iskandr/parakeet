import core_types 
from core_types import Int64 as i64
  
class cfn:
  def __init__(self, name, arg_types, return_type):
    self.name = name 
    self.arg_types = arg_types 
    self.return_type = return_type 
    
pyobj = None 
c_str = None 

class PyDict:
  GetItem = cfn("PyDict_GetItem", [pyobj, pyobj], pyobj)
  GetItemString = cfn("PyDict_GetItemString", [pyobj, c_str], pyobj)
  SetItem = cfn("PyDict_SetItem", [pyobj, pyobj, pyobj], i64)
  SetItemString = cfn("PyDict_SetItemString", [pyobj, c_str,pyobj], i64)

class PyLong:
  pass 



"""  
("PyLong_AsDouble", f64type, [pyobj], []),
("PyLong_AsLong", i64type, [pyobj], []),
("PyLong_AsLongAndOverflow", i64type, [pyobj,i64type], []),
("PyLong_AsLongLong", i64type, [pyobj], []),
("PyLong_AsLongLongAndOverflow", i64type, [pyobj,i64type], []),
("PyLong_AsSsize_t", i64type, [pyobj], []),
("PyLong_AsUnsignedLong", i64type, [pyobj], []),
("PyLong_AsUnsignedLongLong", i64type, [pyobj], []),
("PyLong_AsUnsignedLongLongMask", i64type, [pyobj], []),
("PyLong_AsUnsignedLongMask", i64type, [pyobj], []),
("PyLong_AsVoidPtr", i8ptrtype, [pyobj], []),
("PyLong_FromDouble", pyobj, [f64type], []),
("PyLong_FromLong", pyobj, [i64type], []),
("PyLong_FromLongLong", pyobj, [i64type], []),
("PyLong_FromSize_t", pyobj, [i64type], []),
("PyLong_FromSsize_t", pyobj, [i64type], []),
("PyLong_FromString", pyobj, [i8ptrtype,i8ptrtype,i64type], []),
("PyLong_FromUnsignedLong", pyobj, [i64type], []),
("PyLong_FromUnsignedLongLong", pyobj, [i64type], []),
("PyLong_FromVoidPtr", pyobj, [i8ptrtype], []),
"""