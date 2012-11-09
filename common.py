import sys
def dispatch(node, prefix = "",  default = None, locals_dict = None):
  if locals_dict is None:
    last_frame = sys._getframe()
    locals_dict = last_frame.f_back.f_locals
  node_type = node.__class__.__name__
  if len(prefix) > 0:
    fn_name = prefix + "_" + node_type
  else:
    fn_name = node_type

  if fn_name in locals_dict:
    return locals_dict[fn_name]()
  elif default:
    return default(node)
  else:
    available = [k.split(prefix + "_")[1] for k in locals_dict.keys() if k.startswith(prefix + "_")]
    raise RuntimeError("Unsupported node %s with type %s, available: %s" % (node, node_type, available))

import ctypes
def list_to_ctypes_array(ctypes_object_list, elt_type = None, pointers = False):
  if elt_type is None:
    elt = ctypes_object_list[0]
    elt_type = elt.__class__
  if pointers:
    elt_type = ctypes.POINTER(elt_type)
  array_t = elt_type * len(ctypes_object_list)
  array_object = array_t()
  for (i, x) in enumerate(ctypes_object_list):
    array_object[i] = ctypes.pointer(x) if pointers else x
  return array_object
