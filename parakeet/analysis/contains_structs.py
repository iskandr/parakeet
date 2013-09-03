from ..ndtypes import ScalarT, PtrT

def contains_structs(fn):
  for n, t in fn.type_env.iteritems():
    if not isinstance(t, (ScalarT, PtrT)):
      return True
  return False