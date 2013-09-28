 
from ..config import print_specialized_function_names
from ..ndtypes.closure_type import _closure_type_cache

def clear_specializations():   
  for clos_t in _closure_type_cache.itervalues():
    clos_t.specializations.clear()


def print_specializations():

  if print_specialized_function_names:
    print
    print "FUNCTION SPECIALIZATIONS"
    count = 0
    for ((untyped,closure_types), clos_t) in sorted(_closure_type_cache.items()):
      specializations = clos_t.specializations.items()
      if len(specializations) > 0:
        name = untyped.name if hasattr(untyped, 'name') else str(untyped)
        print "Closure %s %s" % (name, closure_types)
        for (arg_types, typed_fn) in sorted(specializations):
          print "  -- %s ==> %s" % (arg_types, typed_fn.name)
          count += 1
    print
    print "Total: %d function specializations" % count


import atexit
atexit.register(print_specializations)