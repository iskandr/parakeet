

  def from_python(self, python_fn):
    import ast_conversion
    untyped_fundef = ast_conversion.translate_function_value(python_fn)
    closure_args = untyped_fundef.python_nonlocals()
    closure_arg_types = map(type_conv.typeof, closure_args)

    closure_t = make_closure_type(untyped_fundef, closure_arg_types)
    closure_id = id_of_closure_type(closure_t)

    def field_value(closure_arg):
      obj = type_conv.from_python(closure_arg)
      parakeet_type = type_conv.typeof(closure_arg)
      if isinstance(parakeet_type, StructT):
        return ctypes.pointer(obj)
      else:
        return obj

    converted_args = [field_value(closure_arg) for closure_arg in closure_args]
    return closure_t.ctypes_repr(closure_id, *converted_args)
import config

def print_specializations():
  if config.print_specialized_function_names:
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