import ptype 

_flat_fns = {}

def flat_repr(t):
  if isinstance(t, ptype.ScalarT):
    return t
  elif isinstance(t, ptype.PtrT):
    return ptype.ptr(flat_repr(t.elt_type))
  elif isinstance(t, ptype.TupleT):
    return map(flat_repr, t.elt_types)
  else:
    


    
    


