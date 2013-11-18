from .. ndtypes import ScalarT, ArrayT

from .. analysis.collect_vars import collect_var_names
from .. analysis import escape_analysis
from .. analysis.find_local_arrays import FindLocalArrays
from .. analysis.usedef import UseDefAnalysis
from .. syntax import AllocArray, ArrayView, Index, Struct, Var

from .. syntax import Adverb
from transform import Transform

class CopyElimination(Transform):
  def apply(self, fn):
    if all(isinstance(t, ScalarT) for t in fn.type_env.itervalues()):
      return fn
    else:
      return Transform.apply(self, fn)

  def pre_apply(self, fn):
    local_array_analysis = FindLocalArrays()
    local_array_analysis.visit_fn(fn)


    self.local_alloc = local_array_analysis.local_allocs
    self.local_arrays = local_array_analysis.local_arrays

    escape_info = escape_analysis(fn)
    self.may_escape = escape_info.may_escape
    self.may_alias = escape_info.may_alias
    self.may_return = escape_info.may_return
    
    self.usedef = UseDefAnalysis()
    self.usedef.visit_fn(fn)

    
    self.pointers_by_size = {}
    self.arrays_by_size = {}
    

  def no_array_aliases(self, array_name):
    alias_set = self.may_alias.get(array_name, [])
    array_aliases = [name for name in alias_set
                     if self.type_env[name].__class__ is ArrayT]
    # you're allowed one alias for yourself, but
    # any extras are other arrays with whom you share data
    # BEWARE: this will get convoluted and probably broken
    # if we ever have mutable compound objects in arrays
    return len(array_aliases) <= 1

  def is_array_alloc(self, expr):
    return expr.__class__ in [ArrayView, Struct, AllocArray] or isinstance(expr, Adverb)
    
  def transform_Assign(self, stmt):
    # pattern match only on statements of the form
    # dest[complex_indexing] = src
    # when:
    #   1) dest hasn't been used before as a value
    #   2) src doesn't escape
    #   3) src was locally allocated
    # ...then transform the code so instead of allocating src

    if stmt.lhs.__class__ is not Index or stmt.lhs.value.__class__ is not Var:
      return stmt 
    
    lhs_name = stmt.lhs.value.name

    # why assign to an array if it never gets used?
    if lhs_name not in self.usedef.first_use and \
       lhs_name not in self.may_escape and \
       self.no_array_aliases(lhs_name):
      return None
    
    # only match statements like array[idx] = some_rhs_var
    if stmt.lhs.type.__class__ is not ArrayT or stmt.rhs.__class__ is not Var:
      return stmt 
    
    curr_path = self.usedef.stmt_paths[id(stmt)]
    rhs_name = stmt.rhs.name

    
    if lhs_name in self.usedef.first_use and self.usedef.first_use[lhs_name] <= curr_path:
      return stmt 
    
    if self.usedef.last_use[rhs_name] != curr_path:
      return stmt
    
    if rhs_name in self.may_return:
      return stmt  
    
    if rhs_name not in self.local_arrays:
      return stmt 
            
    array_stmt = self.local_arrays[rhs_name]
    prev_path = self.usedef.stmt_paths[id(array_stmt)]
    if not self.is_array_alloc(array_stmt.rhs):
      return stmt 
    
    for lhs_depends_on in collect_var_names(stmt.lhs):
      created_on = self.usedef.created_on.get(lhs_depends_on)
      if created_on is None or created_on >= prev_path:
        return stmt 
             
    array_stmt.rhs = stmt.lhs
    return None
    