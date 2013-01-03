import escape_analysis

from array_type import ArrayT
from collect_vars import collect_var_names
from core_types import ScalarT  
from find_local_arrays import FindLocalArrays
from syntax import Index, Var
from syntax import ArrayView, Struct
from transform import Transform
from usedef import UseDefAnalysis 

class CopyElimination(Transform):
  def apply(self, fn):
    if all(isinstance(t, ScalarT) for t in fn.type_env.itervalues()):
      return fn 
    else:
      return Transform.apply(self, fn)  
  
  def pre_apply(self, fn):
    find_local_arrays = FindLocalArrays()
    find_local_arrays.visit_fn(fn)

    self.local_alloc = find_local_arrays.local_allocs
    self.local_arrays = find_local_arrays.local_arrays

    escape_info = escape_analysis.run(fn)
    self.may_escape = escape_info.may_escape
    self.may_alias = escape_info.may_alias 

    self.usedef = UseDefAnalysis()
    self.usedef.visit_fn(fn)

  def transform_Assign(self, stmt):
    # pattern match only on statements of the form
    # dest[complex_indexing] = src
    # when:
    #   1) dest hasn't been used before as a value
    #   2) src doesn't escape
    #   3) src was locally allocated
    # ...then transform the code so instead of allocating src

    if stmt.lhs.__class__ is Index and  stmt.lhs.value.__class__ is Var:
      lhs_name = stmt.lhs.value.name
      if lhs_name not in self.usedef.first_use and \
         lhs_name not in self.may_escape and \
         len(self.may_alias.get(lhs_name, [])) <= 1:
        # why assign to an array if it never gets used?
        return None
      elif stmt.lhs.type.__class__ is ArrayT and stmt.rhs.__class__ is Var:
        curr_stmt_number = self.usedef.stmt_number[id(stmt)]
        rhs_name = stmt.rhs.name

        if self.usedef.last_use[rhs_name] == curr_stmt_number and \
           self.usedef.first_use[lhs_name] > curr_stmt_number and \
           rhs_name not in self.may_escape and \
           rhs_name in self.local_arrays:
          array_stmt = self.local_arrays[rhs_name]
          prev_stmt_number = self.usedef.stmt_number[id(array_stmt)]
          if array_stmt.rhs.__class__ in (Struct, ArrayView) and \
             all(self.usedef.created_on[lhs_depends_on] < prev_stmt_number
                 for lhs_depends_on in collect_var_names(stmt.lhs)):
            array_stmt.rhs = stmt.lhs
            return None
    return stmt
