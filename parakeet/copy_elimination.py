import escape_analysis

from adverbs import Map, Reduce, Scan, AllPairs
from array_type import ArrayT
from collect_vars import collect_var_names
from core_types import ScalarT
from find_local_arrays import FindLocalArrays
from syntax import AllocArray, ArrayView, Index, Struct, Var
from transform import Transform
from usedef import UseDefAnalysis

array_constructors = [ArrayView, Struct, AllocArray,
                      Map, Reduce, Scan, AllPairs]

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
         self.no_array_aliases(lhs_name):
        # why assign to an array if it never gets used?
        return None
      elif stmt.lhs.type.__class__ is ArrayT and stmt.rhs.__class__ is Var:
        curr_path = self.usedef.stmt_paths[id(stmt)]
        rhs_name = stmt.rhs.name
        if lhs_name not in self.usedef.first_use or \
           self.usedef.first_use[lhs_name] > curr_path:
          if self.usedef.last_use[rhs_name] == curr_path and \
             rhs_name not in self.may_escape and \
             rhs_name in self.local_arrays:
            array_stmt = self.local_arrays[rhs_name]
            prev_path = self.usedef.stmt_paths[id(array_stmt)]
            if array_stmt.rhs.__class__ in array_constructors and \
               all(self.usedef.created_on[lhs_depends_on] < prev_path
                   for lhs_depends_on in collect_var_names(stmt.lhs)):
              array_stmt.rhs = stmt.lhs
              return None
    """
    elif stmt.lhs.__class__ is Var and stmt.rhs.__class__ is Alloc:
      lhs_name = stmt.lhs.name
      curr_path = self.usedef.stmt_number[id(stmt)]
      print stmt
      for prev_name in self.pointers_by_size.get(stmt.rhs.count, []):
        if self.type_env[prev_name] == self.type_env[lhs_name] and \
           prev_name in self.local_alloc and \
           lhs_name in self.local_alloc and \
           self.usedef.last_use[prev_name] <  curr_stmt_number:
          stmt.rhs = Var(prev_name, type = self.type_env[prev_name])
          self.usedef.last_use[prev_name] = curr_stmt_number
          return stmt
      if lhs_name not in self.may_escape and \
         len(self.may_alias.get(lhs_name, [])) <= 1:
        alloc_set = self.pointers_by_size.setdefault(stmt.rhs.count, set([]))
        alloc_set.add(lhs_name)
    """
    return stmt
