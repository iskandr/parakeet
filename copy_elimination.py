import names

import syntax
from syntax import Assign, Index, Slice, Var, Return, RunExpr
from syntax import ArrayView,  Alloc, Array, Struct, Tuple, Attribute
from syntax_helpers import none, slice_none, zero_i64

from adverb_helpers import max_rank
from adverbs import Map, Reduce, Scan, AllPairs, Adverb

from core_types import NoneT, NoneType
import array_type
from array_type import ArrayT

import shape_inference
import shape_codegen

from collect_vars import collect_var_names, collect_var_names_from_exprs
from transform import Transform
from clone_function import CloneFunction

from syntax_visitor import SyntaxVisitor
from compiler.ast import Stmt

class FindLocalArrays(SyntaxVisitor):
  def __init__(self):
    # hash table mapping from variable names to
    # statements allocating space
    self.local_allocs = {}

    # hash table mapping from variable names to
    # places where we create array views containing
    # locally allocated data pointers
    self.local_arrays = {}

  def visit_Assign(self, stmt):
    if stmt.lhs.__class__ is Var:
      rhs_class = stmt.rhs.__class__
      if rhs_class is Alloc:
        self.local_allocs[stmt.lhs.name] = stmt
      elif rhs_class is ArrayView and \
          stmt.rhs.data.__class__ is Var and \
          stmt.rhs.data.name in self.local_allocs:
        self.local_arrays[stmt.lhs.name] = stmt
      elif rhs_class is Struct and \
          stmt.rhs.type.__class__ is ArrayT and \
          stmt.rhs.args[0].__class__ is Var and \
          stmt.rhs.args[0].name in self.local_allocs:
        self.local_arrays[stmt.lhs.name] = stmt
      elif rhs_class is Array:
        self.local_arrays[stmt.lhs.name] = stmt

empty = set([])

class EscapeAnalysis(SyntaxVisitor):
  def visit_fn(self, fn):
    self.may_alias = {}
    # every name at least aliases it self
    for name in fn.type_env.iterkeys():
      self.may_alias[name] = set([name])

    self.may_escape = set(fn.arg_names)
    self.visit_block(fn.body)

  def mark_escape(self, name):
    for alias in self.may_alias[name]:
      self.may_escape.add(alias)

  def mark_escape_list(self, names):
    for name in names:
      self.mark_escape(name)

  def visit_Call(self, expr):
    self.mark_escape_list(collect_var_names_from_exprs(expr.args))

  def collect_lhs_names(self, expr):
    if expr.__class__ is Var:
      return [expr.name]
    elif expr.__class__ is Attribute:
      return self.collect_lhs_names(expr.value)
    elif expr.__class__ is Tuple:
      combined = set([])
      for elt in expr.elts:
        combined.update(self.collect_lhs_names(elt))
    else:
      return []

  def visit_Assign(self, stmt):
    lhs_names = self.collect_lhs_names(stmt.lhs)
    rhs_names = collect_var_names(stmt.rhs)
    for lhs_name in lhs_names:
      self.may_alias[lhs_name].update(rhs_names)

  def visit_Return(self, expr):
    self.mark_escape_list(collect_var_names(expr.value))

  def visit_merge(self, merge):
    for (name, (l,r)) in merge.iteritems():
      left_aliases = self.may_alias[l.name] if l.__class__ is Var else empty
      right_aliases = self.may_alias[r.name] if r.__class__ is Var else empty
      combined = left_aliases.union(right_aliases)
      self.may_alias[name].update(combined)
      if any(alias in self.may_escape for alias in combined):
        self.may_escape.update(combined)
        self.may_escape.add(name)

class UseDefAnalysis(SyntaxVisitor):
  """
  Number all the statements and track the
  creation as well as first and last uses of variables
  """
  def __init__(self):
    # map from pointers of statement objects to
    # sequential numbering
    # where start is the current statement
    self.stmt_number = {}

    # ..and also track the range of nested statement numbers
    self.stmt_number_end = {}

    self.stmt_counter = 0
    # map from variable names to counter number of their
    # first/last usages
    self.first_use = {}
    self.last_use = {}
    self.created_on = {}

  def visit_fn(self, fn):
    for name in fn.arg_names:
      self.created_on[name] = 0
    SyntaxVisitor.visit_fn(self, fn)

  def visit_lhs(self, expr):
    if expr.__class__ is Var:
      self.created_on[expr.name] = self.stmt_counter
    elif expr.__class__ is Tuple:
      for elt in expr.elts:
        self.visit_lhs(elt)

  def visit_If(self, stmt):
    for name in stmt.merge.iterkeys():
      self.created_on[name] = self.stmt_counter
    SyntaxVisitor.visit_If(self, stmt)

  def visit_While(self, stmt):
    for name in stmt.merge.iterkeys():
      self.created_on[name] = self.stmt_counter
    SyntaxVisitor.visit_While(self, stmt)

  def visit_Var(self, expr):
    name = expr.name
    if name not in self.first_use:
      self.first_use[name] = self.stmt_counter
    self.last_use[name]= self.stmt_counter

  def visit_stmt(self, stmt):
    stmt_id = id(stmt)
    self.stmt_counter += 1
    count = self.stmt_counter
    self.stmt_number[stmt_id] = count
    SyntaxVisitor.visit_stmt(self, stmt)
    self.stmt_number_end[stmt_id] = self.stmt_counter

class CopyElimination(Transform):
  def pre_apply(self, fn):
    find_local_arrays = FindLocalArrays()
    find_local_arrays.visit_fn(fn)

    self.local_alloc = find_local_arrays.local_allocs
    self.local_arrays = find_local_arrays.local_arrays

    escape_analysis = EscapeAnalysis()
    escape_analysis.visit_fn(fn)

    self.may_escape = escape_analysis.may_escape

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
         lhs_name not in self.may_escape:
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
