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
from transform import MemoizedTransform, apply_pipeline, Transform
from clone_function import CloneFunction

from syntax_visitor import SyntaxVisitor

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

class UseAnalysis(SyntaxVisitor):
  """
  Number all the statements and track the
  first and last uses of variables
  """
  def __init__(self, lhs_vars = False):
    # include uses of a variable on the LHS of an assignment?

    self.lhs_vars = lhs_vars

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

  def visit_lhs(self, lhs):
    if self.lhs_vars:
      SyntaxVisitor.visit_lhs(lhs)

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

    self.use_analysis = UseAnalysis()
    self.use_analysis.visit_fn(fn)

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
      if stmt.lhs.type.__class__ is ArrayT and stmt.rhs.__class__ is Var:
      
        stmt_number = self.use_analysis.stmt_number[id(stmt)]
        rhs_name = stmt.rhs.name
        print "STMT_NUMBER", stmt_number
        print "LHS NAME", lhs_name 
        print "RHS NAME", rhs_name  
        print "last use rhs", self.use_analysis.last_use[rhs_name]
        print "first_use lhs", self.use_analysis.first_use[lhs_name]
        print "rhs may escape?", rhs_name  in self.may_escape
        print "rhs is local array?", rhs_name in self.local_arrays
        if self.use_analysis.last_use[rhs_name] == stmt_number and \
            self.use_analysis.first_use[lhs_name] > stmt_number and \
            rhs_name not in self.may_escape and \
            rhs_name in self.local_arrays:
          array_stmt = self.local_arrays[rhs_name]
          print "array stmt", array_stmt 
          if array_stmt.rhs.__class__ in (Struct, ArrayView):
            print "UPDATING", array_stmt
            array_stmt.rhs = stmt.lhs
            return None
      elif lhs_name not in self.use_analysis.first_use and \
          lhs_name not in self.may_escape:
        # why assign to an array if it never gets used? 
        return None 


class PreallocAdverbOutput(MemoizedTransform):
  def niters(self, args, axis):
    max_rank = -1
    biggest_arg = None
    for arg in args:
      r = self.rank(arg)
      if r > max_rank:
        max_rank = r
        biggest_arg = arg
    return self.shape(biggest_arg, axis)


  def call_shape(self, maybe_clos, args):
    fn = self.get_fn(maybe_clos)
    abstract_shape = shape_inference.call_shape_expr(fn)
    closure_args = self.closure_elts(maybe_clos)
    combined_args = closure_args + args
    return shape_codegen.make_shape_expr(self, abstract_shape, combined_args)

  def map_elt_shape(self, maybe_clos, args, axis):
    nested_args = [self.index_along_axis(arg, axis, zero_i64) for arg in args]
    return self.call_shape(maybe_clos, nested_args)

  def map_shape(self, maybe_clos, args, axis):
    niters = self.niters(args, axis)
    inner_shape = self.map_elt_shape(maybe_clos, args, axis)
    return self.concat_tuples(niters, inner_shape)

  def transform_Call(self, expr):
    fn = self.get_fn(expr.fn)
    t = fn.return_type
    if isinstance(t, ArrayT):
      closure_args = self.closure_elts(expr.fn)
      combined_args = tuple(closure_args) + tuple(expr.args)
      try:
        result_shape = self.call_shape(fn, combined_args)
        out = self.alloc_array(t.elt_type, result_shape)
        expr.fn = preallocate_function_output(fn)
        expr.args = combined_args + (out,)
        return expr
      except:
        print "Caught error while transforming call allocations"
    expr.fn = preallocate_local_arrays(fn)
    return expr

  def transform_Map(self, expr):
    if expr.out is None and isinstance(expr.type, ArrayT) and expr.type.rank > 1:

      # transform the function to take an additional output parameter
      output_shape = self.map_shape(expr.fn, expr.args, expr.axis)
      return_t = self.return_type(expr.fn)
      elt_t = return_t.elt_type if hasattr(return_t, 'elt_type') else return_t

      fn = preallocate_function_output(self.get_fn(expr.fn))
      expr.out = self.alloc_array(elt_t, output_shape)

      closure_args = self.closure_elts(expr.fn)
      if len(closure_args) > 0:
        fn = self.closure(fn, closure_args, name = "closure")
      expr.fn  = fn
      expr.type = NoneType
      self.blocks.append_to_current(RunExpr(expr))
      return expr.out

  def reduce_shape(self, maybe_clos, args, axis):
    return self.map_elt_shape(maybe_clos, args, axis)
  """
  def transform_Reduce(self, expr):
    if expr.out is None:
      # transform the function to take an additional output parameter
      acc_shape = self.map_shape(expr.fn, expr.args, expr.axis)
      return_t = self.return_type(expr.fn)
      elt_t = return_t.elt_type if hasattr(return_t, 'elt_type') else return_t
      expr.out = self.alloc_array(elt_t, acc_shape)
      combine = make_output_storage_explicit(self.get_fn(expr.combine))
      closure_args = self.closure_elts(expr.combine)
      if len(closure_args) > 0:
        combine = self.closure(combine, closure_args, name = "closure")
      expr.combine  = combine
      expr.type = NoneType
      self.blocks.append_to_current(RunExpr(expr))
      return expr.out
  """


class PreallocFnOutput(PreallocAdverbOutput):
  """
  Transform a function so that instead of returning a value
  it writes to an additional output parameter.
  Should only be used as part of a recursive transform
  by the parent class 'PreallocateAdverbOutputs'.
  """
  def pre_apply(self, fn):
    output_name = names.fresh("output")
    output_t = fn.return_type

    if output_t.__class__ is ArrayT and output_t.rank > 0:
      idx = self.tuple([slice_none] * output_t.rank)
      elt_t = output_t

    else:
      idx = zero_i64
      elt_t = output_t
      output_t = array_type.increase_rank(output_t, 1)


    self.output_type = output_t
    output_var = Var(output_name, type=output_t)
    self.output_lhs_index = Index(output_var, idx, type = elt_t)

    fn.return_type = NoneType
    fn.arg_names = tuple(fn.arg_names) + (output_name,)
    fn.input_types = tuple(fn.input_types) + (output_t,)
    fn.type_env[output_name]  = output_t
    self.return_none = Return(none)
    return fn

  def transform_Return(self, stmt):
    new_value = self.transform_expr(stmt.value)
    self.assign(self.output_lhs_index, new_value)
    return self.return_none

def preallocate_local_arrays(fn):
  pipeline = [CloneFunction, PreallocAdverbOutput]
  return apply_pipeline(fn, pipeline)

def preallocate_function_output(fn):
  assert fn.__class__ is syntax.TypedFn, \
      "Can't transform expression %s, expected a typed function" % fn
  pipeline = [CloneFunction, PreallocFnOutput]
  return apply_pipeline(fn, pipeline)

