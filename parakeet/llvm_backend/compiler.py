
import llvm.core as llc 
from llvm.core import Builder, ATTR_NO_CAPTURE #, Module 
from llvm.core import Type as lltype

from .. import config, prims, syntax 
from .. analysis import may_escape
from .. ndtypes import BoolT, FloatT, SignedT, UnsignedT, ScalarT, NoneT, Float64
from .. ndtypes import Int32, Int64, PtrT, Bool 
from .. syntax import Var, Struct, Index, TypedFn, Attribute 

import llvm_config 
import llvm_context
import llvm_convert
import llvm_types
import llvm_prims
from llvm_helpers import const, int32, zero 
from llvm_types import llvm_value_type, llvm_ref_type
from parakeet.llvm_backend.llvm_convert import to_bit, from_bit


_escape_analysis_cache = {}
class Compiler(object):
  def __init__(self, fundef, llvm_cxt = llvm_context.global_context):
    self.parakeet_fundef = fundef
    if config.opt_stack_allocation:
      self.may_escape = may_escape(fundef)
    else:
      self.may_escape = None
    self.llvm_context = llvm_cxt
    self.vars = {}
    self.initialized = set([])
    # Initializes the variables dictionary and returns a builder object
    llvm_input_types = map(llvm_ref_type, fundef.input_types)
    llvm_output_type = llvm_ref_type(fundef.return_type)
    llvm_fn_t = lltype.function(llvm_output_type, llvm_input_types)

    self.llvm_fn = self.llvm_context.module.add_function(llvm_fn_t, fundef.name)

    for arg in self.llvm_fn.args:
      if not llvm_types.is_scalar(arg.type):
        arg.add_attribute(ATTR_NO_CAPTURE)

    self.llvm_fn.does_not_throw = True

    self.entry_block, self.entry_builder = self.new_block("entry")
    self._init_vars(self.parakeet_fundef, self.entry_builder)

  def new_block(self, name):
    bb = self.llvm_fn.append_basic_block(name)
    builder = Builder.new(bb)
    return bb, builder

  def _init_vars(self, fundef, builder):
    """
    Create a mapping from variable names to stack locations, these will later be
    converted to SSA variables by the mem2reg pass.
    """
    n_expected = len(fundef.arg_names)
    n_compiled = len(self.llvm_fn.args)
    assert n_compiled == n_expected, \
        "Expected %d args (%s) but compiled code had %d args (%s)" % \
        (n_expected, fundef.arg_names, n_compiled, self.llvm_fn.args)

    for (name, t) in fundef.type_env.iteritems():
      if not name.startswith("$"):
        llvm_t = llvm_ref_type(t)
        stack_val = builder.alloca(llvm_t, name)
        self.vars[name] = stack_val

    for llvm_arg, name in zip(self.llvm_fn.args, fundef.arg_names):
      self.initialized.add(name)
      llvm_arg.name = name
      if name in self.vars:
        builder.store(llvm_arg, self.vars[name])

  def attribute_lookup(self, struct, name, builder):
    """
    Helper for getting the address of an attribute lookup, used both when
    setting and getting attributes
    """
    llvm_struct = self.compile_expr(struct, builder)
    struct_t = struct.type
    field_pos = struct_t.field_pos(name)
    field_type = struct_t.field_type(name)
    indices = [int32(0), int32(field_pos)]
    ptr_name = "%s_ptr" % name
    ptr = builder.gep(llvm_struct, indices, ptr_name)
    return ptr, field_type

  def compile_Var(self, expr, builder):
    name = expr.name
    assert name in self.initialized, "%s uninitialized" % name
    ref = self.vars[expr.name]
    val = builder.load(ref, expr.name + "_val")
    return val

  def compile_Const(self, expr, builder):
    t = expr.type

    if isinstance(t, NoneT):
      return const(0, Int64)
    else:
      assert isinstance(expr.type, ScalarT), \
          "Expected scalar constant but got %s" % expr.type
    return const(expr.value, expr.type)

  def compile_Cast(self, expr, builder):
    llvm_value = self.compile_expr(expr.value, builder)
    return llvm_convert.convert(llvm_value, expr.value.type, expr.type, builder)

  def compile_Struct(self, expr, builder, local = False):
    struct_t = expr.type
    llvm_struct_t = llvm_value_type(struct_t)
    name = expr.type.node_type()
    if local:
      struct_ptr = builder.alloca(llvm_struct_t, name + "_local_ptr")
    else:
      struct_ptr = builder.malloc(llvm_struct_t, name + "_ptr")

    for (i, elt) in enumerate(expr.args):
      field_name, field_type = struct_t._fields_[i]
      assert elt.type == field_type, \
          "Mismatch between expected type %s and given %s for field '%s' " % \
          (field_type, elt.type, field_name)
      elt_ptr = builder.gep(struct_ptr, [int32(0), int32(i)], "field%d_ptr" % i)
      llvm_elt = self.compile_expr(elt, builder)
      builder.store(llvm_elt, elt_ptr)

    return struct_ptr

  def compile_Alloc(self, expr, builder):
    elt_t = expr.elt_type
    llvm_elt_t = llvm_types.llvm_value_type(elt_t)
    n_elts = self.compile_expr(expr.count, builder)
    return builder.malloc_array(llvm_elt_t, n_elts, "data_ptr")

  def compile_Index(self, expr, builder):
    llvm_arr = self.compile_expr(expr.value, builder)
    llvm_index = self.compile_expr(expr.index, builder)
    pointer = builder.gep(llvm_arr, [llvm_index], "elt_pointer")
    elt = builder.load(pointer, "elt", align = 16) #, invariant = True)
    return elt
  
  def compile_Select(self, expr, builder):
    cond = self.compile_expr(expr.cond, builder)
    cond = llvm_convert.to_bit(cond, builder)
    trueval = self.compile_expr(expr.true_value, builder)
    falseval = self.compile_expr(expr.false_value, builder)
    result =  builder.select(cond, trueval, falseval, "select_result")
    return result 
  
  def compile_Attribute(self, expr, builder):
    field_ptr, _ = \
        self.attribute_lookup(expr.value, expr.name, builder)
    field_value = builder.load(field_ptr, "%s_value" % expr.name)
    return field_value

  def compile_TypedFn(self, expr, builder):
    (target_fn, _, _) = compile_fn(expr)
    return target_fn

  def compile_Call(self, expr, builder):
    assert expr.fn.__class__ is TypedFn
    typed_fundef = expr.fn
    (target_fn, _, _) = compile_fn(typed_fundef)
    arg_types = syntax.get_types(expr.args)
    llvm_args = [self.compile_expr(arg, builder) for arg in expr.args]
    assert len(arg_types) == len(llvm_args)
    return builder.call(target_fn, llvm_args, 'call_result')

  def cmp(self, prim, t, llvm_x, llvm_y, builder, result_name = None):
    if result_name is None:
      result_name = prim.name + "_result"

    if isinstance(t, FloatT):
      cmp_op = llvm_prims.float_comparisons[prim]
      return builder.fcmp(cmp_op, llvm_x, llvm_y, result_name)
    elif isinstance(t, SignedT):
      cmp_op = llvm_prims.signed_int_comparisons[prim]
      return builder.icmp(cmp_op, llvm_x, llvm_y, result_name)
    else:
      assert isinstance(t, (BoolT, UnsignedT)), "Unexpected type for comparison %s: %s" % (prim, t)
      cmp_op = llvm_prims.unsigned_int_comparisons[prim]
      return builder.icmp(cmp_op, llvm_x, llvm_y, result_name)

  
  def lt(self, t, llvm_x, llvm_y, builder, result_name = None ):
    return self.cmp(prims.less, t, llvm_x, llvm_y, builder, result_name)
  
  def lte(self, t, llvm_x, llvm_y, builder, result_name = None ):
    return self.cmp(prims.less_equal, t, llvm_x, llvm_y, builder, result_name)
  
  def neq(self, t, llvm_x, llvm_y, builder, result_name = None):
    return self.cmp(prims.not_equal, t, llvm_x, llvm_y, builder, result_name)
  
  def sub(self, t, x, y, builder, result_name = "sub"):
    if isinstance(t, FloatT):
      return builder.fsub(x, y, result_name)
    else:
      return builder.sub(x, y, result_name)
    
  
  def add(self, t, x, y, builder, result_name = "add"):
    if isinstance(t, FloatT):
      return builder.fadd(x, y, result_name)
    else:
      return builder.add(x, y, result_name)
  
  
  def mul(self, t, x, y, builder, result_name = "mul"):
    if isinstance(t, FloatT):
      return builder.fmul(x, y, result_name)
    else:
      return builder.mul(x, y, result_name)
  
    
  def neg(self, x, builder):
    if isinstance(x.type, llc.IntegerType):
      return builder.neg(x, "neg")
    else:
      return builder.fsub(zero(x.type), x, "neg") 
    
  def prim(self, prim, t, llvm_args, builder, result_name = None):
    if result_name is None:
      result_name = prim.name + "_result"

    if isinstance(prim, prims.Cmp):
      bit = self.cmp(prim, t, llvm_args[0], llvm_args[1], builder)
      return llvm_convert.to_bool(bit,builder)
    
    elif prim == prims.maximum:
      x, y = llvm_args
      bit = self.cmp(prims.greater_equal, t, x, y, builder)
      return builder.select(bit, x, y)
    
    elif prim == prims.minimum:
      x,y = llvm_args
      bit = self.cmp(prims.less_equal, t, x, y, builder)
      return builder.select(bit, x, y)
    
    elif prim == prims.negative:
      if t == Bool: 
        bit = llvm_convert.to_bit(llvm_args[0], builder)
        negated = builder.not_(bit)
        return llvm_convert.to_bool(negated, builder)
      return self.neg(llvm_args[0], builder)
    
    # python's remainder is weird in that it preserve's the sign of 
    # the second argument, whereas LLVM's srem/frem operators preserve
    # the sign of the first 
    elif prim == prims.mod:
      x,y = llvm_args 
      if isinstance(t, (UnsignedT, BoolT)):
        return builder.urem(llvm_args[0], llvm_args[1], "modulo")
      elif isinstance(t, SignedT): 
        rem = builder.srem(x,y, "modulo")
      else:
        assert isinstance(t, FloatT)
        rem = builder.frem(llvm_args[0], llvm_args[1], "modulo")

      y_is_negative = self.lt(t, y, zero(y.type), builder, "second_arg_negative")
      rem_is_negative = self.lt(t, rem, zero(rem.type), builder, "rem_is_negative")
      y_nonzero = self.neq(t, y, zero(y.type), builder, "second_arg_nonzero")
      rem_nonzero = self.neq(t, rem, zero(x.type), builder, "rem_nonzero")
      neither_zero = builder.and_(y_nonzero, rem_nonzero, "neither_zero")
      diff_signs = builder.xor(y_is_negative, rem_is_negative, "different_signs")
      should_flip = builder.and_(neither_zero, diff_signs, "should_flip") 
      flipped_rem = self.add(t, y, rem, builder, "flipped_rem")
      return builder.select(should_flip, flipped_rem, rem)

    elif prim == prims.power:
      x,y = llvm_args
       
      if isinstance(t, FloatT):
        new_t = t 
      else:
        new_t = Float64
      x = llvm_convert.convert(x, t, new_t, builder)
      y = llvm_convert.convert(y, t, new_t, builder)
      llvm_op = llvm_prims.get_float_op(prim, new_t)
      result = builder.call(llvm_op, [x,y])
      return llvm_convert.convert(result, new_t, t, builder)
        
    elif isinstance(prim, prims.Arith) or isinstance(prim, prims.Bitwise):
      if isinstance(t, FloatT):
        instr = llvm_prims.float_binops[prim]
      elif isinstance(t, SignedT):
        instr = llvm_prims.signed_binops[prim]
      elif isinstance(t, UnsignedT):
        instr = llvm_prims.unsigned_binops[prim]
      else:
        assert isinstance(t, BoolT)
        instr = llvm_prims.bool_binops[prim]
      op = getattr(builder, instr)
      return op(name = result_name, *llvm_args)

    elif isinstance(prim, prims.Logical):
      if prim == prims.logical_and:
        result = builder.and_(name = result_name, 
                            lhs = to_bit(llvm_args[0], builder), 
                            rhs = to_bit(llvm_args[1], builder))
        return from_bit(result, t, builder)
      elif prim == prims.logical_not:
        result = builder.not_(to_bit(llvm_args[0], builder), name = result_name)
        return from_bit(result, t, builder)
      else:
        assert prim == prims.logical_or
        result = builder.or_(lhs = to_bit(llvm_args[0], builder), 
                           rhs = to_bit(llvm_args[1], builder), name = result_name)
        return from_bit(result, t, builder)
      
    elif prim == prims.abs:
      x = llvm_args[0]
      bit = self.cmp(prims.greater_equal, t,  x, zero(x.type), builder, "gt_zero")
      neg_value = self.neg(x, builder)
      return builder.select(bit, x, neg_value)
    elif isinstance(prim, prims.Float): 
      llvm_op = llvm_prims.get_float_op(prim, t)
      return builder.call(llvm_op, llvm_args)
    
    else:
      assert False, "UNSUPPORTED PRIMITIVE: %s" % prim

  def compile_PrimCall(self, expr, builder):
    args = expr.args
    # type specialization should have made types of arguments uniform,
    # so we only need to check the type of the first arg
    t = args[0].type
    llvm_args = [self.compile_expr(arg, builder) for arg in args]
    return self.prim(expr.prim, t, llvm_args, builder)

  def compile_expr(self, expr, builder):
    method_name = "compile_" + expr.node_type()
    return getattr(self, method_name)(expr, builder)

  def compile_Assign(self, stmt, builder):
    rhs_t = stmt.rhs.type
    # special case for locally allocated structs
    if self.may_escape is not None and \
       stmt.lhs.__class__ is Var and \
       stmt.rhs.__class__ is Struct and \
       stmt.lhs.name  not in self.may_escape:
      value = self.compile_Struct(stmt.rhs, builder, local = True)
    else:
      value = self.compile_expr(stmt.rhs, builder)
    if stmt.lhs.__class__ is Var:
      name = stmt.lhs.name
      lhs_t = stmt.lhs.type
      self.initialized.add(name)
      ref = self.vars[name]
    elif stmt.lhs.__class__ is Index:
      ptr_t = stmt.lhs.value.type
      assert isinstance(ptr_t, PtrT), \
          "Expected pointer, got %s" % ptr_t
      lhs_t = ptr_t.elt_type
      base_ptr = self.compile_expr(stmt.lhs.value, builder)
      index = self.compile_expr(stmt.lhs.index, builder)
      index = llvm_convert.from_signed(index, Int32, builder)
      ref = builder.gep(base_ptr, [index], "elt_ptr")
    else:
      assert stmt.lhs.__class__ is Attribute, \
          "Unexpected LHS: %s" % stmt.lhs
      struct = stmt.lhs.value
      ref, lhs_t = self.attribute_lookup(struct, stmt.lhs.name, builder)

    assert lhs_t == rhs_t, \
        "Type mismatch between LHS %s and RHS %s" % (lhs_t, rhs_t)

    builder.store(value, ref)
    return builder, False

  def compile_ExprStmt(self, stmt, builder):
    self.compile_expr(stmt.value, builder)
    return builder, False

  def compile_Return(self, stmt, builder):
    ret_val = self.compile_expr(stmt.value, builder)
    builder.ret(ret_val)
    return builder, True

  def compile_merge_left(self, phi_nodes, builder):
    for name, (left, _) in phi_nodes.iteritems():
      ref = self.vars[name]
      self.initialized.add(name)
      value = self.compile_expr(left, builder)
      builder.store(value, ref)

  def compile_merge_right(self, phi_nodes, builder):
    for name, (_, right) in phi_nodes.iteritems():
      ref = self.vars[name]
      self.initialized.add(name)
      value = self.compile_expr(right, builder)
      builder.store(value, ref)

  def compile_ForLoop(self, stmt, builder):
    # first compile the starting, boundary, and
    # increment values for the loop counter
    start = self.compile_expr(stmt.start, builder)
    stop = self.compile_expr(stmt.stop, builder)
    step = self.compile_expr(stmt.step, builder)

    # get the memory slot associated with the loop counter
    loop_var = self.vars[stmt.var.name]

    builder.store(start, loop_var)
    self.initialized.add(stmt.var.name)

    # ...and we'll need its Parakeet type later on
    # for calls to 'cmp' and 'prim'
    loop_var_t = stmt.var.type

    # any phi-bound variables should be initialized to their
    # starting values
    self.compile_merge_left(stmt.merge, builder)

    loop_bb, body_start_builder = self.new_block("loop_body")
    after_bb, after_builder = self.new_block("after_loop")

    # WARNING: Assuming loop is always increasing,
    # only enter the loop if we're less than the stopping value
    enter_cond = self.cmp(prims.less, loop_var_t,  start, stop,
                          builder, "enter_cond")
    builder.cbranch(enter_cond, loop_bb, after_bb)

    # TODO: what should we do if the body always ends in a return statement?
    body_end_builder, body_always_returns = \
      self.compile_block(stmt.body, body_start_builder)

    counter_at_end = body_end_builder.load(loop_var)
    # increment the loop counter
    incr = self.prim(prims.add, loop_var_t,
                     [counter_at_end, step],
                     body_end_builder,  "incr_loop_var")
    body_end_builder.store(incr, loop_var)
    self.compile_merge_right(stmt.merge, body_end_builder)

    exit_cond = self.cmp(prims.less, loop_var_t, incr, stop,
                         body_end_builder, "exit_cond")
    body_end_builder.cbranch(exit_cond, loop_bb, after_bb)
    # WARNING: what if the loop doesn't run? Should
    # we still be returning 'body_always_returns'?
    return after_builder, body_always_returns

  def compile_While(self, stmt, builder):
    # current flow ----> loop --------> exit--> after
    #    |                       skip------------|
    #    |----------------------/

    self.compile_merge_left(stmt.merge, builder)
    loop_bb, body_start_builder = self.new_block("loop_body")

    after_bb, after_builder = self.new_block("after_loop")
    enter_cond = self.compile_expr(stmt.cond, builder)
    enter_cond = llvm_convert.to_bit(enter_cond, builder)
    builder.cbranch(enter_cond, loop_bb, after_bb)

    body_end_builder, body_always_returns = \
        self.compile_block(stmt.body, body_start_builder)
    if not body_always_returns:
      exit_bb, exit_builder = self.new_block("loop_exit")
      self.compile_merge_right(stmt.merge, body_end_builder)
      repeat_cond = self.compile_expr(stmt.cond, body_end_builder)
      repeat_cond = llvm_convert.to_bit(repeat_cond, body_end_builder)
      body_end_builder.cbranch(repeat_cond, loop_bb, exit_bb)
      exit_builder.branch(after_bb)

    return after_builder, False

  def compile_If(self, stmt, builder):
    cond = self.compile_expr(stmt.cond, builder)
    cond = llvm_convert.to_bit(cond, builder)

    if len(stmt.true) == 0 and len(stmt.false) == 0:
      # if nothing happens in the loop bodies, just
      # emit select instructions
      for (name, (true_expr, false_expr)) in stmt.merge.iteritems():
        ref = self.vars[name]
        self.initialized.add(name)
        true_val = self.compile_expr(true_expr, builder)
        false_val = self.compile_expr(false_expr, builder)
        select_val = builder.select(cond, true_val, false_val)
        builder.store(select_val, ref)
      return builder, False
    else:
      # compile the two possible branches as distinct basic blocks
      # and then wire together the control flow with branches
      true_bb, true_builder = self.new_block("if_true")
      after_true, true_always_returns = \
        self.compile_block(stmt.true, true_builder)

      false_bb, false_builder = self.new_block("if_false")
      after_false, false_always_returns = \
          self.compile_block(stmt.false, false_builder)

      builder.cbranch(cond, true_bb, false_bb)

      # compile phi nodes as assignments and then branch
      # to the continuation block
      self.compile_merge_left(stmt.merge, after_true)
      self.compile_merge_right(stmt.merge, after_false)

      # if both branches return then there is no point
      # making a new block for more code
      # did both branches end in a return?
      both_always_return = true_always_returns and false_always_returns
      if both_always_return:
        return None, True
      after_bb, after_builder = self.new_block("if_after")
      if not true_always_returns:
        after_true.branch(after_bb)
      if not false_always_returns:
        after_false.branch(after_bb)
      return after_builder, False

  def compile_Comment(self, stmt, builder):
    return builder, False

  def compile_stmt(self, stmt, builder):
    """
    Translate an SSA statement into LLVM. Every translation function returns a
    builder pointing to the end of the current basic block and a boolean
    indicating whether every branch of control flow in that statement ends in a
    return. The latter is needed to avoid creating empty basic blocks, which
    were causing some mysterious crashes inside LLVM.
    """
    method_name = "compile_" + stmt.node_type()
    return getattr(self, method_name)(stmt, builder)

  def compile_block(self, stmts, builder):
    for stmt in stmts:
      builder, always_returns = self.compile_stmt(stmt, builder)
      if always_returns:
        return builder, always_returns
    return builder, False

  def compile_body(self, body):
    return self.compile_block(body, builder = self.entry_builder)
  
  
import os
from collections import namedtuple   
CompiledFn = namedtuple('CompiledFn', ('llvm_fn', 'llvm_exec_engine', 'parakeet_fn')) 

compiled_functions = {}
def compile_fn(fundef):
  key = fundef.cache_key 
  if key in compiled_functions:
    return compiled_functions[key]
  
  if config.print_lowered_function:
    print
    print "=== Lowered function ==="
    print
    print repr(fundef)
    print

  compiler = Compiler(fundef)
  compiler.compile_body(fundef.body)
  if llvm_config.print_unoptimized_llvm:
    print "=== LLVM before optimizations =="
    print
    print compiler.llvm_context.module
    print
  compiler.llvm_context.run_passes(compiler.llvm_fn)

  if config.print_generated_code:
    print "=== LLVM after optimizations =="
    print
    print compiler.llvm_context.module
    print

  if llvm_config.print_x86:
    print "=== Generated assembly =="
    print
    start_printing = False
    w,r = os.popen2("llc")
    w.write(str(compiler.llvm_context.module))
    w.close()
    assembly_str = r.read()
    r.close()
    for l in assembly_str.split('\n'):
      if l.strip().startswith('.globl'):
        if start_printing:
          break
        elif fundef.name in l:
          start_printing = True
      if start_printing:
        print l

  result = CompiledFn(compiler.llvm_fn,
                      compiler.llvm_context.exec_engine, 
                      fundef)
  compiled_functions[key] = result
  return result
