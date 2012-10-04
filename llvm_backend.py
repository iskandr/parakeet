import numpy as np


import llvm.core as llcore
from llvm.core import Type as lltype
from llvm.core import Builder 
import llvm.passes as passes

import ptype
import prims 
import syntax
from common import dispatch  

from llvm_types import to_lltype, convert 
from llvm_state import global_module
from llvm_compiled_fn import CompiledFn
import llvm_prims 


def init_llvm_vars(fundef, llvm_fn, builder):
  env = {}  
  for (name, t) in fundef.type_env.iteritems():
    llvm_t = to_lltype(t)
    stack_val = builder.alloca(llvm_t, name)
    env[name] = stack_val 

  for llvm_arg, parakeet_arg in zip(llvm_fn.args, fundef.args):
    if isinstance(parakeet_arg, str):
      name = parakeet_arg
    elif isinstance(parakeet_arg, syntax.Var):
      name = parakeet_arg.name
    else:
      assert False, "Tuple arg patterns not yet implemented"
    llvm_arg.name = name
    # store the value of the input in the stack value we've already allocated
    # for the input var 
    store = builder.store(llvm_arg, env[name])

  # tell the builder to start inserting 
  return env 



def compile_fn(fundef):
  llvm_input_types = map(to_lltype, fundef.input_types)
  llvm_output_type = to_lltype(fundef.return_type)
  llvm_fn_t = lltype.function(llvm_output_type, llvm_input_types)
  llvm_fn = global_module.add_function(llvm_fn_t, fundef.name)
    
  def new_block(name):
    bb = llvm_fn.append_basic_block(name)
    builder = Builder.new(bb)
    return bb, builder 
 
  _, start_builder = new_block("entry")
  env = init_llvm_vars(fundef, llvm_fn, start_builder)
  
 
  
  def compile_expr(expr, builder):
    print "compile_expr", expr 
    def compile_Var():
      ref =  env[expr.name]
      val = builder.load(ref, expr.name + "_val")
      return val 
    def compile_Const():
      assert isinstance(expr.type, ptype.ScalarT)
      llvm_type = to_lltype(expr.type)
      if isinstance(expr.type, ptype.IntT):
        return llcore.Constant.int(llvm_type, expr.value)
      elif isinstance(expr.type, ptype.FloatT):
        return llcore.Constant.real(llvm_type, expr.value)
      else:
        assert False, "Unsupported constant %s" % expr
    def compile_Cast():
      llvm_value = compile_expr(expr.value, builder)
      return convert(llvm_value, expr.value.type, expr.type, builder)
    def compile_PrimCall():
      prim = expr.prim
      args = expr.args 
      
      # type specialization should have made types of arguments uniform, 
      # so we only need to check the type of the first arg 
      t = args[0].type
      
      llvm_args = [compile_expr(arg, builder) for arg in args]
      
      result_name = prim.name + "_result"
      
      if isinstance(prim, prims.Cmp):
        x, y = llvm_args 
        if isinstance(t, ptype.FloatT):
          cmp_op = llvm_prims.float_comparisons[prim]
          return builder.fcmp(cmp_op, x, y, result_name)
        elif isinstance(t, ptype.SignedT):
          cmp_op = llvm_prims.signed_int_comparisons[prim]
          return builder.icmp(cmp_op, x, y, result_name)
        else:
          assert isinstance(t, ptype.UnsignedT), "Unexpected type: %s" % t
          cmp_op = llvm_prims.unsigned_int_comparisons[prim]
          return builder.icmp(cmp_op, x, y, result_name)
      elif isinstance(prim, prims.Arith) or isinstance(prim, prims.Bitwise):
        if isinstance(t, ptype.FloatT):
          instr = llvm_prims.float_binops[prim]
        elif isinstance(t, ptype.SignedT):
          instr = llvm_prims.signed_binops[prim]
        elif isinstance(t, ptype.UnsignedT):
          instr = llvm_prims.unsigned_binops[prim]  
        return getattr(builder, instr)(*llvm_args)
      else:
        assert False, "UNSUPPORTED PRIMITIVE: %s" % expr 
     
    return dispatch(expr, "compile")
    
  def get_ref(expr):
    # for now only get references to variables
    assert isinstance(expr, syntax.Var)
    return env[expr.name]
      
  def compile_merge_left(phi_nodes, builder):
    for name, (left, _) in phi_nodes.iteritems():
      ref = env[name]
      value = compile_expr(left, builder)
      builder.store(value, ref)
      
  def compile_merge_right(phi_nodes, builder):
    for name, (_, right) in phi_nodes.iteritems():
      ref = env[name]
      value = compile_expr(right, builder)
      builder.store(value, ref)
    
  def compile_stmt(stmt, builder):
    """Translate an SSA statement into llvm. Every translation
    function returns a builder pointing to the end of the current 
    basic block and a boolean indicating whether every branch of 
    control flow in that statement ends in a return. 
    The latter is needed to avoid creating empty basic blocks, 
    which were causing some mysterious crashes inside LLVM"""
    if isinstance(stmt, syntax.Assign):
      ref = get_ref(stmt.lhs)
      value = compile_expr(stmt.rhs, builder)
      builder.store(value, ref)
      return builder, False 
    elif isinstance(stmt, syntax.While):
      # current flow ----> loop --------> exit--> after  
      #    |                       skip------------|
      #    |----------------------/
      compile_merge_left(stmt.merge_before, builder)
      loop_bb, loop_builder = new_block("loop_body")
      
      skip_bb, skip_builder = new_block("skip_loop")
      after_bb, after_builder = new_block("after_loop")
      enter_cond = compile_expr(stmt.cond, builder)
      builder.cbranch(enter_cond, loop_bb, skip_bb)
      _, body_always_returns = compile_block(stmt.body, loop_builder)
      if not body_always_returns:
        exit_bb, exit_builder = new_block("loop_exit")
        compile_merge_right(stmt.merge_before, loop_builder)
        repeat_cond = compile_expr(stmt.cond, loop_builder)
        loop_builder.cbranch(repeat_cond, loop_bb, exit_bb)
        compile_merge_right(stmt.merge_after, exit_builder)
        exit_builder.branch(after_bb)
      compile_merge_left(stmt.merge_after, skip_builder)
      skip_builder.branch(after_bb)
      return after_builder, False 
 
    elif isinstance(stmt, syntax.Return):
      builder.ret(compile_expr(stmt.value, builder))
      return builder, True 
    elif isinstance(stmt, syntax.If):
      cond = compile_expr(stmt.cond, builder)
      
      # compile the two possible branches as distinct basic blocks
      # and then wire together the control flow with branches
      true_bb, true_builder = new_block("if_true")
      _, true_always_returns = compile_block(stmt.true, true_builder)
      
      false_bb, false_builder = new_block("if_false")
      _, false_always_returns = compile_block(stmt.false, false_builder)
      
      # did both branches end in a return? 
      both_always_return = true_always_returns and false_always_returns 

      builder.cbranch(cond, true_bb, false_bb)
      # compile phi nodes as assignments and then branch
      # to the continuation block 
      compile_merge_left(stmt.merge, true_builder)
      compile_merge_right(stmt.merge, false_builder)
      
      # if both branches return then there is no point
      # making a new block for more code 
      if both_always_return:

        return None, True 
      else:
        after_bb, after_builder = new_block("if_after")
        if not true_always_returns:
          true_builder.branch(after_bb)
        if not false_always_returns:
          false_builder.branch(after_bb)      
        return after_builder, False  
  
  def compile_block(stmts, builder = None):
    print "compile_block", stmts
    for stmt in stmts:
      builder, always_returns = compile_stmt(stmt, builder)
      if always_returns:
        return builder, True 
    return builder, False

  compile_block(fundef.body, start_builder)
  print llvm_fn
  
  return CompiledFn(llvm_fn, fundef)
