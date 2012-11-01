
from llvm.core import Type as lltype
from llvm.core import Builder 

from core_types import BoolT, FloatT, SignedT, UnsignedT, ScalarT, Bool, Int32, Int64, PtrT
from closure_type import ClosureT
import prims 
import syntax
from common import dispatch  
from function_registry import find_specialization

import llvm_types
from llvm_types import llvm_value_type, llvm_ref_type
from llvm_helpers import const, int32 #, zero, one 
import llvm_convert 
import llvm_prims 
import llvm_context 
from compiled_fn import CompiledFn


 

class CompilationEnv:
  def __init__(self, llvm_cxt = llvm_context.verify_context):
    self.parakeet_fundef = None
    self.llvm_fn = None
    self.llvm_context = llvm_cxt
    self.vars = {}
  
  def new_block(self, name):
    bb = self.llvm_fn.append_basic_block(name)
    builder = Builder.new(bb)
    return bb, builder 

  
  def init_fn(self, fundef):
    """
    Initializes the variables dictionary and returns a builder object
    """
    llvm_input_types = map(llvm_ref_type, fundef.input_types)
    llvm_output_type = llvm_ref_type(fundef.return_type)   
    llvm_fn_t = lltype.function(llvm_output_type, llvm_input_types)
    self.llvm_fn = self.llvm_context.module.add_function(llvm_fn_t, fundef.name)
    _, builder = self.new_block("entry")
    self._init_vars(fundef, builder)
    return builder 
  
  
  
  def _init_vars(self, fundef, builder):  

    """
    Create a mapping from variable names to stack locations,  
    these will later be converted to SSA variables by the mem2reg pass.
   """
  
   
    n_expected = len(fundef.args.arg_slots)
    assert len(self.llvm_fn.args) == n_expected
  
    for (name, t) in fundef.type_env.iteritems():
      if not name.startswith("$"):
        llvm_t = llvm_ref_type(t)
        stack_val = builder.alloca(llvm_t, name)
        self.vars[name] = stack_val 
  
    for llvm_arg, parakeet_arg in zip(self.llvm_fn.args, fundef.args.arg_slots):
      if isinstance(parakeet_arg, str):
        name = parakeet_arg
      elif isinstance(parakeet_arg, syntax.Var):
        name = parakeet_arg.name
      else:
        assert False, "Tuple arg patterns not yet implemented"
    
      llvm_arg.name = name
      builder.store(llvm_arg, self.vars[name])
      
  
  def __getitem__(self, name):
    if isinstance(name, syntax.Var):
      name = name.name
    assert isinstance(name, str)
    return self.vars[name]
  
  def __setitem__(self, name, val):
    if isinstance(name, syntax.Var):
      name = name.name
    assert isinstance(name, str)
    self.vars[name] = val
  




def compile_expr(expr, env, builder):
  # print "  EXPR: ", expr 
  def compile_Var():
    ref =  env[expr.name]
    val = builder.load(ref, expr.name + "_val")
    return val 
  def compile_Const():
    assert isinstance(expr.type, ScalarT)
    return const(expr.value, expr.type)
  
  def compile_Cast():
    llvm_value = compile_expr(expr.value, env, builder)
    return llvm_convert.convert(llvm_value, expr.value.type, expr.type, builder)
  

  def compile_Struct():
    llvm_struct_t = llvm_value_type(expr.type)
    name = expr.type.node_type() 
    struct_ptr = builder.malloc(llvm_struct_t, name + "_ptr")
    
      
    for (i, elt)  in enumerate(expr.args):
      elt_ptr = builder.gep(struct_ptr, [int32(0), int32(i)], "field%d_ptr" % i)
      llvm_elt = compile_expr(elt, env, builder)
      builder.store(llvm_elt, elt_ptr)

    return struct_ptr
  
  def compile_Alloc():
    elt_t = expr.elt_type
    llvm_elt_t = llvm_types.llvm_value_type(elt_t)
    n_elts = compile_expr(expr.count, env, builder)
    return builder.malloc_array(llvm_elt_t, n_elts, "data_ptr")
    
  
  def compile_Index():
    llvm_arr = compile_expr(expr.value, env, builder)
    llvm_index = compile_expr(expr.index, env, builder)
     
    index_t = expr.index.type 
    llvm_idx = llvm_convert.convert(llvm_index, index_t, Int32, builder)
    
    pointer = builder.gep(llvm_arr, [ llvm_idx], "elt_pointer")
    elt = builder.load(pointer, "elt")
    return elt 
    
  def compile_Attribute():
    llvm_value = compile_expr(expr.value, env, builder)
    struct_type = expr.value.type 
    field_pos = struct_type.field_pos(expr.name)
    field_type = struct_type.field_type(expr.name)
    field_ptr =  builder.gep(llvm_value, [int32(0), int32(field_pos)], "%s_ptr" % expr.name)
    field_value = builder.load(field_ptr, "%s_value" % expr.name)
    if isinstance(field_type, BoolT):
      return llvm_convert.to_bit(field_value)
    else:
      return field_value  
  
  
  def compile_Invoke():

    closure_t = expr.closure.type
    assert isinstance(closure_t, ClosureT)
    arg_types = [arg.type for arg in expr.args] 
    
    closure_object = compile_expr(expr.closure, env, builder)

    # get the int64 identifier which maps to an untyped_fn/arg_types pairs
    
    untyped_fn_id = closure_t.fn
    assert isinstance(untyped_fn_id, str), "Expected %s to be string identifier" % (untyped_fn_id)
    full_arg_types = closure_t.args + tuple(arg_types)
    # either compile the function we're about to invoke or get its compiled form from a cache
    typed_fundef = find_specialization(untyped_fn_id, full_arg_types)
    target_fn_info = compile_fn(typed_fundef)
    target_fn = target_fn_info.llvm_fn 
      
    llvm_closure_args = []
    
    for (closure_arg_idx, _) in enumerate(closure_t.args):
      arg_ptr = builder.gep(closure_object, [int32(0), int32(closure_arg_idx)], "closure_arg%d_ptr" % closure_arg_idx)
      arg = builder.load(arg_ptr, "closure_arg%d" % closure_arg_idx)
      llvm_closure_args.append(arg)
    
    
    llvm_direct_args = [compile_expr(arg, env, builder) for arg in expr.args]
    full_args_list = llvm_closure_args + llvm_direct_args 
    assert len(full_args_list) == len(full_arg_types)  
    return builder.call(target_fn, full_args_list, 'invoke_result')
 
    
  def compile_PrimCall():
    prim = expr.prim
    args = expr.args 
    
    # type specialization should have made types of arguments uniform, 
    # so we only need to check the type of the first arg 
    t = args[0].type
    
    llvm_args = [compile_expr(arg, env, builder) for arg in args]
    
    result_name = prim.name + "_result"
    
    if isinstance(prim, prims.Cmp):
      x, y = llvm_args 
      if isinstance(t, FloatT):
        cmp_op = llvm_prims.float_comparisons[prim]
        bit = builder.fcmp(cmp_op, x, y, result_name)
      elif isinstance(t, SignedT):
        cmp_op = llvm_prims.signed_int_comparisons[prim]
        bit = builder.icmp(cmp_op, x, y, result_name)
      else:
        assert isinstance(t, UnsignedT), "Unexpected type: %s" % t
        cmp_op = llvm_prims.unsigned_int_comparisons[prim]
        bit = builder.icmp(cmp_op, x, y, result_name)
      return llvm_convert.to_bool(bit,builder)
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
      return op(name = "%s_result" % prim.name, *llvm_args)
    else:
      assert False, "UNSUPPORTED PRIMITIVE: %s" % expr 
   
  return dispatch(expr, "compile")

def compile_merge_left(phi_nodes, env, builder):
  for name, (left, _) in phi_nodes.iteritems():
    ref = env[name]
    value = compile_expr(left, env, builder)
    builder.store(value, ref)
     
    
def compile_merge_right(phi_nodes, env, builder):
  for name, (_, right) in phi_nodes.iteritems():
    ref = env[name]
    value = compile_expr(right, env, builder)
    builder.store(value, ref)


def compile_stmt(stmt, env, builder):
  """Translate an SSA statement into llvm. Every translation
  function returns a builder pointing to the end of the current 
  basic block and a boolean indicating whether every branch of 
  control flow in that statement ends in a return. 
  The latter is needed to avoid creating empty basic blocks, 
  which were causing some mysterious crashes inside LLVM"""
  
  # print "STMT ", stmt 
  
  def compile_Assign():
    
    
    value = compile_expr(stmt.rhs, env, builder)

    if isinstance(stmt.lhs, syntax.Var):
      ref = env[stmt.lhs.name]
      builder.store(value, ref)
    elif isinstance(stmt.lhs, syntax.Index):
      assert isinstance(stmt.lhs.value.type, PtrT), \
        "Expected pointer, got %s" % stmt.lhs.value.type
      base_ptr = compile_expr(stmt.lhs.value, env, builder)
      index = compile_expr(stmt.lhs.index, env, builder)
      index = llvm_convert.from_signed(index, Int32, builder)  
      elt_ptr = builder.gep(base_ptr, [index], "elt_ptr")
      builder.store(value, elt_ptr)
      
      
    return builder, False 
  
  def compile_While():
    
    # current flow ----> loop --------> exit--> after  
    #    |                       skip------------|
    #    |----------------------/
    
    compile_merge_left(stmt.merge_before, env, builder)
    loop_bb, body_start_builder = env.new_block("loop_body")
    
    skip_bb, skip_builder = env.new_block("skip_loop")
    after_bb, after_builder = env.new_block("after_loop")
    enter_cond = compile_expr(stmt.cond, env, builder)
    enter_cond = llvm_convert.to_bit(enter_cond, builder)
    builder.cbranch(enter_cond, loop_bb, skip_bb)
    body_end_builder, body_always_returns = \
      compile_block(stmt.body, env, body_start_builder)
    if not body_always_returns:
      exit_bb, exit_builder = env.new_block("loop_exit")
      compile_merge_right(stmt.merge_before, env, body_end_builder)
      repeat_cond = compile_expr(stmt.cond, env, body_end_builder)
      repeat_cond = llvm_convert.to_bit(repeat_cond, body_end_builder)
      body_end_builder.cbranch(repeat_cond, loop_bb, exit_bb)
      compile_merge_right(stmt.merge_after, env, exit_builder)
      exit_builder.branch(after_bb)
    compile_merge_left(stmt.merge_after, env, skip_builder)
    skip_builder.branch(after_bb)
    return after_builder, False 
  
  def compile_Return():
    
    ret_val = compile_expr(stmt.value, env, builder)
    builder.ret(ret_val)
    return builder, True 
  
  def compile_If():
    cond = compile_expr(stmt.cond, env, builder)
    cond = llvm_convert.to_bit(cond, builder)
    
    
    # compile the two possible branches as distinct basic blocks
    # and then wire together the control flow with branches
    true_bb, true_builder = env.new_block("if_true")
    _, true_always_returns = compile_block(stmt.true, env, true_builder)
    
    false_bb, false_builder = env.new_block("if_false")
    _, false_always_returns = compile_block(stmt.false, env, false_builder)
    
    # did both branches end in a return? 
    both_always_return = true_always_returns and false_always_returns 
    
    builder.cbranch(cond, true_bb, false_bb)
    # compile phi nodes as assignments and then branch
    # to the continuation block 
    compile_merge_left(stmt.merge, env, true_builder)
    compile_merge_right(stmt.merge, env, false_builder)
    
    # if both branches return then there is no point
    # making a new block for more code 
    if both_always_return:
      return None, True 
    else:
      after_bb, after_builder = env.new_block("if_after")
      if not true_always_returns:
        true_builder.branch(after_bb)
      if not false_always_returns:
        false_builder.branch(after_bb)      
    return after_builder, False  
  
  return dispatch(stmt, 'compile')

def compile_block(stmts, env, builder):
  for stmt in stmts:
    builder, always_returns = compile_stmt(stmt, env, builder)
    if always_returns:
      return builder, always_returns 
  return builder, False

compiled_functions = {}

from transform import apply_pipeline
from lower_adverbs import LowerAdverbs
from lower_structs import LowerStructs
from lower_indexing import LowerIndexing
from simplify import Simplify


def prepare_fn(fundef):
  return apply_pipeline(fundef, [LowerAdverbs, LowerIndexing, LowerStructs, Simplify])

def compile_fn(fundef):
  
  if fundef.name in compiled_functions:
    return compiled_functions[fundef.name]
  
  print "FUNDEF"
  print fundef 
  fundef = prepare_fn(fundef)
  print
  print "LOWERED"
  print fundef 
  env = CompilationEnv()
  start_builder = env.init_fn(fundef)   
  compile_block(fundef.body, env, start_builder)
  print env.llvm_fn 
  env.llvm_context.run_passes(env.llvm_fn)

  # print env.llvm_fn 
  result = CompiledFn(env.llvm_fn, fundef) 
  compiled_functions[fundef.name] = result 
  
  return result 
