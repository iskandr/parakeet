import numpy as np

from llvm import *
from llvm.core import *
import llvm.core as llcore
from llvm.core import Type as lltype
import llvm.passes as passes

from llvm_types import to_lltype
from llvm_state import global_module
from llvm_compiled_fn import CompiledFn

import ptype
import syntax
from common import dispatch  

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
    builder.store(llvm_arg, env[name])
    
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
  
  def compile_expr(expr):
    def compile_Var():
      return env[expr.name]
    def compile_Const():
      assert isinstance(expr.type, ptype.Scalar)
      llvm_type = to_lltype(expr.type)
      if expr.type.is_int():
        return llcore.Constant.int(llvm_type, expr.value)
      elif expr.type.is_float():
        return llcore.Constant.real(llvm_type, expr.value)
      else:
        assert False, "Unsupported constant %s" % expr
    return dispatch(expr, "compile")
  def compile_merge_left(phi_nodes, builder):
    for name, (left, _) in phi_nodes.iteritems():
      builder.store(env[name], compile_expr(left))
      
  def compile_merge_right(phi_nodes, builder):
    for name, (_, right) in phi_nodes.iteritems():
      builder.store(env[name], compile_expr(right))
    
  def compile_stmt(stmt, builder):
    if isinstance(stmt, syntax.Assign):
      return builder 
    elif isinstance(stmt, syntax.While):
      return builder
    elif isinstance(stmt, syntax.If):
      cond = compile_expr(stmt.cond)
      
      # compile the two possible branches as distinct basic blocks
      # and then wire together the control flow with branches
      true_bb, true_builder = new_block("if_true")
      compile_block(stmt.true, true_builder)
      
      false_bb, false_builder = new_block("if_false")
      compile_block(stmt.false, false_builder)
      
      builder.cbranch(cond, true_bb, false_bb)

      # compile phi nodes as assignments and then branch
      # to the continuation block 
      compile_merge_left(stmt.merge, true_builder)
      compile_merge_right(stmt.merge, false_builder)
      
      after_bb, after_builder = new_block("if_merge")
      true_builder.branch(after_bb)
      false_builder.branch(after_bb)      
      
      return after_builder 
  
  def compile_block(stmts, builder = None):
    
    for stmt in stmts:
      builder = compile_stmt(stmt, builder)
    return builder

  compile_block(fundef.body, start_builder)
  print llvm_fn
  
  return CompiledFn(llvm_fn, fundef)

"""
  let init_local_var (fnInfo:fn_info) (id:ID.t) =
    let impT = ID.Map.find id fnInfo.imp_types in
    let shape = ID.Map.find id fnInfo.imp_shapes in
    let llvmT = LlvmType.of_imp_type impT in
    IFDEF DEBUG THEN
      let initStr =
        Printf.sprintf
          "Initializing local %s : %s%s to have lltype %s\n%!"
          (ID.to_str id)
          (ImpType.to_str impT)
          (if SymbolicShape.is_scalar shape then ""
           else "(shape=" ^ SymbolicShape.to_str shape ^ ")")
          (Llvm.string_of_lltype llvmT)
      in
      (*debug  initStr fnInfo.builder*)
      ()
    ENDIF;
    let varName = ID.to_str id in
    let stackVal : llvalue =
      if ImpType.is_scalar impT || ImpType.is_vector impT then
      Llvm.build_alloca llvmT varName fnInfo.builder
      (* local array *)
      else (
        let localArray : llvalue =
          allocate_local_array_struct fnInfo varName impT
        in
        if ID.Map.find id fnInfo.imp_storage = Imp.Local then
          allocate_local_array fnInfo localArray impT shape
        ;
        localArray
      )
    in
    Hashtbl.add fnInfo.named_values varName stackVal

  let preload_array_metadata fnInfo (input:llvalue) (impT:ImpType.t) =
    match impT with
      | ImpType.ArrayT (eltT, rank) ->
        ignore $ get_array_field ~add_to_cache:true fnInfo input Imp.ArrayData;
        for i = 0 to rank - 1 do
          ignore $
            get_array_field_elt ~add_to_cache:true fnInfo input Imp.ArrayShape i
          ;
          ignore $
            get_array_field_elt
              ~add_to_cache:true fnInfo input Imp.ArrayStrides i
        done
      | ImpType.PtrT(eltT, Some len) ->
          Hashtbl.add array_field_cache (input, Imp.PtrData) input;
          Hashtbl.add array_field_cache (input, Imp.PtrLen) (mk_int32 len);
      | ImpType.ScalarT _ ->
        (* do nothing for scalars *)
        ()
      | _ -> failwith "ImpType not supported"

  let init_nonlocal_var (fnInfo:fn_info) (id:ID.t) (param:Llvm.llvalue) =
    let impT = ID.Map.find id fnInfo.imp_types in
    let llvmT = LlvmType.of_imp_type impT in
    IFDEF DEBUG THEN
      let initStr =
        Printf.sprintf "Initializing nonlocal %s : %s to have lltype %s\n%!"
          (ID.to_str id)
          (ImpType.to_str impT)
          (Llvm.string_of_lltype llvmT)
      in
      (*debug initStr fnInfo.builder*)
      ()
    ENDIF;
    let varName = ID.to_str id in
    Llvm.set_value_name varName param;
    let stackVal =
      match impT with
      | ImpType.VectorT _
      | ImpType.ScalarT _ when List.mem id fnInfo.input_ids ->
        (* scalar input *)
        let stackVal = Llvm.build_alloca llvmT varName fnInfo.builder in
        ignore $ Llvm.build_store param stackVal fnInfo.builder;
        stackVal
      | _ ->
        (* output or array input *)
        let ptrT = Llvm.pointer_type llvmT in
        let ptrName = varName^"_ptr" in
        let ptr = Llvm.build_inttoptr param ptrT ptrName fnInfo.builder in
        preload_array_metadata fnInfo ptr impT;
        ptr
    in
    Hashtbl.add fnInfo.named_values varName stackVal

  let init_compiled_fn (fnInfo:fn_info) =
    let get_imp_type id =  ID.Map.find id fnInfo.imp_types in
    let get_imp_types ids = List.map get_imp_type ids in

    let impInputTypes = get_imp_types fnInfo.input_ids in
    let impOutputTypes = get_imp_types fnInfo.output_ids in

    let replace_array_with_int64 t =
      if ImpType.is_scalar t then LlvmType.of_imp_type t
      else LlvmType.int64_t
    in
    let llvmInputTypes = List.map replace_array_with_int64 impInputTypes in
    (* IMPORTANT: outputs are allocated outside the function and the *)
    (* addresses of their locations are passed in *)
    let llvmOutputTypes =
      List.map (fun _ -> LlvmType.int64_t) impOutputTypes
    in
    let paramTypes = llvmInputTypes @ llvmOutputTypes in
    (* since we have to pass output address as int64s, convert them all *)
    (* in the signature *)
    let fnT = Llvm.function_type void_t (Array.of_list paramTypes) in
    let llvmFn = Llvm.declare_function fnInfo.name fnT llvm_module in
    let bb = Llvm.append_block context "entry" llvmFn in
    Llvm.position_at_end bb fnInfo.builder;
    (* To avoid having to manually encode phi-nodes around the *)
    (* use of mutable variables, we instead allocate stack space *)
    (* for every input and local variable at the beginning of the *)
    (* function. We don't need to allocate space for inputs since *)
    (* they are already given to us as pointers. *)
    List.iter2
      (init_nonlocal_var fnInfo)
      (fnInfo.input_ids @ fnInfo.output_ids)
      (Array.to_list (Llvm.params llvmFn))
    ;
    List.iter (init_local_var fnInfo) fnInfo.local_ids;
    llvmFn
end

let compile_fn (fn : Imp.fn) : Llvm.llvalue =
  let fnInfo = create_fn_info fn in
  let llvmFn : Llvm.llvalue = Init.init_compiled_fn fnInfo in
  let initBasicBlock : Llvm.llbasicblock = Llvm.entry_block llvmFn in
  let _ = compile_stmt_seq fnInfo initBasicBlock fn.body in
  ignore $ Llvm.build_ret_void fnInfo.builder;
  llvmFn
"""

