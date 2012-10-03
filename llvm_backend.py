import numpy as np

from llvm import *
from llvm.core import *
import llvm.core as llcore
from llvm.core import Type as lltype
import llvm.passes as passes

import ptype
import syntax
from common import dispatch  

void_t = lltype.void()
int1_t = lltype.int(1)
int8_t = lltype.int(8)
int16_t = lltype.int(16)
int32_t = lltype.int(32)
int64_t = lltype.int(64)
float32_t = lltype.float()
float64_t = lltype.double()
float128_t = lltype.fp128()

ptr_int32_t = lltype.pointer(int32_t)
ptr_int64_t = lltype.pointer(int64_t)

dtype_to_llvm_types = {
  
  np.dtype('int8') : int8_t,
  np.dtype('uint8') : int8_t,
  np.dtype('uint16') : int16_t, 
  np.dtype('int16') : int16_t,
  np.dtype('uint32') : int32_t, 
  np.dtype('int32') : int32_t,
  np.dtype('uint64') : int64_t, 
  np.dtype('int64') : int64_t,
  np.dtype('float16') : float32_t, 
  np.dtype('float32') : float32_t,
  np.dtype('float64') : float64_t,
}

def dtype_to_lltype(dt):
  return dtype_to_llvm_types[dt]

def to_lltype(t):
  if isinstance(t, ptype.Scalar):
    return dtype_to_lltype(t.dtype)
  elif isinstance(t, ptype.Tuple):
    llvm_elt_types = map(to_lltype, t.elt_types)
    return lltype.struct(llvm_elt_types)
  else:
    elt_t = dtype_to_lltype(t.dtype)
    arr_t = lltype.pointer(elt_t)
    # arrays are a pointer to their data and
    # pointers to shape and strides arrays
    return lltype.struct([arr_t, ptr_int64_t, ptr_int64_t])

# we allocate heap slots for output scalars before entering the
# function
def to_llvm_output_type(t):
  llvm_type = to_lltype(t)
  if isinstance(t, ptype.Scalar):
    return lltype.pointer(llvm_type)
  else:
    return llvm_type

global_module = llcore.Module.new("global_module")


def init_llvm_fn(fundef):
  llvm_input_types = map(to_lltype, fundef.input_types)
  llvm_output_type = to_llvm_output_type(fundef.result_type)
  llvm_fn_t = lltype.function(llvm_output_type, llvm_input_types)
  llvm_fn = global_module.add_function(llvm_fn_t, fundef.name)
  
  bb = llvm_fn.append_basic_block("entry")
  builder = Builder.new(bb)

  env = {}
  
  for (name, t) in fundef.type_env.iteritems():
    llvm_t = to_lltype(t)
    stack_val = builder.alloca(llvm_t, name)
    env[name] = stack_val 

  n_inputs = len(llvm_input_types)  
  for i, llvm_arg in enumerate(llvm_fn.args):
    if i < n_inputs:
      parakeet_arg = fundef.args[i]
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
    else:
      assert False, "Output args not yet implemented"
  
  
  # tell the builder to start inserting 
  return llvm_fn, builder, env  
 
def compile_fn(fundef):
  fn, init_builder, env  = init_llvm_fn(fundef)
  
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
  def compile_stmt(stmt, builder):
    if isinstance(stmt, syntax.Assign):
      assert False
    elif isinstance(stmt, syntax.While):
      assert False
    elif isinstance(stmt, syntax.If): 
      assert False
  
  def compile_block(stmts, builder):
    for stmt in stmts:
      compile_stmt(stmt, builder)
  
  compile_block(fundef.body, init_builder)
  return fn

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

