import ctypes
import numpy as np

import testing_helpers

from parakeet import adverbs, array_type, core_types
from parakeet import prims, llvm_backend, run_function 
from parakeet import syntax, syntax_helpers, type_conv
from parakeet.pipeline import lower_tiled 

x_array = np.arange(10, dtype = np.int64)
x2_array = np.arange(100, dtype = np.int64).reshape(10,10)
x_array_t = array_type.make_array_type(core_types.Int64, 1)
x_2_array_t = array_type.make_array_type(core_types.Int64, 2)

id_fn = syntax.TypedFn(
  name = "id_fn",
  arg_names = ["x"],
  input_types = [core_types.Int64],
  body = [syntax.Return(syntax.Var("x", type=core_types.Int64))],
  return_type = core_types.Int64,
  type_env = {"x":core_types.Int64})

id_fn_2 = syntax.TypedFn(
  name = "id_fn_2",
  arg_names = ["x"],
  input_types = [core_types.Int64],
  body = [syntax.Return(syntax.Var("x", type=core_types.Int64))],
  return_type = core_types.Int64,
  type_env = {"x":core_types.Int64})

map_fn = syntax.TypedFn(
  name = "map_fn",
  arg_names = ["X"],
  input_types = [x_array_t],
  body = [syntax.Return(adverbs.Map(id_fn, [syntax.Var("X", type=x_array_t)],
                                    0, type=x_array_t))],
  return_type = x_array_t,
  type_env = {"X":x_array_t})

nested_id_fn = syntax.TypedFn(
  name = "nested_id_fn",
  arg_names = ["x"],
  input_types = [core_types.Int64],
  body = [syntax.Return(syntax.Var("x", type=core_types.Int64))],
  return_type = core_types.Int64,
  type_env = {"x":core_types.Int64})

nested_map_fn = syntax.TypedFn(
  name = "nested_map_fn",
  arg_names = ["X"],
  input_types = [x_array_t],
  body = [syntax.Return(adverbs.Map(nested_id_fn,
                                    [syntax.Var("X", type=x_array_t)],
                                    0, type=x_array_t))],
  return_type = x_array_t,
  type_env = {"X":x_array_t})

map2_fn = syntax.TypedFn(
  name = "map2_fn",
  arg_names = ["X"],
  input_types = [x_2_array_t],
  body = [syntax.Return(adverbs.Map(nested_map_fn,
                                    [syntax.Var("X", type=x_2_array_t)],
                                    0, type=x_2_array_t))],
  return_type = x_2_array_t,
  type_env = {"X":x_2_array_t})

id_fn_2d = syntax.TypedFn(
  name = "id_fn_2d",
  arg_names = ["x"],
  input_types = [x_array_t],
  body = [syntax.Return(syntax.Var("x", type=x_array_t))],
  return_type = x_array_t,
  type_env = {"x":x_array_t})

map2d_1map_fn = syntax.TypedFn(
  name = "map2d_1map_fn",
  arg_names = ["X"],
  input_types = [x_2_array_t],
  body = [syntax.Return(adverbs.Map(id_fn_2d,
                                    [syntax.Var("X", type=x_2_array_t)],
                                    0, type=x_2_array_t))],
  return_type = x_2_array_t,
  type_env = {"X":x_2_array_t})

add_x_y2 = syntax.TypedFn(
  name = "add_x_y2",
  arg_names = ["x", "y"],
  input_types = [core_types.Int64, core_types.Int64],
  body = [syntax.Return(syntax.PrimCall(prims.add,
                                        [syntax.Var("x", type=core_types.Int64),
                                         syntax.Var("y", type=core_types.Int64)
                                        ], type=core_types.Int64))],
  return_type = core_types.Int64,
  type_env = {"x":core_types.Int64, "y":core_types.Int64})

red_fn2 = syntax.TypedFn(
  name = "red_fn2",
  arg_names = ["X"],
  input_types = [x_array_t],
  body = [syntax.Return(adverbs.Reduce(add_x_y2, syntax_helpers.zero_i64,
                                       id_fn_2,
                                       [syntax.Var("X", type=x_array_t)],
                                       0, type=core_types.Int64))],
  return_type = core_types.Int64,
  type_env = {"X":x_array_t})

def get_np_ptr(arr):
  obj = type_conv.from_python(arr)
  return ctypes.pointer(obj)

def test_1d_map():
  new_fn = lower_tiled(map_fn)
  assert isinstance(new_fn, syntax.TypedFn)
  llvm_fn, parakeet_fn, exec_engine = llvm_backend.compile_fn(new_fn)
  wrapper = run_function.CompiledFn(llvm_fn, parakeet_fn, exec_engine)
  rslt = wrapper(x_array, (5,))
  assert testing_helpers.eq(rslt, x_array)

def test_2d_map():
  new_fn = lower_tiled(map2d_1map_fn)
  assert isinstance(new_fn, syntax.TypedFn)
  llvm_fn, parakeet_fn, exec_engine = llvm_backend.compile_fn(new_fn)
  wrapper = run_function.CompiledFn(llvm_fn, parakeet_fn, exec_engine)
  tile_array = (3,) * new_fn.num_tiles
  rslt = wrapper(x2_array, tile_array)
  assert testing_helpers.eq(rslt, x2_array)

def test_2_maps():
  new_fn = lower_tiled(map2_fn)
  assert isinstance(new_fn, syntax.TypedFn)
  llvm_fn, parakeet_fn, exec_engine = llvm_backend.compile_fn(new_fn)
  wrapper = run_function.CompiledFn(llvm_fn, parakeet_fn, exec_engine)
  tile_array = (3,) * new_fn.num_tiles
  rslt = wrapper(x2_array, tile_array)
  assert testing_helpers.eq(rslt, x2_array), \
      "Expected %s but got %s" % (x2_array, rslt)

def test_1d_reduce():
  new_fn = lower_tiled(red_fn2)
  assert isinstance(new_fn, syntax.TypedFn)
  llvm_fn, parakeet_fn, exec_engine = llvm_backend.compile_fn(new_fn)
  wrapper = run_function.CompiledFn(llvm_fn, parakeet_fn, exec_engine)
  rslt = wrapper(x_array, (3,) * new_fn.num_tiles)
  assert testing_helpers.eq(rslt, sum(x_array))

if __name__ == '__main__':
  testing_helpers.run_local_tests()
