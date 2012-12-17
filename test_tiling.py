import adverbs
import array_type
import core_types
import interp
import llvm_backend
import lowering
import numpy as np
import parakeet
import prims
import run_function
import syntax
import syntax_helpers
import testing_helpers
import tile_adverbs
import transform

from parakeet import each
from run_function import specialize_and_compile

id_fn = syntax.TypedFn(
  name = "id_fn",
  arg_names = ["x"],
  input_types = [core_types.Int64],
  body = [syntax.Return(syntax_helpers.const(1))],
  return_type = core_types.Int64,
  type_env = {})

x_array = np.arange(10, dtype = np.int64)
x2_array = np.arange(100, dtype = np.int64).reshape(10,10)
x_array_t = array_type.make_array_type(core_types.Int64, 1)
x_2_array_t = array_type.make_array_type(core_types.Int64, 2)

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
  body = [syntax.Return(adverbs.Map(id_fn_2, [syntax.Var("X", type=x_array_t)],
                                    0, type=x_array_t))],
  return_type = x_array_t,
  type_env = {"X":x_array_t})

map2_fn = syntax.TypedFn(
  name = "map2_fn",
  arg_names = ["X"],
  input_types = [x_2_array_t],
  body = [syntax.Return(adverbs.Map(map_fn, [syntax.Var("X", type=x_2_array_t)],
                                    0, type=x_2_array_t))],
  return_type = x_2_array_t,
  type_env = {"X":x_2_array_t})

id_fn_3 = syntax.TypedFn(
  name = "id_fn_3",
  arg_names = ["x"],
  input_types = [x_array_t],
  body = [syntax.Return(syntax.Var("x", type=x_array_t))],
  return_type = x_array_t,
  type_env = {"x":x_array_t})

axis_fn = syntax.TypedFn(
  name = "axis_fn",
  arg_names = ["X"],
  input_types = [x_2_array_t],
  body = [syntax.Return(adverbs.Map(id_fn_3,
                                    [syntax.Var("X", type=x_2_array_t)],
                                    1, type=x_2_array_t))],
  return_type = x_2_array_t,
  type_env = {"X":x_2_array_t})

add_x_y = syntax.TypedFn(
  name = "add_x_y",
  arg_names = ["x", "y"],
  input_types = [core_types.Int64, core_types.Int64],
  body = [syntax.Return(syntax.PrimCall(prims.add,
                                        [syntax.Var("x", type=core_types.Int64),
                                         syntax.Var("y", type=core_types.Int64)
                                        ], type=core_types.Int64))],
  return_type = core_types.Int64,
  type_env = {"x":core_types.Int64, "y":core_types.Int64})

red_fn = syntax.TypedFn(
  name = "red_fn",
  arg_names = ["X"],
  input_types = [x_array_t],
  body = [syntax.Return(adverbs.Reduce(add_x_y, 0, add_x_y,
                                       [syntax.Var("X", type=x_array_t)],
                                       0, type=core_types.Int64))],
  return_type = core_types.Int64,
  type_env = {"X":x_array_t})

def identity(x):
  return x

def map_id(X):
  return each(identity, X)

def map2_id(X):
  return each(map_id, X)

#def test_axes():
#  new_fn = lowering.lower(axis_fn, False)
#  assert isinstance(new_fn, syntax.TypedFn)
#  llvm_fn, parakeet_fn, exec_engine = llvm_backend.compile_fn(new_fn)
#  wrapper = run_function.CompiledFn(llvm_fn, parakeet_fn, exec_engine)
#  rslt = wrapper(x2_array)
#  print rslt
#  assert testing_helpers.eq(rslt, x2_array)
#
#def test_nested_tiling():
#  new_fn = lowering.lower(axis_fn, True)
#  assert isinstance(new_fn, syntax.TypedFn)
#  llvm_fn, parakeet_fn, exec_engine = llvm_backend.compile_fn(new_fn)
#  wrapper = run_function.CompiledFn(llvm_fn, parakeet_fn, exec_engine)
#  rslt = wrapper(x2_array, np.array([5,4], dtype=np.int64))
#  print rslt
#  assert testing_helpers.eq(rslt, x2_array)

#def test_axes():
#  new_fn = lowering.lower(axis_fn, False)
#  assert isinstance(new_fn, syntax.TypedFn)
#  llvm_fn, parakeet_fn, exec_engine = llvm_backend.compile_fn(new_fn)
#  print parakeet_fn
#  wrapper = run_function.CompiledFn(llvm_fn, parakeet_fn, exec_engine)
#  rslt = wrapper(x2_array)
#  assert testing_helpers.eq(rslt, x2_array.T), \
#      "Expected %s but got %s" % (x2_array.T, rslt)
#  print rslt

def test_1d_map():
  new_fn = lowering.lower(map2_fn, True)
  assert isinstance(new_fn, syntax.TypedFn)
  llvm_fn, parakeet_fn, exec_engine = llvm_backend.compile_fn(new_fn)
  wrapper = run_function.CompiledFn(llvm_fn, parakeet_fn, exec_engine)
  rslt = wrapper(x2_array, np.array([10,5], dtype=np.int64))
  print rslt
  #assert testing_helpers.eq(rslt, x2_array)

#def test_1d_reduce():
#  print red_fn
#  new_fn = lowering.lower(red_fn, True)
#  print new_fn
#  assert isinstance(new_fn, syntax.TypedFn)
#  llvm_fn, parakeet_fn, exec_engine = llvm_backend.compile_fn(new_fn)
#  wrapper = run_function.CompiledFn(llvm_fn, parakeet_fn, exec_engine)
#  rslt = wrapper(x_array, np.array([6], dtype=np.int64))
#  print rslt

if __name__ == '__main__':
  testing_helpers.run_local_tests()
