import adverbs
import array_type
import core_types
import llvm_backend
import lowering
import numpy as np
import prims
import run_function
import syntax
import syntax_helpers
import testing_helpers
import time

from closure_type import make_closure_type
from parakeet import allpairs

x = 1000
x2_array = np.arange(x*x, dtype = np.float).reshape(x,x)
y2_array = np.arange(x*x, 2*x*x, dtype = np.float).reshape(x,x)
x_array_t = array_type.make_array_type(core_types.Int64, 1)
x_2_array_t = array_type.make_array_type(core_types.Int64, 2)

mul_x_y = syntax.TypedFn(
  name = "mul_x_y",
  arg_names = ["mx", "my"],
  input_types = [core_types.Int64, core_types.Int64],
  body = [syntax.Return(syntax.PrimCall(prims.multiply,
                                        [syntax.Var("mx", type=core_types.Int64),
                                         syntax.Var("my", type=core_types.Int64)
                                        ], type=core_types.Int64))],
  return_type = core_types.Int64,
  type_env = {"mx":core_types.Int64, "my":core_types.Int64})

add_x_y = syntax.TypedFn(
  name = "add_x_y",
  arg_names = ["ax", "ay"],
  input_types = [core_types.Int64, core_types.Int64],
  body = [syntax.Return(syntax.PrimCall(prims.add,
                                        [syntax.Var("ax", type=core_types.Int64),
                                         syntax.Var("ay", type=core_types.Int64)
                                        ], type=core_types.Int64))],
  return_type = core_types.Int64,
  type_env = {"ax":core_types.Int64, "ay":core_types.Int64})

red_fn = syntax.TypedFn(
  name = "red_fn",
  arg_names = ["X_Red", "Y_Red"],
  input_types = [x_array_t, x_array_t],
  body = [syntax.Return(adverbs.Reduce(add_x_y, syntax_helpers.zero_i64,
                                       mul_x_y,
                                       [syntax.Var("X_Red", type=x_array_t),
                                        syntax.Var("Y_Red", type=x_array_t)],
                                       0, type=core_types.Int64))],
  return_type = core_types.Int64,
  type_env = {"X_Red":x_array_t, "Y_Red":x_array_t})

red_fixed_args = [syntax.Var("X_MM2", type=x_array_t)]
red_closure_t = make_closure_type(red_fn, [x_array_t])
red_fn_closure = syntax.Closure(red_fn, red_fixed_args, type=red_closure_t)

mm2_fn = syntax.TypedFn(
  name = "mm2_fn",
  arg_names = ["Y_MM2", "X_MM2"],
  input_types = [x_2_array_t, x_array_t],
  body = [syntax.Return(adverbs.Map(red_fn_closure,
                                    [syntax.Var("Y_MM2", type=x_2_array_t)],
                                    0, type=x_array_t))],
  return_type = x_array_t,
  type_env = {"X_MM2":x_array_t, "Y_MM2":x_2_array_t})

mm2_fixed_args = [syntax.Var("Y", type=x_2_array_t)]
mm2_closure_t = make_closure_type(mm2_fn, [x_2_array_t])
mm2_fn_closure = syntax.Closure(mm2_fn, mm2_fixed_args, type=mm2_closure_t)

mm_fn = syntax.TypedFn(
  name = "mm_fn",
  arg_names = ["X", "Y"],
  input_types = [x_2_array_t, x_2_array_t],
  body = [syntax.Return(adverbs.Map(mm2_fn_closure,
                                    [syntax.Var("X", type=x_2_array_t)],
                                    0, type=x_2_array_t))],
  return_type = x_2_array_t,
  type_env = {"X":x_2_array_t, "Y":x_2_array_t})

map_mul_fn = syntax.TypedFn(
  name = "map_mul_fn",
  arg_names = ["X", "Y"],
  input_types = [x_2_array_t, x_2_array_t],
  body = [syntax.Return(adverbs.Map(red_fn,
                                    [syntax.Var("X", type=x_2_array_t),
                                     syntax.Var("Y", type=x_2_array_t)],
                                    0, type=x_array_t))],
  return_type = x_array_t,
  type_env = {"X":x_2_array_t, "Y":x_2_array_t})

#def test_tiled_mm():
#  new_fn = lowering.lower(mm_fn, True)
#  assert isinstance(new_fn, syntax.TypedFn)
#  llvm_fn, parakeet_fn, exec_engine = llvm_backend.compile_fn(new_fn)
#  wrapper = run_function.CompiledFn(llvm_fn, parakeet_fn, exec_engine)
#  a2_array = np.arange(12).reshape(4,3)
#  b2_array = np.arange(9,21).reshape(4,3)
#  rslt = wrapper(a2_array, b2_array, (2,2,2))
#  nprslt = np.dot(a2_array, b2_array.T)
#  assert(testing_helpers.eq(rslt, nprslt)), \
#      "Expected %s but got %s" % (nprslt, rslt)

def dot(x, y):
  return sum(x*y)

def adverb_matmult(X, Y):
  return allpairs(dot, X, Y)

def test_par_mm():
  start = time.time()
  rslt = adverb_matmult(x2_array, y2_array)
  par_time = time.time() - start
  start = time.time()
  nprslt = np.dot(x2_array, y2_array.T)
  np_time = time.time() - start
  start = time.time()
  adverb_matmult(x2_array, y2_array)
  no_comp_time = time.time() - start
  assert(testing_helpers.eq(rslt, nprslt))
  print "Parakeet time with compilation:", par_time
  print "Parakeet time without compilation:", no_comp_time
  print "NumPy time:", np_time

if __name__ == '__main__':
  testing_helpers.run_local_tests()
