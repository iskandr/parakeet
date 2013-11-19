

from parakeet import Int64, make_tuple_type, make_array_type
from parakeet.analysis.escape_analysis import EscapeAnalysis
from parakeet.syntax import Tuple, Var, TypedFn, Assign, Return, Array, TupleProj, zero_i64, one_i64
from parakeet.testing_helpers import run_local_tests   

 

array_t = make_array_type(Int64, 1)
tuple_t = make_tuple_type((Int64, array_t))
nested_tuple_t = make_tuple_type((tuple_t, tuple_t))
array_const = Array([one_i64], type = array_t)

a_int = Var("a_int", type = Int64)
b_array = Var("b_array", type = array_t)
c_tuple = Var("c_tuple", type = tuple_t)
d_tuple = Var("d_tuple", type = tuple_t)
e_nested_tuple = Var("e_nested_tuple", type= nested_tuple_t)
f_nested_tuple = Var("f_nested_tuple", type = nested_tuple_t)
body = [
  Assign(a_int, one_i64),  
  Assign(b_array, array_const), 
  Assign(c_tuple, Tuple(elts = (a_int, b_array), type = tuple_t)),
  Assign(d_tuple, Tuple(elts = (a_int, array_const), type = tuple_t)),
  Assign(e_nested_tuple, Tuple(elts = (c_tuple, c_tuple), type = nested_tuple_t)),
  Assign(f_nested_tuple, Tuple(elts = (d_tuple, d_tuple), type = nested_tuple_t)),  
  Return(b_array) 
]
tenv = { "a_int": Int64, 
         "b_array": array_t, 
         "c_tuple": tuple_t, 
         "d_tuple": nested_tuple_t, 
         "e_nested_tuple": nested_tuple_t,
         "f_nested_tuple" : nested_tuple_t, 
        }

fn = TypedFn(name = "test_escape_analysis", 
             type_env = tenv, 
             input_types = (), 
             arg_names = (), 
             body = body, 
             return_type = nested_tuple_t
            )

def test_escape_analysis():
  escape_analysis = EscapeAnalysis()
  escape_analysis.visit_fn(fn)
  may_escape =  escape_analysis.may_escape
 
  assert "a_int" not in may_escape, "Scalars don't count as escaping"
  assert "b_array" in may_escape
  assert "c_tuple" in may_escape
  assert "d_tuple" not in may_escape
  assert "e_nested_tuple" in may_escape 
  assert "f_nested_tuple" not in may_escape
  
if __name__ == '__main__':
  run_local_tests()
