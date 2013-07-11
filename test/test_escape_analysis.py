import parakeet 

from parakeet.core_types import Int64 
from parakeet.escape_analysis import EscapeAnalysis
from parakeet.syntax import Tuple, Var, TypedFn, Assign, Return, TupleProj 
from parakeet.syntax_helpers import zero_i64, one_i64   
from testing_helpers import run_local_tests
from parakeet.tuple_type import make_tuple_type
 

tuple_t = make_tuple_type((Int64, Int64))
nested_tuple_t = make_tuple_type((tuple_t, tuple_t))

a_var = Var("a", type = Int64)
b_var = Var("b", type = tuple_t)
c_var = Var("c", type = tuple_t)
d_var = Var("d", type= nested_tuple_t)
e_var = Var("e", type = nested_tuple_t)
body = [
  Assign(a_var, one_i64),  
  Assign(b_var, Tuple(elts = (zero_i64, a_var), type = tuple_t)), 
  Assign(c_var, Tuple(elts = (TupleProj(b_var, 0, type = Int64), a_var), 
                      type = tuple_t)), 
  Assign(d_var, Tuple(elts = (b_var, b_var), type = nested_tuple_t)), 
  Assign(e_var, Tuple(elts = (c_var, c_var), type = nested_tuple_t)), 
  Return(d_var) 
]
tenv = { "a": Int64, 
         "b": tuple_t, 
         "c": tuple_t, 
         "d": nested_tuple_t, 
         "e": nested_tuple_t
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
  assert "a" not in escape_analysis.may_escape, "Scalars can't escape function"
  assert "b" in escape_analysis.may_escape, "Nested tuples also escape!"
  assert "e" not in escape_analysis.may_escape
  
if __name__ == '__main__':
  run_local_tests()
