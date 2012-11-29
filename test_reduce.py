from parakeet import reduce, add
import testing_helpers 
import numpy as np 


short_int_vec = np.arange(5, dtype=int)
long_int_vec = np.arange(100, dtype=int)
short_float_vec = short_int_vec.astype(float)
long_float_vec = long_int_vec.astype(float)
short_bool_vec = short_float_vec < np.mean(short_float_vec)
long_bool_vec = long_float_vec < np.mean(long_float_vec)


vecs = [short_int_vec, long_int_vec, 
        short_float_vec, long_float_vec, 
        short_bool_vec, long_bool_vec]

def sum(xs):
    return reduce(add, xs)

def test_sum():
    testing_helpers.expect_each(sum, np.sum, vecs)

if __name__ == '__main__':
    testing_helpers.run_local_tests()
