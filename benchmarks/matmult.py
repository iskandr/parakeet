import parakeet
import numpy as np 
 
def matmult_high_level(X,Y):
  return np.array([[np.dot(x,y) for y in Y.T] for x in X])


n, d = 1200, 1200
m = 1200
dtype = 'float32'
X = np.random.randn(m,d).astype(dtype)
Y = np.random.randn(d,n).astype(dtype)

from compare_perf import compare_perf

extra = {'BLAS (np.dot)' : np.dot}
try:
  import pycuda
  import pycuda.gpuarray 
  import pycuda.autoinit

  import scikits.cuda.linalg
  import scikits.cuda.cublas 
  cublas_context = scikits.cuda.cublas.cublasCreate()
  def gpu_dot(X,Y):
    X_gpu = pycuda.gpuarray.to_gpu(X)
    Y_gpu = pycuda.gpuarray.to_gpu(Y)
    Z_gpu = scikits.cuda.linalg.dot(X_gpu, Y_gpu, 'N', 'N', handle = cublas_context)
    return Z_gpu.get()
  extra['cuBLAS'] = gpu_dot 
except:
  print "Failed to import PyCUDA + scikits.cuda" 

try:
  import numba 

  @numba.autojit 
  def matmult_loops(X,Y,Z):
    m, d = X.shape
    n = Y.shape[1]
    for i in xrange(m):
      for j in xrange(n):
        total = X[i,0] * Y[0,j] 
        for k in xrange(1,d):
          total += X[i,k] * Y[k,j]
        Z[i,j] = total 
  
  def call_numba(X,Y):
    Z = np.zeros((X.shape[0],Y.shape[1])).astype(dtype)
    matmult_loops(X,Y,Z)
    return Z 

  extra['numba'] = call_numba 

except:
  print "Failed to import Numba" 
  pass 

compare_perf(matmult_high_level, [X,Y],
             cpython=True, 
             # numba can't run the nested comprehensions so we use
             # a special loopy version instead 
             numba=False,
             extra = extra, 
             suppress_output = False,
             propagate_exceptions = False)

