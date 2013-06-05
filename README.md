Parakeet: Runtime accelerator for numerical Python

=========
Do you have intolerably slow numerical algorithms written in Python? 
Parakeet may be able to significantly speed up your bottleneck through 
*type specialization* and *native code generation*. 

To accelerate wrap it with Parakeet's *@jit* decorator:

```python 

   import numpy as np 
   
   alpha = 1.0
   beta = 0.3
   def slow(x):
     y = np.empty_like(x)
     for i in xrange(len(x)):
       y[i] = np.tanh(x[i] * alpha + beta)
     return y
     
  from parakeet import jit 
  @jit
  def fast(x,y):
    y = np.empty_like(x)
    for i in xrange(len(x)):
      y[i] = np.tanh(x[i] * alpha + beta)
    return x 
    
  @jit
  def fast_comprehension(x):
    return [np.tanh(xi*alpha + beta) for xi in x] 
```

Parakeet cannot accelerate arbitrary Python code, it only supports a limited subset of the language:

  * Scalar operations (i.e. addition, multiplication, etc...)
  * Control flow (if-statements, loops, etc...)
  * Tuples
  * Slices
  * NumPy arrays (and some NumPy library functions) 
  * List literals (interpreted as array construction)
  * List comprehensions (interpreted as array comprehensions)
  * Parakeet's "adverbs" (higher order array operations like parakeet.map, parakeet.reduce)
  

Parakeet is written for Python 2.7 (sorry internet) and depends on:

* [llvmpy](https://github.com/llvmpy/llvmpy)
* [NumPy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/) 
* [nose](https://nose.readthedocs.org/en/latest/) for unit tests


