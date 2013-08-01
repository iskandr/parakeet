Parakeet: Runtime accelerator for numerical Python
====

If you have intolerably slow numerical algorithms written in Python, 
Parakeet may be able to significantly speed up your bottleneck through 
*type specialization* and *native code generation*. 

To accelerate a function, wrap it with Parakeet's **@jit** decorator:

```python 

   import numpy as np 
   
   def slow(x, alpha = 0.5, beta = 0.3):
     y = np.empty_like(x)
     for i in xrange(len(x)):
       y[i] = np.tanh(x[i] * alpha + beta)
     return y
     
  from parakeet import jit 
  @jit
  def fast(x, alpha = 0.5, beta = 0.3):
    y = np.empty_like(x)
    for i in xrange(len(x)):
      y[i] = np.tanh(x[i] * alpha + beta)
    return x 
    
  @jit
  def fast_comprehension(x, alpha = 0.5, beta = 0.3):
    return [np.tanh(xi*alpha + beta) for xi in x] 
```

Supported Subset of Python
====

Parakeet cannot accelerate arbitrary Python code, it only supports a limited subset of the language:

  * Scalar operations (i.e. addition, multiplication, etc...)
  * Control flow (if-statements, loops, etc...)
  * Tuples
  * Slices
  * NumPy arrays (and some NumPy library functions) 
  * List literals (interpreted as array construction)
  * List comprehensions (interpreted as array comprehensions)
  * Parakeet's "adverbs" (higher order array operations like parakeet.map, parakeet.reduce)

Dependencies
====

Parakeet is written for Python 2.7 (sorry internet) and depends on:

* [treelike](https://github.com/iskandr/treelike)
* [llvmpy](https://github.com/llvmpy/llvmpy)
* [NumPy and SciPy](http://www.scipy.org/install.html)
* [nose](https://nose.readthedocs.org/en/latest/) for unit tests

Install
====
You should be able to install Parakeet from its [PyPI package](https://pypi.python.org/pypi/parakeet/) by running "pip install parakeet". 

More Information
====
If you have any questions about the project either check out our [HotPar slides](https://www.usenix.org/conference/hotpar12/parakeet-just-time-parallel-accelerator-python) 
from last year or contact the [main developer](http://www.rubinsteyn.com).
