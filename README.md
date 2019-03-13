Parakeet 
====

**This project is no longer being maintained**. 

Parakeet was a runtime accelerator for an array-oriented subset of Python. In retrospect, I don't think that whole-function type specialization at the AST level is a scalable approach to speeding up a sufficiently large subset of Python. General-purpose Python code should probably be accelerated using a bytecode JIT, whereas high-performance numerical code should use a DSL with explicit parallel operators. 




Example
=======


To accelerate a function, wrap it with Parakeet's **@jit** decorator:

```python 
import numpy as np 
from parakeet import jit 

alpha = 0.5
beta = 0.3
x = np.array([1,2,3])
y = np.tanh(x * alpha) + beta
   
@jit
def fast(x, alpha = 0.5, beta = 0.3):
  return np.tanh(x * alpha) + beta 
   
@jit 
def loopy(x, alpha = 0.5, beta = 0.3):
  y = np.empty_like(x, dtype = float)
  for i in xrange(len(x)):
    y[i] = np.tanh(x[i] * alpha) + beta
  return y
     
@jit
def comprehension(x, alpha = 0.5, beta = 0.3):
  return np.array([np.tanh(xi*alpha) + beta for xi in x])
  
assert np.allclose(fast(x), y)
assert np.allclose(loopy(x), y)
assert np.allclose(comprehension(x), y)

```



Install
====
You should be able to install Parakeet from its [PyPI package](https://pypi.python.org/pypi/parakeet/) by running:

    pip install parakeet


Dependencies
====

Parakeet is written for Python 2.7 (sorry internet) and depends on:

* [dsltools](https://github.com/iskandr/dsltools)
* [nose](https://nose.readthedocs.org/en/latest/) for unit tests
* [NumPy](http://www.scipy.org/install.html)
* [appdirs](https://pypi.python.org/pypi/appdirs/)

The default backend (which uses OpenMP) requires `gcc` 4.4+. 

*Windows*: If you have a 32-bit Windows install, your compiler should come from [Cygwin](http://cygwin.com/install.html) or [MinGW](http://www.mingw.org/). Getting Parakeet working on 64-bit Windows is non-trivial and seems to require [colossal hacks](http://eli.thegreenplace.net/2008/06/28/compiling-python-extensions-with-distutils-and-mingw/).

*Mac OS X*: By default, your machine probably either has only [clang](http://clang.llvm.org/) or an outdated version of `gcc`. You can get a more recent version using [HomeBrew](http://apple.stackexchange.com/questions/38222/how-do-i-install-gcc-via-homebrew)

If you want to use the CUDA backend, you need to have an NVIDIA graphics card and install both the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [PyCUDA](http://mathema.tician.de/software/pycuda/). 


How does it work? 
====
Your untyped function gets used as a template from which multiple *type specializations* are generated 
(for each distinct set of input types). 
These typed functions are then churned through many optimizations before finally getting translated into native code. 

More information
===

  * Ask questions on the [discussion group](http://groups.google.com/forum/#!forum/parakeet-python)
  * Watch the [Parakeet presentation](https://vimeo.com/73895275) from this year's [PyData Boston](http://pydata.org/bos2013), look at the [HotPar slides](https://www.usenix.org/conference/hotpar12/parakeet-just-time-parallel-accelerator-python) from last year 
  * Contact the [main developer](http://www.rubinsteyn.com) directly



Supported language features
====

Parakeet cannot accelerate arbitrary Python code, it only supports a limited subset of the language:

  * Scalar operations (i.e. `x + 3 * y`)
  * Control flow (if-statements, loops, etc...)
  * Nested functions and lambdas
  * Tuples
  * Slices
  * NumPy array expressions (i.e. `x[1:, :] + 2 * y[:-1, ::2]`)
  * Some NumPy library functions like `np.ones` and `np.sin` (look at the [mappings](https://github.com/iskandr/parakeet/blob/master/parakeet/mappings.py) module for a full list)
  * List literals (interpreted as array construction)
  * List comprehensions (interpreted as array comprehensions)
  * Parakeet's higher order array operations like `parakeet.imap`, `parakeet.scan`, and `parakeet.allpairs`

Backends
===
Parakeet currently supports compilation to sequential C, multi-core C with OpenMP (default), or LLVM (deprecated). To switch between these options change `parakeet.config.backend` to one of:

  * *"openmp"*: compiles with gcc, parallel operators run across multiple cores (default)
  * *"c"*: lowers all parallel operators to loops, compile sequential code with gcc
  * *"cuda"*: launch parallel operations on the GPU (experimental)
  * *"llvm"*: older backend, has fallen behind and some programs may not work
  * *"interp"* : pure Python intepreter used for debugging optimizations, only try this if you think CPython is about 10,000x too fast for your taste 


