__author__  = 'Alex Rubinsteyn'
__email__   = 'alex -dot- rubinsteyn -at- gmail -dot- com'
__desc__    = 'Runtime compiler for numerical Python'
__license__     = 'BSD3'
__version__     = '0.14.1'
__website__     = 'https://github.com/iskandr/parakeet'

import config
from ndtypes import * 
from analysis import SyntaxVisitor, verify
from builder import Builder, build_fn 
from transforms import clone_function, Transform
from prims import *   
from lib import * 
from frontend import jit, macro, run_python_fn, run_typed_fn, typed_repr, specialize   

