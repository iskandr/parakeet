import numpy as np 
from llvm_runtime import scalar_to_generic_value  
from llvm_state import global_exec_engine


class CompiledFn:
  def __init__(self, llvm_fn, parakeet_fn):
    self.llvm_fn = llvm_fn
    self.parakeet_fn = parakeet_fn
  
  def __call__(self, *args):
    gv_inputs = []  
    for (v, t) in zip(args, self.parakeet_fn.input_types):
      if np.isscalar(v):
        gv_inputs.append(scalar_to_generic_value(v, t))
      else:
        assert False, (v,t)
    return global_exec_engine.run_function(self.llvm_fn, gv_inputs)