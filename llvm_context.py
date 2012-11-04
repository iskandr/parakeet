import llvm.core as core 
import llvm.ee as ee
import llvm.passes as passes

class LLVM_Context:
  """Combine a module, exec engine, and pass manager into a single object"""
  _default_passes = [
    'mem2reg', 
    'simplifycfg', 'dce', 'sccp', 'gvn',  
    'memcpyopt', 
    'licm', 'loop-simplify', 'indvars',
  ]

  def __init__(self, module_name, optimize = True, verify = False):
    self.module = core.Module.new(module_name)
    self.engine_builder = ee.EngineBuilder.new(self.module)
    self.engine_builder.force_jit()
    if optimize:
      self.engine_builder.opt(3)
    else:
      self.engine_builder.opt(0)
    self.exec_engine = self.engine_builder.create()
    self.pass_manager = passes.FunctionPassManager.new(self.module)
    if optimize:
      for p in self._default_passes:
        self.pass_manager.add(p)
    if verify:
      self.pass_manager.add("verify")
      
  def run_passes(self, llvm_fn, n_iters = 3):
    for _ in xrange(n_iters):
      self.pass_manager.run(llvm_fn)
  

opt_context = LLVM_Context("opt_module", optimize = True, verify = False)
no_opt_context = LLVM_Context("no_opt_module", optimize = False, verify = False)
verify_context = LLVM_Context("verify_module", optimize = False, verify = True)
opt_and_verify_context = LLVM_Context("opt_and_verify_module", optimize = True, verify = True)