import llvm.core as core
import llvm.ee as ee
import llvm.passes as passes

class LLVM_Context:
  """Combine a module, exec engine, and pass manager into a single object"""
  
  _verify_passes = [
    'preverify', 
    'domtree', 
    'verify'
  ]
  _opt_passes = [
    'mem2reg', 
    'targetlibinfo', 
    'tbaa', 
    'basicaa',
    'instcombine', 
    'simplifycfg', 
    'basiccg',
    'memdep', 
    'scalarrepl-ssa',
    'domtree',
    'early-cse',
    'simplify-libcalls',
    'lazy-value-info',
    'correlated-propagation', 
    'simplifycfg', 
    'instcombine', 
    'reassociate', 
    'domtree',
    'loops', 
    'loop-simplify', 
    'lcssa', 
    'loop-rotate', 
    'licm', 
    'lcssa', 
    'loop-unswitch', 
    'instcombine', 
    'scalar-evolution',
    'loop-simplify',
    'lcssa', 'indvars',
    'loop-idiom', 
    'loop-deletion', 
    'loop-unroll',
    'bb-vectorize',
    'gvn',  
    'sccp',
    'correlated-propagation', 
    'jump-threading',
    'dse', 
    'adce',
    'simplifycfg', 
    'instcombine', 
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
    
    self.pass_manager.add(self.exec_engine.target_data)
    for p in self._verify_passes: 
      self.pass_manager.add(p)
    if optimize:
      for p in (self._opt_passes + self._verify_passes):
        self.pass_manager.add(p)

  def run_passes(self, llvm_fn, n_iters = 2):
    for _ in xrange(n_iters):
      self.pass_manager.run(llvm_fn)

opt = LLVM_Context("opt_module", optimize = True)
no_opt = LLVM_Context("no_opt_module", optimize = False)
