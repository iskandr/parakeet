import llvm.core 
import llvm.ee 


global_module = llvm.core.Module.new("global_module")
global_exec_engine = llvm.ee.ExecutionEngine.new(global_module)