import llvm.core 
import llvm.ee 


global_module = llvm.core.Module.new("global_module")
global_exec_engine = llvm.ee.ExecutionEngine.new(global_module)


class ClosureSignatures:
  """
  Map each (untyped fn id, fixed arg) types to a distinct integer
  so that the runtime representation of closures just need to 
  carry this ID
  """
  closure_sig_to_id = {}
  id_to_closure_sig = {}
  max_id = 0  
  
  
  @classmethod
  def get_id(cls, closure_sig):
    if closure_sig in cls.closure_sig_to_id:
      return cls.closure_sig_to_id[closure_sig]
    else:
      num = cls.max_id
      cls.max_id += 1
      cls.closure_sig_to_id[closure_sig] = num
      return num 
  
  @classmethod     
  def get_closure_signature(cls, num):
    return cls.id_to_closure_sig[num]
