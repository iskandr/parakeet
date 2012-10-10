

"""
Map each (untyped fn id, fixed arg) types to a distinct integer
so that the runtime representation of closures just need to 
carry this ID
"""
closure_sig_to_id = {}
id_to_closure_sig = {}
max_id = 0  

def get_id(closure_sig):
  global max_id
  if closure_sig in closure_sig_to_id:
    return closure_sig_to_id[closure_sig]
  else:
    num = max_id
    max_id += 1
    closure_sig_to_id[closure_sig] = num
    return num 
  
def get_closure_signature(num):
  return id_to_closure_sig[num]

