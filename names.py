class NameNotFound(Exception):
  def __init__(self, name):
    self.name = name
     
versions = {}
original_names = {}
  
def get(name):
  version = versions.get(name)
  if version is None:
    raise NameNotFound(name)
  else:
    return "%s.%d" % (name, version)
    
def fresh(name):
  version = versions.get(name, 0) + 1 
  versions[name] = version
  ssa_name = "%s.%d" % (name, version)
  original_names[ssa_name] = name
  return ssa_name 

def original(unique_name):
  original_name = original_names.get(unique_name)
  if original_name is None:
    raise NameNotFound(unique_name)
  else:
    return original_name 
  
def refresh(unique_name):
  """Given an existing unique name, create another versioned name with the same root"""
  return fresh(original(unique_name))