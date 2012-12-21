class NameNotFound(Exception):
  def __init__(self, name):
    self.name = name
    

  def __str__(self):
    return self.name 
  
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

lcase_chars = [chr(i + 97) for i in xrange(26)]
def fresh_list(count):
  prefixes = lcase_chars[:count]
  while len(prefixes) < count:
    count -= 26
    prefixes += lcase_chars[:count]
  return map(fresh, prefixes)


def original(unique_name):
  original_name = original_names.get(unique_name)
  if original_name is None:
    return unique_name 
  else:
    return original_name 
  
def refresh(unique_name):
  """Given an existing unique name, create another versioned name with the same root"""
  try:
    return fresh(original(unique_name))
  except NameNotFound:
    # it wasn't really an SSA name but keep going anyway 
    return fresh(unique_name)