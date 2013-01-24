class ScopedDictionary(object):
  def __init__(self):
    self.scopes = []
    
  def push(self, d = None):
    if d is None:
      d = {}
    self.scopes.append(d)
  
  def pop(self):
    return self.scopes.pop()
  
  def top(self):
    return self.scopes[-1]
  
  def get(self, key):
    for scope in reversed(self.scopes):
      res = scope.get(key)
      if res:
        return res 
    return None
  
  def __getitem__(self, key):
    for scope in reversed(self.scopes):
      res = scope.get(key)
      if res:
        return res 
    assert False, "Key %s not found" % key
    
  def __setitem__(self, key, value):
    self.scopes[-1][key] = value 
    
  def setdefault(self, key, value):
    self.scopes[-1].setdefault(key, value)
  
  def __contains__(self, key):
    for scope in reversed(self.scopes):
      if key in scope:
        return True 
    return False 