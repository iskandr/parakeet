import syntax
import names 
from names import NameNotFound

class ScopedEnv:  
  def __init__(self, outer_env = None):
    self.scopes = [{}]
    self.blocks = [[]]
    # link together environments of nested functions
    self.outer_env = outer_env
    
  def fresh(self, name):
    fresh_name = names.fresh(name)
    self.scopes[-1][name] = fresh_name 
    return fresh_name
  
  def fresh_var(self, name):
    return syntax.Var(self.fresh(name))
  
  def push(self, scope = None, block = None):
    if scope is None:
      scope = {}
    if block is None:
      block = []
    self.scopes.append(scope)
    self.blocks.append(block)
  
  def pop(self):
    scope = self.scopes.pop()
    block = self.blocks.pop()
    return scope, block 
  
  def current_scope(self):
    return self.scopes[-1]
  
  def current_block(self):
    return self.blocks[-1]
  
  def __getitem__(self, key):
    for scope in reversed(self.scopes):
      if key in scope:
        return scope[key]
    raise NameNotFound(key)

  def __contains__(self, key):
    for scope in reversed(self.scopes):
      if key in scope: 
        return True
    return False 
  
  def recursive_lookup(self, key, skip_current = False):
    if not skip_current and key in self:
      return self[key]
    else:
      if self.outer_env:
        self.outer_env.recursive_lookup(key)
      else:
        return None
