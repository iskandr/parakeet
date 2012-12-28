import names
import syntax

from collections import OrderedDict
from names import NameNotFound
from python_ref import GlobalRef, ClosureCellRef

class ScopedEnv:
  def __init__(self, outer_env=None, closure_cell_dict=None, globals_dict=None):
    self.scopes = [{}]
    self.blocks = [[]]
    # link together environments of nested functions
    self.outer_env = outer_env
    # names we had to look up in the outer_env
    self.original_outer_names = []
    # the local SSA names we assigned to some
    # outer name
    self.localized_outer_names = []

    self.closure_cell_dict = closure_cell_dict
    self.globals_dict = globals_dict
    self.python_refs = OrderedDict()

  def __str__(self):
    return "ScopedEnv(blocks = %s, scopes = %s)" % (self.blocks, self.scopes)

  def __repr__(self):
    return str(self)

  def items(self):
    all_items = []
    for s in self.scopes:
      all_items.extend(s.items())
    return all_items

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

  def top_scope(self):
    return self.scopes[0]

  def current_scope(self):
    return self.scopes[-1]

  def top_block(self):
    return self.blocks[0]

  def current_block(self):
    return self.blocks[-1]

  def get(self, key):
    for scope in reversed(self.scopes):
      if key in scope:
        return scope[key]
    return None

  def __getitem__(self, key):
    for scope in reversed(self.scopes):
      res = scope.get(key)
      if res:
        return res

    if self.outer_env:
      # don't actually keep the outer binding name, we just
      # need to check that it's possible and tell the outer scope
      # to register any necessary python refs
      self.outer_env[key]
      local_name = names.fresh(key)
      self.top_scope()[key] = local_name
      self.original_outer_names.append(key)
      self.localized_outer_names.append(local_name)
      return local_name

    if self.closure_cell_dict and key in self.closure_cell_dict:
      ref = ClosureCellRef(self.closure_cell_dict[key], key)
    elif self.globals_dict and key in self.globals_dict:
      ref = GlobalRef(self.globals_dict, key)
    else:
      raise NameNotFound(key)
    for (local_name, other_ref) in self.python_refs.iteritems():
      if ref == other_ref:
        return local_name
    local_name = names.fresh(key)
    self.python_refs[local_name] = ref
    return local_name

  def __setitem__(self, k, v):
    scope = self.top_scope()
    scope[k] = v

  def __contains__(self, key):
    for scope in reversed(self.scopes):
      if key in scope:
        return True
    return False
