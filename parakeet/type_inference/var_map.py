
from .. import names 

class VarMap:
  def __init__(self):
    self._vars = {}

  def rename(self, old_name):
    new_name = names.refresh(old_name)
    self._vars[old_name] = new_name
    return new_name

  def lookup(self, old_name):
    if old_name in self._vars:
      return self._vars[old_name]
    else:
      return self.rename(old_name)

  def __str__(self):
    return "VarMap(%s)" % self._vars
