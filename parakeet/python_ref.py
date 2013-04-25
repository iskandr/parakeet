import abc

class Ref(object):
  __meta__ = abc.ABCMeta

  @abc.abstractmethod
  def deref(self):
    pass

class GlobalValueRef(Ref):
  def __init__(self, value):
    self.value = value 
    
  def deref(self):
    return self.value 
  
  def __str__(self):
    return "GlobalValueRef(%s)" % (self.value,)
  
  def __repr__(self):
    return str(self)
  
    def __eq__(self, other):
      return isinstance(other, GlobalValueRef) and self.value is other.value 

class GlobalNameRef(Ref):
  def __init__(self, globals_dict, name):
    self.globals_dict = globals_dict
    self.name = name

  def __str__(self):
    return "GlobalNameRef(%s)" % self.name

  def __repr__(self):
    return str(self)

  def deref(self):
    return self.globals_dict[self.name]

  def __eq__(self, other):
    return isinstance(other, GlobalNameRef) and \
           self.globals_dict is other.globals_dict and \
           self.name == other.name

class ClosureCellRef(Ref):
  def __init__(self, cell, name):
    self.cell = cell
    self.name = name

  def deref(self):
    return self.cell.cell_contents

  def __str__(self):
    return "ClosureCellRef(%s)" % self.name

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    if isinstance(other, ClosureCellRef):
      try:
        return self.cell == other.cell
      except:
        return self.cell is other.cell
    return False
