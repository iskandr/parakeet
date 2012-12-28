import abc
import numpy

class Ref(object):
  __meta__ = abc.ABCMeta

  @abc.abstractmethod
  def deref(self):
    pass

class GlobalRef(Ref):
  def __init__(self, globals_dict, name):
    self.globals_dict = globals_dict
    self.name = name

  def __str__(self):
    return "GlobalRef(%s)" % self.name

  def __repr__(self):
    return str(self)

  def deref(self):
    return self.globals_dict[self.name]

  def __eq__(self, other):
    return isinstance(other, GlobalRef) and \
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
      # TODO: not sure this is what we want to do, but...
      if isinstance(self.cell.cell_contents, numpy.ndarray) and \
         isinstance(other.cell.cell_contents, numpy.ndarray):
        return (self.cell.cell_contents == other.cell.cell_contents).all()
      else:
        return self.cell == other.cell
    return False
