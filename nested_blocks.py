class NestedBlocks(object):
  def __init__(self):
    self._blocks = []

  def push(self, block = None):
    if block is None:
      block = []
    self._blocks.append(block)

  def pop(self):
    return self._blocks.pop()

  def top(self):
    return self._blocks[-1]
  
  def current(self):
    return self._blocks[-1]

  def append(self, stmt):
    self._blocks[-1].append(stmt)
    
  def append_to_current(self, stmt):
    self._blocks[-1].append(stmt)

  def extend_current(self, stmts):
    self._blocks[-1].extend(stmts)

  def depth(self):
    return len(self._blocks)

  def __iadd__(self, stmts):
    if not isinstance(stmts, (list, tuple)):
      stmts = [stmts]
    self.extend_current(stmts)
    return self
