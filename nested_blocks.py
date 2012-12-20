class NestedBlocks(object):
  def __init__(self):
    self._blocks = []

  def push(self):
    self._blocks.append([])

  def pop(self):
    return self._blocks.pop()

  def current(self):
    return self._blocks[-1]

  def append_to_current(self, stmt):
    self.current().append(stmt)

  def extend_current(self, stmts):
    self.current().extend(stmts)

  def depth(self):
    return len(self._blocks)

  def __iadd__(self, stmts):
    if not isinstance(stmts, (list, tuple)):
      stmts = [stmts]
    self.extend_current(stmts)
    return self
