class ScopedSet(object):
  def __init__(self, initial_set = None):
    self.sets = []
    self.push(initial_set if initial_set is not None else [])

  def push(self, new_set = None):
    if new_set is None:
      new_set = set([])
    elif not isinstance(new_set, set):
      new_set = set(new_set)
    self.sets.append(new_set)
        
  def pop(self):
    return self.sets.pop()
  
  def top(self):
    return self.sets[-1]
  
  def add(self, value):
    self.top().add(value)
  
  def update(self, values):
    self.top().update(values)
      
  def __contains__(self, value):
    return value in self.top()
  
  def __str__(self):
    return "ScopedSet(%s)" % ", ".join(str(s) for s in self.sets)
  