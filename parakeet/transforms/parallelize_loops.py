from loop_transform import LoopTransform
from transform import Transform

class ExtractParallelLoops(Transform):
  
  def __init__(self):
    Transform.__init__(self)
    
  def enter_loop(self, var):
    