from ..ndtypes import Int32 
from ..syntax import SourceExpr, SourceStmt

class dim3(object):
  def __init__(self, x, y, z):
    self.x = x 
    self.y = y 
    self.z = z 
  
  def __iter__(self):
    yield self.x 
    yield self.y 
    yield self.z
    
  def __getitem__(self, idx):
    if idx == 0:
      return self.x 
    elif idx == 1:
      return self.y 
    else:
      assert idx == 2, "Unexpected index %s to %s" % (idx, self)
      return self.z
    
  def __str__(self):
    return "dim3(x = %s, y = %s, z = %s)" % (self.x, self.y, self.z) 
 
# pasted literally into the C source 
blockIdx_x = SourceExpr("blockIdx.x", type=Int32)
blockIdx_y = SourceExpr("blockIdx.y", type=Int32)
blockIdx_z = SourceExpr("blockIdx.z", type=Int32)
blockIdx = dim3(blockIdx_x, blockIdx_y, blockIdx_z)

threadIdx_x = SourceExpr("threadIdx.x", type=Int32)
threadIdx_y = SourceExpr("threadIdx.y", type=Int32)
threadIdx_z = SourceExpr("threadIdx.z", type=Int32)
threadIdx = dim3(threadIdx_x, threadIdx_y, threadIdx_z)

blockDim_x = SourceExpr("blockDim.x", type = Int32)
blockDim_y = SourceExpr("blockDim.y", type = Int32)
blockDim_z = SourceExpr("blockDim.z", type = Int32)
blockDim = dim3(blockDim_x, blockDim_y, blockDim_z)

gridDim_x = SourceExpr("gridDim.x", type = Int32)
gridDim_y = SourceExpr("gridDim.y", type = Int32)
gridDim_z = SourceExpr("gridDim.z", type = Int32)
gridDim = dim3(gridDim_x, gridDim_y, gridDim_z)

warpSize = SourceExpr("wrapSize", type=Int32)
__syncthreads = SourceStmt("__syncthreads;")

