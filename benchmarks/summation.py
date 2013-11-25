     
import numpy as np 
def summation(pos, weights, points):
  n_points = len(points)
  n_weights = len(weights)
  sum_array = np.zeros(n_points)
  sum_array3d = np.zeros((n_points,3))
  def compute(i):
    pxi = points[i, 0]
    pyi = points[i, 1]
    pzi = points[i, 2]
    total = 0.0
    for j in xrange(n_weights):
      weight_j = weights[j]
      xj = pos[j,0]
      yj = pos[j,1]
      zj = pos[j,2]
      dx = pxi - pos[j, 0]
      dy = pyi - pos[j, 1]
      dz = pzi - pos[j, 2]
      dr = 1.0/np.sqrt(dx*dx + dy*dy + dz*dz)
      total += weight_j * dr
      sum_array3d[i,0] += weight_j * dx
      sum_array3d[i,1] += weight_j * dy
      sum_array3d[i,2] += weight_j * dz
    return total 
  sum_array = np.array([compute(i) for i in xrange(n_points)])
  return sum_array, sum_array3d

n_points = 200
n_weights = 400
pos = np.random.randn(n_weights, 3)
weights = np.random.randn(n_weights)
points = np.random.randn(n_points, 3)

from compare_perf import compare_perf 

compare_perf(summation, [pos, weights, points])
