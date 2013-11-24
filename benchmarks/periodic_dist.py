from parakeet import jit, config 
import numpy as np 

def dist(x, y, z, L, periodicX, periodicY, periodicZ):
    " ""Computes distances between all particles and places the result in a matrix such that the ij th matrix entry corresponds to the distance between particle i and j"" "
    N = len(x)
    xtemp = np.tile(x,(N,1))
    dx = xtemp - xtemp.T
    ytemp = np.tile(y,(N,1))
    dy = ytemp - ytemp.T
    ztemp = np.tile(z,(N,1))
    dz = ztemp - ztemp.T

    # Particles 'feel' each other across the periodic boundaries
    if periodicX:
        dx[dx>L/2]=dx[dx > L/2]-L
        dx[dx<-L/2]=dx[dx < -L/2]+L
    if periodicY:
        dy[dy>L/2]=dy[dy>L/2]-L
        dy[dy<-L/2]=dy[dy<-L/2]+L
    if periodicZ:
        dz[dz>L/2]=dz[dz>L/2]-L
        dz[dz<-L/2]=dz[dz<-L/2]+L

    # Total Distances
    d = np.sqrt(dx**2+dy**2+dz**2)

    # Mark zero entries with negative 1 to avoid divergences
    d[d==0] = -1

    return d, dx, dy, dz

@jit 
def parakeet_dist(x, y, z, L, periodicX, periodicY, periodicZ):
  N = len(x)
  def periodic_diff(x1, x2, periodic):
    diff = x1 - x2 
    if periodic:
      if diff > (L / 2):
        diff -= L
      if diff < (-L/2):
        diff += L
    return diff
  dx = np.array([[periodic_diff(x1, x2, periodicX) for x1 in x] for x2 in x])
  dy = np.array([[periodic_diff(y1, y2, periodicY) for y1 in y] for y2 in y])
  dz = np.array([[periodic_diff(z1, z2, periodicZ) for z1 in z] for z2 in z])
  d = np.sqrt(dx**2 + dy**2 + dz**2)
  for i in xrange(N):
    for j in xrange(N):
      if d[i,j] == 0:
        d[i,j] = -1 
  return d, dx, dy, dz 

def periodic_diff(x1, x2, L, periodic):
  diff = x1 - x2 
  if periodic:
    if diff > (L / 2):
      diff -= L
    if diff < (-L/2):
      diff += L
  return diff

@jit 
def loopy_dist(x, y, z, L, periodicX, periodicY, periodicZ):
  N = len(x)
  dx = np.zeros((N,N))
  dy = np.zeros( (N,N) )
  dz = np.zeros( (N,N) )
  d = np.zeros( (N,N) )
  for i in xrange(N):
    for j in xrange(N):
      dx[i,j] = periodic_diff(x[j], x[i], L, periodicX)
      dy[i,j] = periodic_diff(y[j], y[i], L, periodicY)
      dz[i,j] = periodic_diff(z[j], z[i], L, periodicZ)
      d[i,j] = dx[i,j] ** 2 + dy[i,j] ** 2 + dz[i,j] ** 2 
      if d[i,j] == 0:
          d[i,j] = -1
      else:
          d[i,j] = np.sqrt(d[i,j])
  return d, dx, dy, dz 
from timer import timer

N = 2000 
x = y = z = np.random.rand(N)
L = 4
periodic = True
with timer("Python #1"):
  d, dx, dy, dz = dist(x, x, x, L,periodic, periodic, periodic)

with timer("Python #2"):
  d, dx, dy, dz = dist(x, x, x, L, periodic, periodic, periodic)

with timer("Parakeet Dist #1", suppress_stdout = False, suppress_stderr = False):
  pd, pdx, pdy, pdz = parakeet_dist(x,x,x,L, periodic, periodic, periodic)

assert np.allclose(pdx, dx), (pdx-dx, np.max(np.abs(pdx - dx)))
assert np.allclose(pdy, dy), np.max(np.abs(pdy - dy))
assert np.allclose(pdz, dz), np.max(np.abs(pdz - dz))
assert np.allclose(pd, d), (np.max(np.abs(pd - d)), pd -d)


with timer("Parakeet Dist #2"):
  pd, pdx, pdy, pdz = parakeet_dist(x,x,x,L, periodic, periodic, periodic)


with timer("Parakeet Loop Dist #1"):
  ld, ldx, ldy, ldz = loopy_dist(x,x,x,L, periodic, periodic, periodic)

assert np.allclose(ldx, dx), (ldx-dx, np.max(np.abs(ldx - dx)))
assert np.allclose(ldy, dy), np.max(np.abs(ldy - dy))
assert np.allclose(ldz, dz), np.max(np.abs(ldz - dz))
assert np.allclose(ld, d), (np.max(np.abs(ld - d)), ld -d)

with timer("Parakeet Loop Dist #2"):
  ld, ldx, ldy, ldz = loopy_dist(x,x,x,L, periodic, periodic, periodic)

