from numpy import sin, cos, dot, array 


def rotate(phi, theta, orig): 
  """
  This function rotates the point at orig about axis u(depends on phi) by the angle theta.  
  orig and output are in Cartesian coordinates
  """
  
  u = (-sin(phi), cos(phi), 0.0)
  rotM = array([
                 [cos(theta)+u[0]**2*(1-cos(theta)), u[0]*u[1]*(1-cos(theta)), u[1]*sin(theta)],
                 [u[0]*u[1]*(1-cos(theta)), cos(theta)+u[1]**2*(1-cos(theta)), -u[0]*sin(theta)],
                 [-u[1]*sin(theta), u[0]*sin(theta), cos(theta)]
                ])
  rotP = dot(rotM,orig)
  return rotP


from parakeet import testing_helpers
import numpy as np 

def test_rotate():
  x = np.arange(3) / 2.0
  for phi in (0, 0.25):
    for theta in (0, 0.25):
      testing_helpers.expect(rotate, [phi, theta, x], rotate(phi, theta, x))

if __name__ == "__main__":
  testing_helpers.run_local_tests()
