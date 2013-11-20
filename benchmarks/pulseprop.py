"""
NMR Pulse propagation 
From http://themodernscientist.com/posts/2013/2013-06-09-simulation_of_nmr_shaped_pulses/
"""

import numpy as np 

pulseLength = 1000. # in microseconds
offset = [-5000., 5000.] # in hertz
n_freq = 500
inputMagnetization = 'Mz' # 'Mx', 'My', or 'Mz'


deltaomega = np.abs(offset[1]-offset[0])/n_freq
relativeomega = np.arange(np.min(offset),np.max(offset),deltaomega)

n_pulse = 1000 # number of points in the pulse, set by user
totalRotation = 180. # in degrees, set by user

fourierCoeffA = np.array([0.49, -1.02, 1.11, -1.57, 0.83, -0.42, 0.26, -0.16, 0.10, -0.07, 0.04, -0.03, 0.01, -0.02, 0.0, -0.01])
x = np.linspace(1,n_pulse,n_pulse)/n_pulse*2.*np.pi
nCosCoef = np.arange(1,len(fourierCoeffA))
cosMat = np.cos(nCosCoef[np.newaxis,:]*x[:,np.newaxis])
cosMat = np.append(np.ones((n_pulse,1)),cosMat,axis=1)*fourierCoeffA
sumMat = np.sum(cosMat,axis=1)

pulseShapeArray = np.zeros((n_pulse,2))
pulseShapeArray[:,0] = np.abs(sumMat)
pulseShapeArray[sumMat<0,1] = 180.

pulseShapeInten = pulseShapeArray[:,0] / np.max(np.abs(pulseShapeArray[:,0]))
pulseShapePhase = pulseShapeArray[:,1] * np.pi/180

xPulseShape = pulseShapeInten * np.cos(pulseShapePhase)
yPulseShape = pulseShapeInten * np.sin(pulseShapePhase)

scalingFactor = np.sum(xPulseShape)/n_pulse
gammaB1max = 1./(pulseLength * 360./totalRotation)/scalingFactor * 1e6
nu1maxdt = 2*np.pi*1e-6*gammaB1max*pulseLength/n_pulse

inputVector = np.array([[0],[0],[1]])
inputMagnetizationDict = {'mx':np.array([[1],[0],[0]]), 'my':np.array([[0],[1],[0]]), 'mz':np.array([[0],[0],[1]]) }
if inputMagnetization.lower() in inputMagnetizationDict.keys():
        inputVector = inputMagnetizationDict[inputMagnetization.lower()]
vectorComponent = inputVector.argmax()

def pulseprop(relativeomega, pulseShapeInten, pulseShapePhase, gammaB1max, nu1maxdt, inputVector, n_pulse, n_freq):
    # Functions for the y and z-rotations and the the function for a generic rotation
    def yrotation(beta):
      return np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    def zrotation(beta):
      return np.array([[np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0], [0, 0, 1]])
    def grotation(alpha,theta,phi):
      return np.dot(zrotation(phi),
                    np.dot(yrotation(theta),
                           np.dot(zrotation(alpha), 
                                  np.dot(yrotation(-theta),zrotation(-phi)))))
    
    
    xyzdata = np.zeros((3,len(relativeomega)))
    phi = pulseShapePhase
    
    # Loop through the entire frequency range calculating the rotation matrix (r) at each frequency
    for ind in range(len(relativeomega)):
    
        theta = np.arctan2(pulseShapeInten, relativeomega[ind]/gammaB1max)
        alpha = nu1maxdt * np.sqrt(pulseShapeInten**2+(relativeomega[ind]/gammaB1max)**2)
        
        prop = np.eye(3)
        # The rotation matrix is a recursive loop through each step of the shaped pulse
        for pulseindex in range(n_pulse):
            r = grotation(alpha[pulseindex],theta[pulseindex],phi[pulseindex])
            prop = np.dot(r,prop)
        res = np.dot(prop, inputVector)
        xyzdata[:,ind] = np.dot(prop,inputVector)[:, 0]
        
    return xyzdata

from compare_perf import compare_perf 
from parakeet import jit 
f = jit(pulseprop)
f(relativeomega, pulseShapeInten, pulseShapePhase, gammaB1max, nu1maxdt, inputVector, n_pulse, n_freq)
