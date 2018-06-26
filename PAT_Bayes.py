from __future__ import absolute_import, division, print_function

from hippylib import nb
import dolfin as dl
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0,'/home/fenics/Installations/MUQ_INSTALL/lib')

import pandas as pd


# MUQ Includes
import pymuqModeling as mm # Needed for Gaussian distribution
import pymuqApproximation as ma # Needed for Gaussian processes
import pymuqSamplingAlgorithms as ms # Needed for MCMC


#obsData = np.array( ??? )
numIntervals=10
logPriorMu = 10*np.ones(numIntervals)
logPriorCov = 4.0*np.eye(numIntervals)

# Create a gaussian and turn it into a density
logPrior = mm.Gaussian(logPriorMu, logPriorCov).AsDensity() 

nx = ny = 20
mesh = dl.RectangleMesh(dl.Point(0, 0),dl.Point(1,1),nx,ny,"right")
V = dl.FunctionSpace(mesh, 'P', 1)

class PAT_forward(mm.PyModPiece):
    
    def __init__(self, time_final, numSteps, c, V):
        """ 
        INPUTS:
        
        """
        mm.PyModPiece.__init__(self, [V.dim()],
                                [V.dim()])
                  
        self.T_f = time_final
        self.numSteps = numSteps
        self.dt = self.T_f / self.numSteps
        self.c = c
        self.V = V
        
        self.p_trial = dl.TrialFunction(self.V)
        self.v = dl.TestFunction(self.V)
            
    def EvaluateImpl(self, inputs):
        """
        
        """
        m = dl.Function(self.V)
        m.vector().set_local(inputs[0])
        
        p_n = dl.interpolate(m, self.V)
        p_nm1 = dl.interpolate(m, self.V)
        p_trial = self.p_trial
        v = self.v
        
        F = (self.c**2)*(self.dt**2)*dl.inner(dl.grad(p_trial), dl.grad(v))*dl.dx - 2.*p_n*v*dl.dx + p_trial*v*dl.dx + p_nm1*v*dl.dx 
        a, L = dl.lhs(F), dl.rhs(F)
        
        # Time-stepping
        p = dl.Function(self.V)
        t = 0
        for n in range(self.numSteps):
            # Update current timtime
            t += self.dt

            # Compute solution
            dl.solve(a == L, p)

            # Update previous solution
            p_nm1.assign(p_n)
            p_n.assign(p)

        out = p.vector().array()[:]
        
        self.outputs = [out]


        
time_final = 0.05
numSteps = 1000
c = 1.5
test = PAT_forward(time_final, numSteps, c, V)
m = dl.interpolate(dl.Expression('std::log( 8. - 4.*(pow(x[0] - .5,2) + pow(x[1] - .5,2) < pow(0.2,2) ) )', degree=5), V)
m3 = m.vector().array()[:]
vecout = test.Evaluate([m3])   

vecout
m = dl.Function(V)
m.vector().set_local(vecout[0])
nb.plot(m)
plt.show()
