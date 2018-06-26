from __future__ import absolute_import, division, print_function

from hippylib import nb
import dolfin as dl
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


T = .005 # final time
num_steps = 10000 # number of time steps
dt = T / num_steps # time step size
c = 1500    #speed of sound

# Create mesh and define function space
nx = ny = 60
mesh = dl.RectangleMesh(dl.Point(0, 0),dl.Point(10,10),nx,ny,"right")
V = dl.FunctionSpace(mesh, 'P', 1)
# Define boundary condition
m = dl.interpolate(dl.Expression('std::log( 8. - 4.*(pow(x[0] - 5.,2) + pow(x[1] - 5.,2) < pow(0.2,2) ) )', degree=5), V)#dl.Constant(1.0)
p_n = dl.interpolate(m, V)
p_nm1 = dl.interpolate(m, V)
nb.plot(p_n, vmin=1., vmax = 3.)
plt.show()
#p_n = dl.project(m, V)
# Define variational problem
p = dl.TrialFunction(V)
v = dl.TestFunction(V)
#f = dl.Constant(beta - 2 - 2*alpha)
F = c**2*dt**2*dl.inner(dl.grad(p), dl.grad(v))*dl.dx - 2.*p_n*v*dl.dx + p*v*dl.dx + p_nm1*v*dl.dx 
a, L = dl.lhs(F), dl.rhs(F)
# Time-stepping
p = dl.Function(V)
t = 0
for n in range(num_steps):
    # Update current timtime
    t += dt
    #u_D.t = t
    
    # Compute solution
    dl.solve(a == L, p)
    
    if (n % 100)== 0 :
        nb.plot(p, vmin=1., vmax = 3.)
        plt.show()
    # Compute error at vertices
#     u_e = dl.interpolate(u_D, V)
#     error = np.abs(u_e.vector().array() - u.vector().array()).max()
#     print('t = %.2f: error = %.3g' % (t, error))
    # Update previous solution
    p_nm1.assign(p_n)
    p_n.assign(p)
    
  
