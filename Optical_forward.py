import sys
sys.path.insert(0,'/home/fenics/Installations/MUQ_INSTALL/lib')
import pymuqModeling as mm
import dolfin as dl
import numpy as np
from hippylib import nb
import matplotlib.pyplot as plt

class Optical_forward(mm.PyModPiece):
    
    def __init__(self, V, gamma, Gamma):
        """ 
        INPUTS:
            V -- Function space of solution and parameters

        
        """
        mm.PyModPiece.__init__(self, [V.dim(), V.dim()],[V.dim()])
        
        self.Gamma = Gamma
        self.V = V
        self.p_trial = dl.TrialFunction(self.V)
        self.v = dl.TestFunction(self.V)
        self.gamma = gamma
        self.u_trial = dl.TrialFunction(V)
        self.sigma_trial = dl.TrialFunction(V)
        self.mu_trial = dl.TrialFunction(V)
        self.u_test = dl.TestFunction(V)
        self.sigma_test = dl.TestFunction(V)
        self.mu_test = dl.TestFunction(V)
            
    def EvaluateImpl(self, inputs):
        """
        INPUTS:
            input[0] -- nodal values of sigma, photon absorption coefficient
            input[1] -- nodal values of mu, two-photon absorption coefficient
        
        """
        sigma = dl.Function(self.V)
        sigma.vector().set_local(inputs[0])
        mu = dl.Function(self.V)
        mu.vector().set_local(inputs[1])
        
        def boundary(x,on_boundary):
            return on_boundary

        g = dl.Constant(1.0)
        bc_state = dl.DirichletBC(self.V, g, boundary)
        
        u = dl.Function(self.V)
        F_fwd = dl.inner(self.gamma * dl.grad(u), dl.grad(self.u_test)) * dl.dx + \
            sigma * u * self.u_test * dl.dx + \
            mu * abs(u) * u * self.u_test * dl.dx 
        
        dl.solve(F_fwd == 0, u, bcs = bc_state, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})
        
        H = dl.project(self.Gamma * sigma * u + self.Gamma * mu * u * abs(u), self.V)
        
        nb.plot(H)
        plt.title("Output H")
        plt.show()
        
        self.outputs = [H.vector().get_local()]