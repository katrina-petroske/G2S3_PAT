import sys
sys.path.insert(0,'/home/fenics/Installations/MUQ_INSTALL/lib')
import pymuqModeling as mm
import dolfin as dl
import numpy as np

class PAT_forward(mm.PyModPiece):
    
    def __init__(self, time_final, numSteps, c, V, numObs):
        """ 
        INPUTS:
        
        
        """
        mm.PyModPiece.__init__(self, [V.dim()],[numObs * numSteps])
        
        np.random.seed(1337)
        self.obs_indices = np.random.choice(V.dim(), numObs)          
        self.T_f = time_final
        self.numSteps = numSteps
        self.dt = self.T_f / self.numSteps
        self.c = c
        self.V = V
        self.numObs = numObs
        self.p_trial = dl.TrialFunction(self.V)
        self.v = dl.TestFunction(self.V)
        
    def ObservationOperator(self, p):
        p_arr = p.vector().get_local()
        return p_arr[self.obs_indices]
            
    def EvaluateImpl(self, inputs):
        """
        
        """
        numObs = self.numObs
        numSteps = self.numSteps
        # Each 
        output = np.zeros((numObs * numSteps))
        m = dl.Function(self.V)
        m.vector().set_local(inputs[0])
        
        p_n = dl.Function(self.V)
        p_nm1 = dl.Function(self.V)
        p_n.vector().set_local(m.vector().array()[:])
        p_nm1.vector().set_local(m.vector().array()[:])
        p_trial = self.p_trial
        v = self.v
        
        F = (self.c**2)*(self.dt**2)*dl.inner(dl.grad(p_trial), dl.grad(v))*dl.dx - 2.*p_n*v*dl.dx + p_trial*v*dl.dx + p_nm1*v*dl.dx 
        a, L = dl.lhs(F), dl.rhs(F)
        
        # Time-stepping
        p = dl.Function(self.V)
        t = 0
        
        output[0:numObs] = self.ObservationOperator(p_nm1)
        output[numObs:2*numObs] = self.ObservationOperator(p_n)
        
        for n in range(2, self.numSteps):
            # Update current timtime
            t += self.dt

            # Compute solution
            dl.solve(a == L, p)
            
            output[n*numObs:(n+1)*numObs] = self.ObservationOperator(p)

            # Update previous solution
            p_nm1.assign(p_n)
            p_n.assign(p)           
        
        self.outputs = [output]