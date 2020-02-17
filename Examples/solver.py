import netgen.gui
from netgen.geom2d import unit_square
from ngsolve import *
import pandas as pd


class ConvectionDiffusion:
    def __init__(self, mesh, order, method):
        self.mesh = mesh
        self.order = order
        
    def finitespace(self):
        fes = L2(self.mesh, order=self.order, dgjumps=True)
        u, v = fes.TnT()
        h = specialcf.mesh_size

    def convection_diffusion(self, method):
        acd = BilinearForm(fes)
        if(method == 'dg'):  
            jump_u = u-u.Other()
            jump_v = v-v.Other()
            n = specialcf.normal(2) 
            mean_dudn = 0.5 * n * (grad(u) + grad(u.Other()))
            mean_dvdn = 0.5 * n * (grad(v) + grad(v.Other()))

            # diffusion
            diffusion = grad(u) * grad(v) * dx \
                +alpha * order ** 2/ h * jump_u * jump_v * dx(skeleton=True) \
                +(-mean_dudn * jump_v - mean_dvdn * jump_u) * dx(skeleton=True) \
                +alpha * order ** 2/h * u * v * ds(skeleton=True) \
                + (-n * grad(u) * v -n * grad(v) * u) * ds(skeleton=True)
            
            # convection
            self.b = CoefficientFunction((beta[0],beta[1]))
            dS = dx(element_boundary=True)
            uup = IfPos(self.b * n, u, u.Other())
            convection = -self.b * u * grad(v) * dx + self.b * n * uup * jump_v * dx(skeleton=True)

            acd += eps * diffusion + convection
            acd.Assemble()
            
        elif (method == 'hdg'):
            n = specialcf.normal(mesh.dim)
            dS = dx(element_boundary=True)
            
            # diffusion
            diffusion = grad(u) * grad(v) * dx + \
                alpha * order ** 2/h * jump_u * jump_v * dS + \
                (-grad(u) * n * jump_v - grad(v) * n * jump_u) * dS

            # convection
            self.b = CoefficientFunction((beta[0],beta[1]))
            uup = IfPos(self.b * n, u, uhat)
            convection = -self.b * u * grad(v) * dx + self.b * n * uup * jump_v * dS

            acd += eps * diffusion + convection
            acd.Assemble()

        elif (method == 'edg'):
            acd += eps * diffusion + convection
            acd.Assemble()
            pass

        elif (method == 'ehdg'):
            acd += eps * diffusion + convection
            acd.Assemble()
            pass

        else:
            pass
        


class EnrichmentProxy(CoefficientFunction):
    """
    Provide wrappers for grad/Other and multiplication of enrichment lists.
    """
    def __init__(self, func, enr_list):
        self.func = func
        self.enr_list = enr_list
        self.grad_list = [CoefficientFunction((coeff.Diff(x), coeff.Diff(y))) for coeff in self.enr_list ]
    
    def __call__(self):
        return self.func[0] + sum([self.func[i]*self.enr_list[i-1] for i in range(1,len(self.enr_list)+1)])
    def __mul__(self, other):
        if type(other) == EnrichmentProxy:
            return self() * other()
        else:
            return self() * other
    def __rmul__(self,other):
        return self.__mul__(other)    
    
    def __add__(self, other):
        if type(other) == EnrichmentProxy:
            return self() + other()
        else:
            return self() + other
    def __radd__(self,other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if type(other) == EnrichmentProxy:
            return self() - other()
        else:
            return self() - other
    def __rsub__(self,other):
        return self.__sub__(other)
    
    def grad(self):
        mygrad = oldgrad(self.func[0])
        for i in range(1,len(self.enr_list)):
            mygrad += self.func[i] * self.grad_list[i-1]
            mygrad += oldgrad(self.func[i])*self.enr_list[i-1]
        return mygrad
    
    def Other(self):
        return EnrichmentProxy([f.Other() for f in self.func],self.enr_list)
    
def grad(q):
    if type(q) == EnrichmentProxy:
        return q.grad()
    else:
        return oldgrad(q)
        


class EnrichmentSpaces:
    def __init__(self, Q, axes, mesh_size):
        self.Q = Q
        self.axes = axes
        self.mesh_size = mesh_size
    
    def mark_element(self):
        ba = BitArray(self.Q.ndof)        
        ba.Clear()
        for el in self.Q.Elements():
            mark = False
            for v in el.vertices:
                if (mesh[v].point[self.axes] > 1-(0.001 * self.mesh_size)):
                    mark = True
            for dof in el.dofs:
                ba[dof] = mark
            Qx = Compress(self.Q, active_dofs=ba)
        return Qx