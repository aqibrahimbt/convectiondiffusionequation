#import netgen.gui
from netgen.geom2d import unit_square
from ngsolve import *
from ngsolve import grad as oldgrad
import numpy as np
from ngsolve import ngsglobals
ngsglobals.msg_level = 0
import scipy as sp
import sympy 

## Parameter setup
orders = [1]
beta = (2,0.001)
mesh_size = [0.125]
eps = 0.01
bonus_int = 10

p = lambda x: x + (exp(beta[0]*(x-1)/eps)-exp(-beta[0]/eps))/(exp(-beta[0]/eps)-1)
q = lambda y: y + (exp(beta[1]*(y-1)/eps)-exp(-beta[1]/eps))/(exp(-beta[1]/eps)-1)

exact = p(x) * q(y)

coeff =  beta[1] * p(x) +  beta[0] * q(y)

coeff_x = p(x)
coeff_y = q(y)

alpha = 10

val = 15 # bonusintorder
for order in orders:
    for size in mesh_size:
        eps_size = 1e-8
        mesh = Mesh(unit_square.GenerateMesh(maxh=size))
        V = L2(mesh, order=order, dgjumps=True)
        Q = L2(mesh, order=0)
        F = FacetFESpace(mesh, order=order, dirichlet="bottom")
        
        ba_x = BitArray(Q.ndof)        
        ba_x.Clear()
        
        for el in Q.Elements():
            mark = False
            for v in el.vertices:
                if (mesh[v].point[0] > 1 - eps_size):
                    mark = True
            for dof in el.dofs:
                ba_x[dof] = mark
        
        ba_y = BitArray(Q.ndof)
        ba_y.Clear()
        
        
        for el in Q.Elements():
            mark = False
            for v in el.vertices:
                if (mesh[v].point[1] > 1 - eps_size):
                    mark = True
            for dof in el.dofs:
                ba_y[dof] = mark
        
        Qx = Compress(Q, active_dofs=ba_x)
        Qy = Compress(Q, active_dofs=ba_y)
        
        fes = FESpace([V, Qx, Qy], dgjumps = True)

        (us, px, py), (vs, qx, qy) = fes.TnT()
        
        ## Enrichment
        p = (coeff_x * px) + (coeff_y * py)
        q = (coeff_x * qx) + (coeff_y * qy)

        u = us + p
        v = vs + q
        
        u_Other = us.Other() + coeff_x * px.Other() + coeff_y * py.Other()
        v_Other = vs.Other() + coeff_x * qx.Other() + coeff_y * qy.Other()
        
        n = specialcf.normal(2)
        
        grad_u = grad(us) \
        + CoefficientFunction((coeff_x.Diff(x), coeff_x.Diff(y))) * px \
        + CoefficientFunction((coeff_y.Diff(x), coeff_y.Diff(y))) * py \
        + coeff_x * grad(px) + coeff_y * grad(py)
        
        grad_v = grad(vs) \
        + CoefficientFunction((coeff_x.Diff(x), coeff_x.Diff(y))) * qx \
        + CoefficientFunction((coeff_y.Diff(x), coeff_y.Diff(y))) * qy \
        + coeff_x * grad(qx) + coeff_y * grad(qy)
        
        grad_uOther = grad(us.Other()) \
        + CoefficientFunction((coeff_x.Diff(x), coeff_x.Diff(y))) * px.Other() \
        + CoefficientFunction((coeff_y.Diff(x), coeff_y.Diff(y))) * py.Other() \
        + coeff_x * grad(px.Other()) + coeff_y * grad(py.Other())
        
        grad_vOther = grad(vs.Other()) \
        + CoefficientFunction((coeff_x.Diff(x), coeff_x.Diff(y))) * qx.Other() \
        + CoefficientFunction((coeff_y.Diff(x), coeff_y.Diff(y))) * qy.Other() \
        + coeff_x * grad(qx.Other()) + coeff_y * grad(qy.Other())
        
        h = specialcf.mesh_size

        a_diff = SymbolicBFI(grad_u * grad_v, bonus_intorder=bonus_int, element_boundary=True) 
        
        fee = SymbolicLFI(h ** ((-2-order)/2)* v) 
                            
        # mass
        m = SymbolicBFI(h * (grad_u * n)*(grad_v * n), element_boundary=True, bonus_intorder=bonus_int)

        for el in fes.Elements():
            a_elmat = (a_diff.CalcElementMatrix(el.GetFE(),el.GetTrafo())).NumPy()

            #print((a_elmat.transpose() == a_elmat).all())

            f_elmat = (fee.CalcElementVector(el.GetFE(),el.GetTrafo())).NumPy()

            for i in range(len(f_elmat)):
                for j in range(len(f_elmat)):
                        a_elmat[i,j] += f_elmat[i]* f_elmat[j]

            m_elmat = (m.CalcElementMatrix(el.GetFE(),el.GetTrafo())).NumPy()
          
            print("lambda_max =", np.max(np.linalg.eig(np.linalg.pinv(a_elmat)@m_elmat)[0]))
            print("lambda_max =", np.max(sp.linalg.eig(m_elmat,b=a_elmat)[0]))

            alpha_stab = GridFunction(L2(mesh))
            alpha_stab.vec[el.nr] = np.max(np.linalg.eig(np.linalg.pinv(a_elmat)@m_elmat)[0])
        