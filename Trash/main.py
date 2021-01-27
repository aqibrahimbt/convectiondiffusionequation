from netgen.geom2d import unit_square
from ngsolve import *
import numpy as np
from scipy import linalg
import pandas as pd
import numpy as np
import scipy as sp

mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))

order=4
fes1 = L2(mesh, order=order, dgjumps=True)
fes2 = L2(mesh, order=order)
fes = FESpace([fes1, fes2])
u,v = fes.TnT()

alpha = 4
h = specialcf.mesh_size
n = specialcf.normal(2)

bonus_int = 20

a_diff = SymbolicBFI(grad(u) * grad(v), bonus_intorder=bonus_int) 
fee = SymbolicLFI(h**((-2-order)/2)* v, bonus_intorder=bonus_int) 
                            
# mass
m = SymbolicBFI(h * (grad(u) * n)*(grad(v) * n), element_boundary=True, bonus_intorder=bonus_int)

for el in fes.Elements():
    a_elmat = (a_diff.CalcElementMatrix(el.GetFE(),el.GetTrafo())).NumPy()
                                
    f_elmat = (fee.CalcElementVector(el.GetFE(),el.GetTrafo())).NumPy()
    for i in range(len(f_elmat)):
        for j in range(len(f_elmat)):
            a_elmat[i,j] += f_elmat[i]*f_elmat[j]
            
            m_elmat = (m.CalcElementMatrix(el.GetFE(),el.GetTrafo())).NumPy()
            
            #x = np.max(np.linalg.eig(np.linalg.inv(a_elmat)@m_elmat)[0])
            L = np.linalg.cholesky(a_elmat) ## test for positive definiteness
            x = np.max((sp.linalg.eig(m_elmat,b=a_elmat))[0]) # this gives a complex value 
            alpha_stab = GridFunction(L2(mesh))
            # print(x.real)