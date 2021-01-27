#import netgen.gui
from netgen.geom2d import unit_square
from ngsolve import *
#import pandas as pd
from ngsolve import grad as oldgrad
import numpy as np
from ngsolve import ngsglobals
ngsglobals.msg_level = 0

## Parameter setup
orders = [1]
beta = (2,0.001)
mesh_size = [0.5]
#mesh_size = [0.0313]
eps = 0.01
bonus_int = 10

for order in orders:
    for size in mesh_size:
        mesh = Mesh(unit_square.GenerateMesh(maxh=size))
        fes = L2(mesh, order=order, dgjumps=True, complex=True)
        #fes = FESpace([h1,number])

        u, v = fes.TnT()

        n = specialcf.normal(2)

        h = specialcf.mesh_size

        a_diff = SymbolicBFI(grad(u) * grad(v), bonus_intorder=bonus_int) 
        
        fee = SymbolicLFI(h**((-2-order)/2)* v, bonus_intorder=bonus_int) 
                            
                        # mass
        #m = SymbolicBFI(h * (grad(u) * n)*(grad(v) * n), element_boundary=True, bonus_intorder=bonus_int)

        for el in fes.Elements():
            a_elmat = (a_diff.CalcElementMatrix(el.GetFE(),el.GetTrafo())).NumPy()

            print(a_elmat)
            print((a_elmat.transpose() == a_elmat).all())

            #print(a_elmat)
                                
            f_elmat = (fee.CalcElementVector(el.GetFE(),el.GetTrafo())).NumPy()

            for i in range(len(f_elmat)):
                for j in range(len(f_elmat)):
                        a_elmat[i,j] += f_elmat[i]* f_elmat[j]

            L = np.linalg.cholesky(a_elmat)