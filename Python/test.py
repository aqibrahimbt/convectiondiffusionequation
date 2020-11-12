#import netgen.gui
from netgen.geom2d import unit_square
from ngsolve import *
import pandas as pd
from ngsolve import grad as oldgrad
from ngsolve import ngsglobals
ngsglobals.msg_level = 0
import numpy as np



## Parameter setup
orders = [1, 2, 3, 4]
beta = (2,0.001)
mesh_size = np.logspace(0,-1,num=50)
#mesh_size = [1.0, 0.5, 0.25, 0.125]
eps = 0.01

# Exact Solution
#p = lambda x: x + (exp(beta[0]*x/eps)-1)/(1-exp(beta[0]/eps))
#q = lambda y: y + (exp(beta[1]*y/eps)-1)/(1-exp(beta[1]/eps))

p = lambda x: x + (exp(beta[0]*(x-1)/eps)-exp(-beta[0]/eps))/(exp(-beta[0]/eps)-1)
q = lambda y: y + (exp(beta[1]*(y-1)/eps)-exp(-beta[1]/eps))/(exp(-beta[1]/eps)-1)

exact = p(x) * q(y)

coeff =  beta[1] * p(x) +  beta[0] * q(y)

coeff_x = p(x)
coeff_y = q(y)

alpha = 5

## Dataframes
columns = ['Order', 'Mesh Size', 'Error', 'Alpha', 'Bonus Int', 'Type']



ehdg = pd.DataFrame(columns=columns)
val = 10
for order in orders:
    for size in mesh_size:        
        mesh = Mesh(unit_square.GenerateMesh(maxh=size))
        V = L2(mesh, order=order)
        Q = L2(mesh, order=0)
        QF = FacetFESpace(mesh, order=0)
        
        eps_size = size / 2
        
        ba_x = BitArray(Q.ndof)        
        ba_x.Clear()
        
        for el in Q.Elements():
            mark = False
            for v in el.vertices:
                if (mesh[v].point[0] > 1-eps_size):
                    mark = True
            for dof in el.dofs:
                ba_x[dof] = mark
        
        ba_y = BitArray(Q.ndof)
        ba_y.Clear()
        
        for el in Q.Elements():
            mark = False
            for v in el.vertices:
                if (mesh[v].point[1] > 1-eps_size):
                    mark = True
            for dof in el.dofs:
                ba_y[dof] = mark
        
        
        gfF = GridFunction(QF)
        
        gfF.vec[:] = 0
        for el in QF.Elements():
            if ba_x[el.nr]:
                for dof in el.dofs:
                    gfF.vec[dof] += 1
        ba_F_x = BitArray(QF.ndof)
        ba_F_x.Clear()
        for i in range(QF.ndof):
            if gfF.vec[i] > 1.5:
                ba_F_x[i] = True

        gfF.vec[:] = 0
        for el in QF.Elements():
            if ba_y[el.nr]:
                for dof in el.dofs:
                    gfF.vec[dof] += 1
        ba_F_y = BitArray(QF.ndof)
        ba_F_y.Clear()
        for i in range(QF.ndof):
            if gfF.vec[i] > 1.5:
                ba_F_y[i] = True
        
        Qx = Compress(Q, active_dofs=ba_x)
        Qy = Compress(Q, active_dofs=ba_y)
        QFx = Compress(QF, active_dofs=ba_F_x)
        QFy = Compress(QF, active_dofs=ba_F_y)
        
        F = FacetFESpace(mesh, order=order, dirichlet=".*")
        fes = FESpace([V, Qx, Qy, F, QFx, QFy])

        (us, px, py, uhat, uhatx, uhaty), (vs, qx, qy, vhat, vhatx, vhaty) = fes.TnT()
        
        #coeff=exact
        p = (coeff_x * px) + (coeff_y * py) 
        q = (coeff_x * qx) + (coeff_y * qy)

        u = us + p
        v = vs + q
        
        vhat = vhat + vhatx * coeff_x + vhaty * coeff_y
        uhat = uhat + uhatx * coeff_x + uhaty * coeff_y
        
        jump_u = u-uhat
        jump_v = v-vhat
        
        grad_u = grad(us) \
        + CoefficientFunction((coeff_x.Diff(x), coeff_x.Diff(y))) * px \
        + CoefficientFunction((coeff_y.Diff(x), coeff_y.Diff(y))) * py \
        + coeff_x * grad(px) + coeff_y * grad(py)
        
        grad_v = grad(vs) \
        + CoefficientFunction((coeff_x.Diff(x), coeff_x.Diff(y))) * qx \
        + CoefficientFunction((coeff_y.Diff(x), coeff_y.Diff(y))) * qy \
        + coeff_x * grad(qx) + coeff_y * grad(qy)
        
        #condense = True
        
        h = specialcf.mesh_size
        n = specialcf.normal(mesh.dim)
        dS = dx(element_boundary=True, bonus_intorder=val)
        dy = dx(bonus_intorder=val)

                
        diffusion = grad_u * grad_v *dy(bonus_intorder=val) + \
            alpha * order**2/h*jump_u*jump_v*dS + \
            (-grad_u *n*jump_v - grad_v *n*jump_u)*dS
        
        b = CoefficientFunction((beta[0],beta[1]))
        uup = IfPos(b * n, u, uhat)
        convection = -b * u * grad_v * dy(bonus_intorder=val) + b * n * uup * jump_v * dS

        acd = BilinearForm(fes)
        #acd += convection
        acd += eps * diffusion + convection
        
        with TaskManager():
            acd.Assemble()

        f = LinearForm(fes)
        f += SymbolicLFI(coeff*v,bonus_intorder=val)
        
        with TaskManager():
            f.Assemble()

        gfu = GridFunction(fes)
        gfu.vec.data = acd.mat.Inverse(freedofs=fes.FreeDofs(),inverse="pardiso") * f.vec
        u2 = gfu.components[0] + gfu.components[1] * coeff_x + gfu.components[2] * coeff_y
        
        error = sqrt (Integrate ((u2 - exact)*(u2- exact), mesh, order=30 + val))
        ehdg.loc[len(ehdg)] = [order, size, error, alpha, 0, 'hdg']
        
        #Draw(u2,mesh,"u",sd=5)
        #netgen.gui.Snapshot(w=800,h=500, filename='Images/' + "ehdg-h_"+str(size)+"-k_"+str(order)+".png")
        
        print ('Order:', order, 'Alpha:', alpha, 'Bonus Int:', val, 'Mesh Size:', size , "L2-error:", error)
    print('.......................................................................')
ehdg.to_csv('old.csv')