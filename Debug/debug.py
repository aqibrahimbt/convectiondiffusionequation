from numpy.linalg import inv
from netgen.geom2d import unit_square
from ngsolve import *
from ngsolve.comp import ProxyFunction
from ngsolve import ngsglobals
ngsglobals.msg_level = 0
import math
from numpy import linalg as LA
from utils import *
from enrichment_proxy import *


from scipy import linalg
import pandas as pd
import scipy.sparse as spe
import scipy as sp
import numpy as np
import sympy

#import netgen.gui

# packages


# Problem configuration
beta = (2, 0.001)
eps = 0.01


def p(x): return x + (exp(beta[0]*(x-1)/eps) -
                      exp(-beta[0]/eps))/(exp(-beta[0]/eps)-1)


def q(y): return y + (exp(beta[1]*(y-1)/eps) -
                      exp(-beta[1]/eps))/(exp(-beta[1]/eps)-1)


exact = p(x) * q(y)
coeff = beta[1] * p(x) + beta[0] * q(y)

orders = [1, 2, 3, 4]
#order = 1
#mesh_size = [1.0, 0.5, 0.25, 0.125, 0.0625]
mesh_size = np.logspace(0, -1, num=20)
mesh_size = [0.0625]
epsilon = 0.01
bonus_int = 10
enrich_functions = [p(x), q(y)]
enrich_domain_ind = [lambda x, y, h: x > 1 - h, lambda x, y, h: y > 1 - h]
depend = 'yes'
# Without enrichment - empty array
#enrich_functions = []
#enrich_domain_ind = []

for size in mesh_size:
    for order in orders:
        mesh = Mesh(unit_square.GenerateMesh(maxh=size))
        V = L2(mesh, order=order, dgjumps=True)
        Q = L2(mesh, order=0)
        Vlist = [V]

        for enr_indicator in enrich_domain_ind or []:
            Vlist.append(mark_dofs(Q, mesh, enr_indicator, size))

        fes = FESpace(Vlist, dgjumps=True)

        u = EnrichmentProxy(fes.TrialFunction(), enrich_functions)
        v = EnrichmentProxy(fes.TestFunction(), enrich_functions)

        jump_u = u-u.Other()
        jump_v = v-v.Other()

        n = specialcf.normal(2)
        mean_dudn = 0.5 * n * (grad(u) + grad(u.Other()))
        mean_dvdn = 0.5 * n * (grad(v) + grad(v.Other()))

        h = specialcf.mesh_size
        dy = dx(bonus_intorder=bonus_int)
        dX = dx(skeleton=True, bonus_intorder=bonus_int)
        dS = ds(skeleton=True, bonus_intorder=bonus_int)

        ba_active_dofs = BitArray(fes.FreeDofs())
        ba_active_dofs[:] = fes.FreeDofs()

        alpha_stab = GridFunction(Q)
        gf_indicator = GridFunction(Q)

        if len(enrich_functions) > 0:
            ipintegrator = SymbolicBFI(u() * v(), bonus_intorder=bonus_int)

            ba_active_elements = BitArray(mesh.ne)
            ba_active_elements[:] = False

            for enr_indicator in enrich_domain_ind:
                ba_active_elements |= mark_elements(
                    mesh, enr_indicator, size)

            # Linear dependence check           
            for el in fes.Elements():
                if ba_active_elements[el.nr]:
                    N = len(el.dofs)
                    Nstd = V.GetFE(ElementId(el)).ndof
                    elmat = ipintegrator.CalcElementMatrix(
                        el.GetFE(), el.GetTrafo())
                    important = [True if el.dofs[i] >=
                                 0 else False for i in range(N)]
                    factors = []
                    for i in range(Nstd, N):
                        if important[i]:
                            active = [j for j in range(i)
                                      if important[j]]
                            factor = 1 - 2 * \
                                sum([elmat[i, j]**2/elmat[i, i] /
                                     elmat[j, j] for j in active])
                            factor += sum([elmat[i, j]*elmat[i, k]*elmat[j, k]/elmat[i, i] /
                                           elmat[j, j]/elmat[k, k] for j in active for k in active])
                            factor = sqrt(abs(factor))
                            factors.append(factor)
                            if (factor <= 1e-2):
                                important[i] = False
                                if el.dofs[i] >= 0:
                                    ba_active_dofs[el.dofs[i]] = False
            
            # stiffness matrix           
            a_diff = SymbolicBFI(grad(u) * grad(v), bonus_intorder=bonus_int)

            f = SymbolicLFI(h**((-2-mesh.dim)/2) * v, bonus_intorder=bonus_int)

            # mass matrix
            m = SymbolicBFI(h*(grad(u)*n)*(grad(v)*n),
                            element_boundary=True, bonus_intorder=bonus_int)

            for el in fes.Elements():
                a_elmat = (a_diff.CalcElementMatrix(
                    el.GetFE(), el.GetTrafo())).NumPy()

                f_elmat = (f.CalcElementVector(
                    el.GetFE(), el.GetTrafo())).NumPy()

                for i in range(len(f_elmat)):
                    for j in range(len(f_elmat)):
                        a_elmat[i, j] += f_elmat[i]*f_elmat[j]

                m_elmat = (m.CalcElementMatrix(
                    el.GetFE(), el.GetTrafo())).NumPy()

                
                important = []
                for i, dof in enumerate(el.dofs):
                    if dof > 0 and fes.FreeDofs()[dof]:
                        important.append(i)
                
                m1_elmat = m_elmat[np.ix_(important,important)]
                a1_elmat = a_elmat[np.ix_(important,important)]
                
                eigs = np.sort(sp.linalg.eigvals(m1_elmat, b=a1_elmat).real)

                # print(eigs)
                lmax = eigs[1]
                lmin = eigs[-1]

                print("before:", lmin, lmax, lmax/lmin)
                print(LA.cond(a1_elmat))
                important = []
                for i, dof in enumerate(el.dofs):
                    if dof > 0 and ba_active_dofs[dof]:
                        important.append(i)
                # print("a_elmat:", a_elmat)
                m2_elmat = m_elmat[np.ix_(important,important)]
                a2_elmat = a_elmat[np.ix_(important,important)]
                # print("a_elmat:", a_elmat)
                
                eigs = np.sort(sp.linalg.eigvals(m2_elmat, b=a2_elmat).real)
                #print(eigs)
                lmax = eigs[1]
                lmin = eigs[-1]
                print("after :", lmin, lmax, lmax/lmin)
                print(LA.cond(a2_elmat))
                input("")
                #print(LA.cond(a_elmat))
                #print(x.real)

        # alpha = CoefficientFunction(alpha_stab)

        # # symmetric diffusion equation
        # diffusion = grad(u) * grad(v) * dy \
        #     + alpha * order ** 2 / h * jump_u * jump_v * dX \
        #     + (-mean_dudn * jump_v - mean_dvdn * jump_u) * dX \
        #     + alpha * order ** 2/h * u * v * dS \
        #     + (-n * grad(u) * v - n * grad(v) * u) * dS

        # # convection equation
        # b = CoefficientFunction((beta[0],beta[1]) )
        # uup = IfPos(b * n, u(), u.Other()())
        # convection = -b * u * \
        #     grad(v) * dy + b * n * uup * jump_v * dX

        # # lhs
        # acd = BilinearForm(fes, symmetric=True)
        # acd += epsilon * diffusion + convection
        # with TaskManager():
        #     acd.Assemble()

        # # rhs
        # f = LinearForm(fes)
        # f += coeff * v * dy
        # with TaskManager():
        #     f.Assemble()

        # # solve the system
        # gfu = GridFunction(fes, name="uDG")
        # gfu.vec.data = acd.mat.Inverse(
        #     ba_active_dofs, inverse="pardiso") * f.vec

        # gfu = gfu.components[0] + sum([gfu.components[i+1] * enrich_functions[i] for i in range(len(enrich_functions))])

        # # error
        # error = sqrt(Integrate((gfu-exact) * (gfu-exact), mesh, order= 5 + bonus_int))

        # print('order: ', order, 'mesh_size: ', size, 'error: ', error)


# print('max', np.max(max_alpha))
# print('min', np.min(max_alpha))
