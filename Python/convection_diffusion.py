import numpy as np
from scipy import linalg
import pandas as pd

from enrichment_proxy import *


"""
Solving the Convection-Diffusion using the Enriched (DG and HDG) method
TODO: Linear Dependence for HDG - Issue: (Division by zero)
BUG: Spurious kinks in error for both DG and HDG
"""


class Convection_Diffusion():

    # Set Default parameters
    config = {

    }

    def __init__(self, new_config={}):
        self.config.update(new_config)
        self.columns = ['Order', 'Mesh Size',
                        'Error', 'Alpha', 'Bonus Int', 'Type']
        self.results = pd.DataFrame(columns=self.columns)


    '''Enriched Discontinuous Galerkin Methods'''
    def _solveEDG(self):
        self.results.iloc[0:0]
        for order in self.config['order']:
            for size in self.config['mesh_size']:
                for alpha in self.config['alpha']:
                    for bonus_int in self.config['bonus_int_order']:
                        mesh = Mesh(unit_square.GenerateMesh(maxh=size))
                        V = L2(mesh, order=order, dgjumps=True)
                        Q = L2(mesh, order=0)
                        Vlist = [V]

                        for enr_indicator in self.config['enrich_domain_ind'] or []:
                            Vlist.append(mark_dofs(Q, mesh, enr_indicator, size))

                        fes = FESpace(Vlist, dgjumps=True)

                        u = EnrichmentProxy(
                            fes.TrialFunction(), self.config['enrich_functions'])
                        v = EnrichmentProxy(
                            fes.TestFunction(), self.config['enrich_functions'])

                        ba_active_dofs = BitArray(fes.FreeDofs())
                        ba_active_dofs[:] = fes.FreeDofs()

                        if len(self.config['enrich_functions']) > 0:
                            type = str('edg')
                            print(type)
                        # Checking linear dependence
                            ipintegrator = SymbolicBFI(
                                u() * v(), bonus_intorder=bonus_int)
                            ba_active_elements = BitArray(mesh.ne)
                            for enr_indicator in self.config['enrich_domain_ind']:
                                ba_active_elements |= mark_elements(
                                    mesh, enr_indicator, size)

                            for el in fes.Elements():
                                if ba_active_elements[el.nr]:
                                    i = ElementId(el)
                                    N = len(el.dofs)
                                    element = fes.GetFE(el)
                                    elementstd = V.GetFE(i)
                                    Nstd = elementstd.ndof
                                    trafo = mesh.GetTrafo(i)
                                    # Get element matrix
                                    elmat = ipintegrator.CalcElementMatrix(
                                        element, trafo)
                                    important = [True if el.dofs[i] >=
                                                0 else False for i in range(N)]
                                    before_important = [
                                        True if el.dofs[i] >= 0 else False for i in range(N)]

                                    factors = []
                                    for i in range(Nstd, N):
                                        if important[i]:
                                            active = [j for j in range(
                                                i) if important[j]]
                                            factor = 1 - 2 * \
                                                sum([elmat[i, j]**2/elmat[i, i] /
                                                    elmat[j, j] for j in active])
                                            factor += sum([elmat[i, j]*elmat[i, k]*elmat[j, k]/elmat[i, i] /
                                                        elmat[j, j]/elmat[k, k] for j in active for k in active])
                                            factor = sqrt(abs(factor))
                                            factors.append(factor)
                                            if (factor <= 1e-3):
                                                important[i] = False
                                                if el.dofs[i] >= 0:
                                                    ba_active_dofs[el.dofs[i]
                                                                ] = False
                        else:
                            type = str('dg')
                            print(type)
                        
                        jump_u = u-u.Other()
                        jump_v = v-v.Other()

                        n = specialcf.normal(2)
                        mean_dudn = 0.5 * n * (grad(u) + grad(u.Other()))
                        mean_dvdn = 0.5 * n * (grad(v) + grad(v.Other()))

                        h = specialcf.mesh_size
                        dy = dx(bonus_intorder=bonus_int)
                        dX = dx(skeleton=True, bonus_intorder=bonus_int)
                        dS = ds(skeleton=True, bonus_intorder=bonus_int)

                        # diffusion equation
                        diffusion = grad(u) * grad(v) * dy \
                            + alpha * order ** 2 / h * jump_u * jump_v * dX \
                            + (-mean_dudn * jump_v - mean_dvdn * jump_u) * dX \
                            + alpha * order ** 2/h * u * v * dS \
                            + (-n * grad(u) * v - n * grad(v) * u) * dS

                        # convection equation
                        b = CoefficientFunction(
                            (self.config['beta'][0], self.config['beta'][1]))
                        uup = IfPos(b * n, u(), u.Other()())
                        convection = -b * u * \
                            grad(v) * dy + b * n * uup * jump_v * dX

                        # lhs
                        acd = BilinearForm(fes)
                        acd += self.config['epsilon'] * diffusion + convection
                        with TaskManager():
                            acd.Assemble()

                        # rhs
                        f = LinearForm(fes)
                        f += self.config['coeff'] * v * dy
                        with TaskManager():
                            f.Assemble()

                        # solve the system
                        gfu = GridFunction(fes, name="uDG")
                        gfu.vec.data = acd.mat.Inverse(ba_active_dofs, inverse="pardiso") * f.vec

                        gfu = gfu.components[0] + sum([gfu.components[i+1]*self.config['enrich_functions'][i]
                                                       for i in range(len(self.config['enrich_functions']))])

                        # error
                        error = sqrt(Integrate(
                            (gfu-self.config['exact'])*(gfu-self.config['exact']), mesh, order=bonus_int))

                        self.results.loc[len(self.results)] = [
                            order, size, error, alpha, bonus_int, type]
                        print('Order:', order, 'Alpha:', alpha, 'Bonus Int:',
                              bonus_int, 'Mesh Size:', size, "L2-error:", error, 'type:', type)

        return self.results


    '''Enriched (Hybrid) Discontinuous Galerkin Methods'''
    def _solveEHDG(self):
        self.results.iloc[0:0]
        for order in self.config['order']:
            for size in self.config['mesh_size']:
                for alpha in self.config['alpha']:
                    for bonus_int in self.config['bonus_int_order']:
                        mesh = Mesh(unit_square.GenerateMesh(maxh=size))
                        V = L2(mesh, order=order)
                        F = FacetFESpace(mesh, order=order, dirichlet=".*")
                        Q = L2(mesh, order=0)
                        QF = FacetFESpace(mesh, order=0, dirichlet=".*")

                        Vlist = [V]
                        Vlist.append(F)

                        for enr_indicator in self.config['enrich_domain_ind'] or []:
                            Vlist.append(
                                mark_dofs(Q, mesh, enr_indicator, size))
                            Vlist.append(mark_element_bnd(
                                Q, QF, mesh, enr_indicator, size))

                        fes = FESpace(Vlist)  # [V, F, Qx, QFx, Qy, QFy])
                        u = EnrichmentProxy_VOL(
                            fes.TrialFunction(), self.config['enrich_functions'])
                        v = EnrichmentProxy_VOL(
                            fes.TestFunction(), self.config['enrich_functions'])

                        uhat = EnrichmentProxy_FAC(
                            fes.TrialFunction(), self.config['enrich_functions'])
                        vhat = EnrichmentProxy_FAC(
                            fes.TestFunction(), self.config['enrich_functions'])

                        h = specialcf.mesh_size
                        n = specialcf.normal(mesh.dim)
                        dS = dx(element_boundary=True,
                                bonus_intorder=bonus_int)
                        dy = dx(bonus_intorder=bonus_int)

                        ba_active_dofs = BitArray(fes.FreeDofs())
                        ba_active_dofs[:] = fes.FreeDofs()

                        if len(self.config['enrich_functions']) > 0:
                            type = 'ehdg'
                        else:
                            type = 'hdg'

                        # Checking linear dependence
                        # if len(self.config['enrich_functions']) > 0:
                        #     ipintegrator = SymbolicBFI(
                        #         u() * v(), bonus_intorder=bonus_int)
                        #     ba_active_elements = BitArray(mesh.ne)
                        #     for enr_indicator in self.config['enrich_domain_ind']:
                        #         ba_active_elements |= mark_elements(
                        #             mesh, enr_indicator, size)

                        #     for el in fes.Elements():
                        #         if ba_active_elements[el.nr]:
                        #             i = ElementId(el)
                        #             N = len(el.dofs)
                        #             element = fes.GetFE(el)
                        #             elementstd = V.GetFE(i)
                        #             Nstd = elementstd.ndof
                        #             trafo = mesh.GetTrafo(i)
                        #             # Get element matrix
                        #             elmat = ipintegrator.CalcElementMatrix(
                        #                 element, trafo)
                        #             important = [True if el.dofs[i] >=
                        #                         0 else False for i in range(N)]
                        #             before_important = [
                        #                 True if el.dofs[i] >= 0 else False for i in range(N)]

                        #             factors = []
                        #             for i in range(Nstd, N):
                        #                 if important[i]:
                        #                     active = [j for j in range(
                        #                         i) if important[j]]
                        #                     try:
                        #                         factor = 1 - 2 * \
                        #                             sum([elmat[i, j]**2/elmat[i, i] /
                        #                                 elmat[j, j] for j in active])
                        #                         factor += sum([elmat[i, j]*elmat[i, k]*elmat[j, k]/elmat[i, i] /
                        #                                     elmat[j, j]/elmat[k, k] for j in active for k in active])
                        #                         factor = sqrt(abs(factor))
                        #                         factors.append(factor)
                        #                         if (factor <= 1e-3):
                        #                             important[i] = False
                        #                             if el.dofs[i] >= 0:
                        #                                 ba_active_dofs[el.dofs[i]
                        #                                             ] = False
                        #                     except:
                        #                         ba_active_dofs[el.dofs[i]] = True

                        jump_u = u-uhat()
                        jump_v = v-vhat()

                        # diffusion
                        diffusion = grad(u) * grad(v) * dy \
                            + alpha * order ** 2/h * jump_u * jump_v * dS \
                            + (-grad(u) * n * jump_v - grad(v) * n * jump_u) * dS

                        # convection
                        b = CoefficientFunction(
                            (self.config['beta'][0], self.config['beta'][1]))
                        uup = IfPos(b * n, u(), uhat())
                        convection = -b * u * \
                            grad(v) * dy + b * n * uup * jump_v * dS + \
                            IfPos(b * n, (uhat()-u)*vhat(), 0) * dS

                        # lhs
                        acd = BilinearForm(fes, symmetric=False)
                        acd += self.config['epsilon'] * diffusion + convection
                        with TaskManager():
                            acd.Assemble()

                        # rhs
                        f = LinearForm(fes)
                        f += self.config['coeff'] * v * dy
                        with TaskManager():
                            f.Assemble()

                        gfu = GridFunction(fes)
                        gfu.vec.data = acd.mat.Inverse(
                            ba_active_dofs, inverse="pardiso") * f.vec

                        gfu = gfu.components[0] + sum([gfu.components[2*i+2] * self.config['enrich_functions'][i]
                                                       for i in range(len(self.config['enrich_functions']))])

                        # Draw(gfu.components[0],mesh,"u")

                        error = sqrt(Integrate(
                            (gfu-self.config['exact'])*(gfu-self.config['exact']), mesh, order=30 + bonus_int))
                        
                        self.results.loc[len(self.results)] = [
                            order, size, error, alpha, bonus_int, type]

                        print('Order:', order, 'Alpha:', alpha, 'Bonus Int:',
                              bonus_int, 'Mesh Size:', size, "L2-error:", error, "type:", type)
        return self.results
