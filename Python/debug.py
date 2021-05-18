import numpy as np
from numpy.linalg.linalg import norm
import pandas as pd
import scipy as sp
import netgen.gui
from numpy.linalg import pinv
from ngsolve import ngsglobals
import numpy as np
ngsglobals.msg_level = 0



from enrichment_proxy import *


"""
Solving the Convection-Diffusion using the Enriched (DG and HDG) method
"""


class Convection_Diffusion():

    # Set Default parameters
    config = {

    }

    def __init__(self, new_config={}):
        self.config.update(new_config)
        self.columns = ['Order', 'DOFs', 'Mesh Size',
                        'Error', 'Bonus Int', 'Type']
        self.col = ['alpha', 'type', 'mesh_size', 'order']
        self.results = pd.DataFrame(columns=self.columns)
        self.alphas = pd.DataFrame(columns=self.col)

    '''Enriched Discontinuous Galerkin Methods'''

    def _solveEDG(self):
        self.results.iloc[0:0]
        self.alphas.iloc[0:0]
        for order in self.config['order']:
            for size in self.config['mesh_size']:
                mesh = Mesh(unit_square.GenerateMesh(maxh=size))
                V = L2(mesh, order=order, dgjumps=True)
                Q = L2(mesh, order=0)
                Vlist = [V]

                for enr_indicator in self.config['enrich_domain_ind'] or []:
                    Vlist.append(
                        mark_dofs(Q, mesh, enr_indicator, size))

                fes = FESpace(Vlist, dgjumps=True)

                u = EnrichmentProxy(
                    fes.TrialFunction(), self.config['enrich_functions'])
                v = EnrichmentProxy(
                    fes.TestFunction(), self.config['enrich_functions'])

                gf_indicator = GridFunction(Q)
                gf_indicator.vec[:] = 0.0

                jump_u = u-u.Other()
                jump_v = v-v.Other()

                n = specialcf.normal(2)
                mean_dudn = 0.5 * n * (grad(u) + grad(u.Other()))
                mean_dvdn = 0.5 * n * (grad(v) + grad(v.Other()))

                h = specialcf.mesh_size
                dy = dx(bonus_intorder=self.config['bonus_int'])
                dX = dx(skeleton=True, bonus_intorder=self.config['bonus_int'])
                dS = ds(skeleton=True, bonus_intorder=self.config['bonus_int'])

                ba_active_dofs = BitArray(fes.FreeDofs())
                ba_active_dofs[:] = fes.FreeDofs()

                if len(self.config['enrich_functions']) > 0:
                    type = str('edg')
                    print('hello world')
                    ipintegrator = SymbolicBFI(
                        u() * v(), bonus_intorder=self.config['bonus_int'])
                        
                    ba_active_elements = BitArray(mesh.ne)
                    ba_active_elements[:] = False

                    for enr_indicator in self.config['enrich_domain_ind']:
                        ba_active_elements |= mark_elements(
                            mesh, enr_indicator, size)

                    # for el in fes.Elements():
                    #     if ba_active_elements[el.nr]:
                    #         gf_indicator.vec[el.nr] += 1
                    # #         N = len(el.dofs)

                    # #         Nstd = V.GetFE(ElementId(el)).ndof
                    # #         elmat = ipintegrator.CalcElementMatrix(el.GetFE(), el.GetTrafo())
                            
                    # #         important = [True if el.dofs[i] >=
                    # #                     0 else False for i in range(N)]
                    # #         factors = []
                    # #         for i in range(Nstd, N):
                    # #             if important[i]:
                    # #                 active = [j for j in range(i)
                    # #                         if important[j]]
                    # #                 factor = 1 - 2 * \
                    # #                     sum([elmat[i, j]**2/elmat[i, i] /
                    # #                         elmat[j, j] for j in active])
                    # #                 factor += sum([elmat[i, j]*elmat[i, k]*elmat[j, k]/elmat[i, i] /
                    # #                             elmat[j, j]/elmat[k, k] for j in active for k in active])
                    # #                 factor = sqrt(abs(factor))
                    # #                 factors.append(factor)
                    # #                 if (factor <= self.config['theta']):
                    # #                     important[i] = False
                    # #                     if el.dofs[i] >= 0:
                    # #                         ba_active_dofs[el.dofs[i]] = False
                else:
                    type = str('dg')   

                #Draw(gf_indicator, mesh, 'enriched')
                #input('')

                alpha = 10

                # # non-symmetric diffusion equation
                # diffusion = grad(u) * grad(v) * dy \
                #     + alpha * order ** 2 / h * jump_u * jump_v * dX \
                #     + (-mean_dudn * jump_v + mean_dvdn * jump_u) * dX \
                #     + (alpha * order ** 2/h * u * v * dS) \
                #     + (-n * grad(u) * v + n * grad(v) * u) * dS

                # symmetric diffusion equation
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
                acd = BilinearForm(fes, symmetric=False)
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
                gfu.vec.data = acd.mat.Inverse(
                    ba_active_dofs, inverse="pardiso") * f.vec

                gfu = gfu.components[0] + sum([gfu.components[i+1] * self.config['enrich_functions'][i]
                                                for i in range(len(self.config['enrich_functions']))])

                # error
                error = sqrt(Integrate(
                    (gfu-self.config['exact'])*(gfu-self.config['exact']), mesh, order= 20 + self.config['bonus_int']))

                Draw(gfu, mesh, 'test')
                input('')
                #netgen.gui.Snapshot(w=1000,h=500, filename="myimage_"+str(order)+".png")
                #input('')
                
                self.results.loc[len(self.results)] = [order, fes.ndof, size, error, self.config['bonus_int'], type]
                
                print('order:', order, 'DOFs:', fes.ndof, 'mesh:', size, "err:", error, 'type:', type)

        return self.results, self.alphas