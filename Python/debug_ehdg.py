import numpy as np
from numpy.linalg.linalg import norm
import pandas as pd
import scipy.sparse as spe
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

    '''Enriched Hybrid Discontinuous Galerkin Methods'''

    def _solveEHDG(self):
        self.results.iloc[0:0]
        self.alphas.iloc[0:0]
        for order in self.config['order']:
            for size in self.config['mesh_size']:
                mesh = Mesh(unit_square.GenerateMesh(maxh=size))
                V = L2(mesh, order=order)
                F = FacetFESpace(mesh, order=order, dirichlet=".*")
                Q = L2(mesh, order=0)
                QF = FacetFESpace(mesh, order=0, dirichlet=".*")

                Vlist = [V]
                Vlist.append(F)

                gf_indicator = GridFunction(L2(mesh))
                gf_indicator.vec[:] = 0.0

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
                        bonus_intorder=self.config['bonus_int'])
                dy = dx(bonus_intorder=self.config['bonus_int'])

                ba_active_dofs = BitArray(fes.FreeDofs())
                ba_active_dofs[:] = fes.FreeDofs()

                alpha_stab = GridFunction(L2(mesh))

                # # Checking linear dependence
                if len(self.config['enrich_functions']) > 0:
                    type = str('ehdg')
                    ipintegrator = SymbolicBFI(
                        u() * v() + uhat() * vhat(), bonus_intorder=self.config['bonus_int'], element_boundary=True)
                            
                    ba_active_elements = BitArray(mesh.ne)

                    for enr_indicator in self.config['enrich_domain_ind']:
                        ba_active_elements |= mark_elements(
                            mesh, enr_indicator, size)

                    for el in fes.Elements():
                        if ba_active_elements[el.nr]:
                            N = len(el.dofs)
                            Nstd = V.GetFE(ElementId(el)).ndof
                            elmat = ipintegrator.CalcElementMatrix(el.GetFE(), el.GetTrafo())
                            
                            important = [True if el.dofs[i] >= 0 else False for i in range(N)]
                            factors = []
                            for i in range(Nstd, N):
                                if important[i]:
                                    active = [j for j in range(i) if important[j]]
                                    try:
                                        factor = 1 - 2 * \
                                            sum([elmat[i, j]** 2/elmat[i, i] /
                                                elmat[j, j] for j in active])
                                        factor += sum([elmat[i, j]*elmat[i, k]*elmat[j, k]/elmat[i, i] /
                                                    elmat[j, j]/elmat[k, k] for j in active for k in active])
                                        factor = sqrt(abs(factor))
                                        factors.append(factor)
                                        if (factor <= 1e-5):
                                            important[i] = False
                                            if el.dofs[i] >= 0:
                                                ba_active_dofs[el.dofs[i]] = False
                                        else:
                                            gf_indicator.vec[el.nr] += 1
                                    except:
                                        pass
                else:
                    type = 'hdg'

                # Draw(gf_indicator,mesh,"gf_ind")
                # input('')                        
                # # stiffness matrix
                a_diff = SymbolicBFI(grad(u) * grad(v))

                f = SymbolicLFI(h**((-2-mesh.dim)/2) * v)

                # mass matrix
                m = SymbolicBFI(h * (grad(u) * n)*(grad(v) * n), element_boundary=True)

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
                        if dof > 0 and ba_active_dofs[dof]:
                            important.append(i)
                    
                    m_elmat = m_elmat[np.ix_(important,important)]
                    a_elmat = a_elmat[np.ix_(important,important)] 

                    # print(a_elmat)  
                    # input('')
                        
                    # x = np.max(sp.linalg.eig(m_elmat,b=a_elmat)[0])
                    x = np.max(np.linalg.eig(np.linalg.pinv(a_elmat)@m_elmat)[0])
                    # print(x)


                    alpha_stab.vec[el.nr] += 20 * x.real
                    #alp = 1 / 2 * np.sqrt(1 + x.real)
                    alp = x.real
                    self.alphas.loc[len(self.alphas)] = [alp, type, size, order]

                alpha = CoefficientFunction(alpha_stab)

                
                #ges = Compress(fes, ba_active_dofs)

                #alpha = 15
                jump_u = u-uhat()
                jump_v = v-vhat()

                # diffusion
                diffusion = grad(u) * grad(v) * dy + alpha * order ** 2/h * jump_u * \
                    jump_v * dS + (-grad(u) * n * jump_v -
                                    grad(v) * n * jump_u) * dS

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

                # # rhs
                f = LinearForm(fes)
                f += self.config['coeff'] * v * dy
                
                with TaskManager():
                    f.Assemble()

                # rows,cols,vals = acd.mat.COO()
                # A = spe.csr_matrix((vals,(rows,cols)))
                # print('third: ', np.linalg.cond(A.todense()))

                gfu = GridFunction(fes)
                gfu.components[1].Set(self.config['exact'],BND)
                f.vec.data -= acd.mat * gfu.vec
                gfu.vec.data += acd.mat.Inverse(
                    ba_active_dofs, inverse="pardiso") * f.vec

                gfu = gfu.components[0] + sum([gfu.components[2*i+2] * self.config['enrich_functions'][i]
                                                for i in range(len(self.config['enrich_functions']))])

                error = sqrt(Integrate(
                    (gfu-self.config['exact'])*(gfu-self.config['exact']), mesh, order= 50 + self.config['bonus_int']))

                self.results.loc[len(self.results)] = [order, fes.ndof, size, error, self.config['bonus_int'], type]
                
                Draw(gfu, mesh, 'test')
                input('')
                print('order:', order, 'DOFs:', fes.ndof, 'mesh:', size, "err:", error, 'type:', type)

        return self.results, self.alphas