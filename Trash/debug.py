from enrichment_proxy import is_pos_def, check_symmetric
import numpy as np
from numpy.linalg.linalg import norm
import pandas as pd
import scipy.linalg
import scipy as sp
import numpy as np
from scipy.linalg import eigh


from enrichment_proxy import *


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
                        V = L2(mesh, order=0, dgjumps=True)
                        Q = L2(mesh, order=0)
                        Vlist = [V, Q]

                        for enr_indicator in self.config['enrich_domain_ind'] or []:
                            Vlist.append(mark_dofs(Q, mesh, enr_indicator, size))

                        fes = FESpace(Vlist)
                        #fes = L2(mesh, order=order, dgjumps=True)

                        #u, v = fes.TnT()

                        u = EnrichmentProxy(
                            fes.TrialFunction(), self.config['enrich_functions'])
                        v = EnrichmentProxy(
                            fes.TestFunction(), self.config['enrich_functions'])

                        n = specialcf.normal(2)
                        h = specialcf.mesh_size
                        

                        a_diff = SymbolicBFI(grad(u) * grad(v), bonus_intorder=bonus_int) 
                        #fee = SymbolicLFI(h**((-2-order)/2)* v(), bonus_intorder=bonus_int) 
                            
                        # mass
                        m = SymbolicBFI(h * (grad(u) * n)*(grad(v) * n), element_boundary=True, bonus_intorder=bonus_int)

                        for el in fes.Elements():
                            a_elmat = (a_diff.CalcElementMatrix(el.GetFE(),el.GetTrafo())).NumPy()
                                
                            #print(a_elmat)
                                
                            #f_elmat = (fee.CalcElementVector(el.GetFE(),el.GetTrafo())).NumPy()

                            # for i in range(len(f_elmat)):
                            #     for j in range(len(f_elmat)):
                            #         a_elmat[i,j] += f_elmat[i]* f_elmat[j]
                                
                            #m_elmat = (m.CalcElementMatrix(el.GetFE(),el.GetTrafo())).NumPy()
                            
                            #x = np.max(np.linalg.eig(np.linalg.inv(a_elmat)@m_elmat)[0])
                            print(a_elmat)
                            print(b_elmat)
                            print(check_symmetric(a_elmat))
                            print((a_elmat.transpose() == a_elmat).all())

                            ## Matrix is symmetric
                            #x = np.max(np.linalg.eigh(np.linalg.pinv(a_elmat)@m_elmat)[0])   
                            L = np.linalg.cholesky(a_elmat)
                            #x = np.max((sp.linalg.eigh(a_elmat,b=m_elmat))[0])
                            # eigvals, eigvecs = sp.linalg.eigh(a_elmat, m_elmat, eigvals_only=True)
                            # print(eigvals)
                            print(x)
            
            return self.results