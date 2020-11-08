import numpy as np
import sympy 
from scipy import linalg
import pandas as pd

from enrichment_proxy import *


"""
(Hybrid) Discontinuous Galerkin Method with Upwinding
"""

class Hybrid_Discontinuous_Galerkin():
    
    '''Specify default parameters if necessary'''
    config = {
    }
    
    def __init__(self, new_config={}):
        self.config.update(new_config)
        self.columns =  ['Order', 'Mesh Size', 'Error', 'Alpha', 'Bonus Int', 'Type']
        self.hdg_table = pd.DataFrame(columns=self.columns)
        self.ehdg_table = pd.DataFrame(columns=self.columns)      
  
    
    '''Solves the convection-diffusion equations by the enriched HDG 
    methods'''
    def _solveEHDG(self):
        self.ehdg_table.iloc[0:0]
        for order in self.config['order']:
            for size in self.config['mesh_size']:
                for alpha in self.config['alpha']:
                    for bonus_int in self.config['bonus_int_order']:
                        mesh = Mesh(unit_square.GenerateMesh(maxh=size))
                        condense = True
                        h = specialcf.mesh_size
                        n = specialcf.normal(mesh.dim)
                        dS = dx(element_boundary=True, bonus_intorder=bonus_int)
                        dy = dx(bonus_intorder=bonus_int)
                        V = L2(mesh, order=order)
                        F = FacetFESpace(mesh, order=order, dirichlet=".*")
                        Q = L2(mesh, order=0)
                        QF = FacetFESpace(mesh, order=0, dirichlet=".*")

                        Vlist = [V]
                        Vlist.append(F)
                        for enr_indicator in self.config['enrich_domain_ind']:
                            Vlist.append(mark_element(Q,mesh,enr_indicator,size))
                            Vlist.append(mark_element_bnd(Q, QF, mesh,enr_indicator, size))
                                
                        fes = FESpace(Vlist) # [V, F, Qx, QFx, Qy, QFy])
                        u = EnrichmentProxy_VOL(fes.TrialFunction(),self.config['enrich_functions'])
                        v = EnrichmentProxy_VOL(fes.TestFunction(), self.config['enrich_functions'])

                        uhat = EnrichmentProxy_FAC(fes.TrialFunction(),self.config['enrich_functions'])       
                        vhat = EnrichmentProxy_FAC(fes.TestFunction(), self.config['enrich_functions'])
                        

                        # # Checking linear dependence
                        # # Get active elements from the enrichment
                        ipintegrator = SymbolicBFI(u() * v(),bonus_intorder=20)
                        ba_active_elements = BitArray(mesh.ne)
                        #ba_active_elements[:] = True  # <-- todo (get this from before...)
                        
                        # # Get dofs
                        ba_active_dofs = BitArray(fes.FreeDofs())
                        ba_active_dofs[:] = fes.FreeDofs()
                        
                        ## apply algorithm on each element
                        for el in fes.Elements():
                            if ba_active_elements[el.nr]:
                                i = ElementId(el)
                                element = fes.GetFE(el)
                                elementstd = V.GetFE(i)
                                Nstd = elementstd.ndof
                                trafo = mesh.GetTrafo(i)
                                # Get element matrix 
                                elmat = ipintegrator.CalcElementMatrix(element, trafo)           
                                N = len(el.dofs)
                                important = [True  if el.dofs[i]>=0 else False for i in range(N)]
                                before_important = [True  if el.dofs[i]>=0 else False for i in range(N)]
                                
                                factors=[]
                                for i in range(Nstd,N):
                                  if important[i]:
                                    active = [j for j in range(i) if important[j]]
                                    #print(elmat[i,i])
                                    ### FIX (elmat entries are zero??)
                                    try:
                                        factor = 1 - 2 * sum([elmat[i,j]**2/elmat[i,i]/elmat[j,j] for j in active])
                                        factor += sum([elmat[i,j]*elmat[i,k]*elmat[j,k]/elmat[i,i]/elmat[j,j]/elmat[k,k] for j in active for k in active])
                                        factor = sqrt(abs(factor))
                                        factors.append(factor)
                                        #print(factor)
                                        if (factor < 1e-5):
                                            important[i] = False
                                            if el.dofs[i] >= 0:
                                                ba_active_dofs[el.dofs[i]] = False
                        
                                    except:
                                        "print nothing"
                        jump_u = u-uhat()
                        jump_v = v-vhat()
                        
                        diffusion = grad(u) * grad(v) * dy \
                        + alpha * order ** 2/h * jump_u * jump_v * dS \
                        + (-grad(u) * n * jump_v - grad(v) * n * jump_u) * dS
                        
                        # convection            
                        b = CoefficientFunction((self.config['beta'][0], self.config['beta'][1]))
                        uup = IfPos(b * n, u(), uhat())
                        
                        convection = -b * u * grad(v) * dy + b * n * uup * v * dS + IfPos( b * n, (uhat()-u)*vhat(), 0) * dS

                        acd = BilinearForm(fes,symmetric=False)
                        acd += self.config['epsilon'] *  diffusion + convection 

                        with TaskManager():
                            acd.Assemble()

                        #rhs
                        f = LinearForm(fes)
                        f += self.config['coeff'] * v * dy
                        
                        with TaskManager():
                            f.Assemble()
                    
                        gfu = GridFunction(fes)
                        
                        try:
                            gfu.vec.data = acd.mat.Inverse(ba_active_dofs,inverse="umfpack") * f.vec
                        except:
                            gfu.vec.data = acd.mat.Inverse(ba_active_dofs,inverse="sparsecholesky") * f.vec

                        gfu = gfu.components[0] + sum([gfu.components[2*i+2] * self.config['enrich_functions'][i] for i in range(len(self.config['enrich_functions']))])
                        
                        error = sqrt (Integrate ((gfu-self.config['exact'])*(gfu-self.config['exact']), mesh, order=100 + bonus_int))
                        self.ehdg_table.loc[len(self.ehdg_table)] = [order, size, error, alpha, bonus_int, 'ehdg']

                        print ('Order:', order, 'Mesh Size:', size , "L2-error:", error)
                
        return self.ehdg_table


    '''Solves the convection-diffusion equations by the standard HDG 
    methods'''
    def _solveHDG(self):
        self.hdg_table.iloc[0:0]
        for order in self.config['order']:
            for size in self.config['mesh_size']:
                for alpha in self.config['alpha']:
                    mesh = Mesh(unit_square.GenerateMesh(maxh=size))
                    V = L2(mesh, order=order)
                    F = FacetFESpace(mesh, order=order, dirichlet="bottom|left|right|top")

                    fes = FESpace([V,F])
                    u,uhat = fes.TrialFunction()
                    v,vhat = fes.TestFunction()

                    jump_u = u-uhat
                    jump_v = v-vhat

                    condense = True
                    h = specialcf.mesh_size
                    n = specialcf.normal(mesh.dim)

                    dS = dx(element_boundary=True)

                    # diffusion
                    diffusion = grad(u) * grad(v) * dx + \
                        alpha * order ** 2/h * jump_u * jump_v * dS + \
                        (-grad(u) * n * jump_v - grad(v) * n * jump_u) * dS

                    # convection            
                    b = CoefficientFunction((self.config['beta'][0], self.config['beta'][1]))
                    uup = IfPos(b * n, u, uhat)
                    convection = -b * u * grad(v) * dx + b * n * uup * jump_v * dS

                    acd = BilinearForm(fes, condense=condense)
                    acd += self.config['epsilon'] * diffusion + convection
                    acd.Assemble()
                    
                    #rhs
                    f = LinearForm(fes)
                    f += SymbolicLFI(self.config['coeff'] * v)
                    f.Assemble()

                    gfu = GridFunction(fes)    
                    if not condense:
                        inv = acd.mat.Inverse(fes.FreeDofs(), "umfpack")
                        gfu.vec.data = inv * f.vec
                    else:
                        f.vec.data += acd.harmonic_extension_trans * f.vec

                        inv = acd.mat.Inverse(fes.FreeDofs(True), "umfpack")
                        gfu.vec.data = inv * f.vec

                        gfu.vec.data += acd.harmonic_extension * gfu.vec
                        gfu.vec.data += acd.inner_solve * f.vec
                        
                    error = sqrt (Integrate ((gfu.components[0] - self.config['exact'])*(gfu.components[0]- self.config['exact']), mesh))

                    self.hdg_table.loc[len(self.hdg_table)] = [order, size, error, alpha, 0 , 'hdg']
                    
                    print ('Order:', order, 'Mesh Size:', size , "L2-error:", error)
        
        return self.hdg_table
