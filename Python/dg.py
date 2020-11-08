import numpy as np
import sympy 
from scipy import linalg
import pandas as pd

from enrichment_proxy import *


"""
Discontinuous Galerkin Method with Upwinding
"""

class Discontinous_Galerkin():
    
    ## Set Default parameters
    config = {

    }
    
    def __init__(self, new_config={}):
        self.config.update(new_config)
        self.columns =  ['Order', 'Mesh Size', 'Error', 'Alpha', 'Bonus Int', 'Type']
        self.dg_table = pd.DataFrame(columns=self.columns)
        self.edg_table = pd.DataFrame(columns=self.columns)

    
    '''Solves the convection-diffusion equations by the DG methods using upwinding'''
    def _solveEDG(self):
        self.edg_table.iloc[0:0]
        for order in self.config['order']:
            for size in self.config['mesh_size']:
                for alpha in self.config['alpha']:
                    for bonus_int in self.config['bonus_int_order']:
                        mesh = Mesh(unit_square.GenerateMesh(maxh=size))
                        V = L2(mesh, order=order, dgjumps=True)    
                        Q = L2(mesh, order=0)
    
                        Vlist = [V]
                            
                        for enr_indicator in self.config['enrich_domain_ind']:
                            Vlist.append(mark_dofs(Q,mesh,enr_indicator, size))
                        
                        fes = FESpace(Vlist, dgjumps = True)

                        u = EnrichmentProxy(fes.TrialFunction(), self.config['enrich_functions'])
                        v = EnrichmentProxy(fes.TestFunction(), self.config['enrich_functions'])


                        # # Checking linear dependence
                        # # Get active elements from the enrichment
                        ipintegrator = SymbolicBFI(u() * v(),bonus_intorder=bonus_int)
                        ba_active_elements = mark_elements(mesh,enr_indicator, size) #BitArray(mesh.ne)                        
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

                                # print("Nstd:",Nstd)
                                # print("el.dofs:",el.dofs)

                                trafo = mesh.GetTrafo(i)
                                # Get element matrix 
                                elmat = ipintegrator.CalcElementMatrix(element, trafo)           
                                N = len(el.dofs)
                                important = [True  if el.dofs[i]>=0 else False for i in range(N)]
                                before_important = [True  if el.dofs[i]>=0 else False for i in range(N)]
                                
                                factors=[]
                                for i in range(Nstd,N):
                                  # print(i,el.dofs[i])
                                  if important[i]:
                                    ba_active_dofs[el.dofs[i]] = False
                                    active = [j for j in range(i) if important[j]]
                                    factor = 1 - 2 * sum([elmat[i,j]**2/elmat[i,i]/elmat[j,j] for j in active])
                                    factor += sum([elmat[i,j]*elmat[i,k]*elmat[j,k]/elmat[i,i]/elmat[j,j]/elmat[k,k] for j in active for k in active])
                                    factor = sqrt(abs(factor))
                                    factors.append(factor)
                                    # print("factor:",factor)
                                    if (factor >= 0):
                                        important[i] = False
                                        if el.dofs[i] >= 0:
                                            ba_active_dofs[el.dofs[i]] = False

                                            

                        # print(ba_active_dofs)
                        # print("active dofs:",sum(ba_active_dofs))
                        # print("fes dofs:", fes.ndof)
                        jump_u = u-u.Other()
                        jump_v = v-v.Other()

                        n = specialcf.normal(2)
                        mean_dudn = 0.5 * n * (grad(u) + grad(u.Other()))
                        mean_dvdn = 0.5 * n * (grad(v) + grad(v.Other()))

                        h = specialcf.mesh_size
                        dy = dx(bonus_intorder = bonus_int)
                        dX = dx(skeleton=True, bonus_intorder= bonus_int)
                        dS = ds(skeleton=True, bonus_intorder= bonus_int)

                        # diffusion equation
                        diffusion = grad(u) * grad(v) * dy \
                            + alpha * order ** 2/ h * jump_u * jump_v * dX \
                            +(-mean_dudn * jump_v - mean_dvdn * jump_u) * dX \
                            +alpha * order ** 2/h * u * v * dS \
                            + (-n * grad(u) * v -n * grad(v) * u) * dS

                        # convection equation
                        b = CoefficientFunction((self.config['beta'][0], self.config['beta'][1]))
                        uup = IfPos(b * n, u(), u.Other()())
                        convection = -b * u * grad(v) * dy + b * n * uup * jump_v * dX

                        acd = BilinearForm(fes)
                        acd += self.config['epsilon'] * diffusion + convection
                        
                        with TaskManager():
                            acd.Assemble()

                        # rhs
                        f = LinearForm(fes)
                        f += self.config['coeff'] * v * dy
                        
                        with TaskManager():
                            f.Assemble()

                        gfu = GridFunction(fes, name="uDG")
                        
                        
                        # try:
                        gfu.vec.data = acd.mat.Inverse(ba_active_dofs,inverse="umfpack") * f.vec
                        # except:
                        #     gfu.vec.data = acd.mat.Inverse(ba_active_dofs,inverse="sparsecholesky") * f.vec
                        
                        gfu = gfu.components[0] + sum([gfu.components[i+1]*self.config['enrich_functions'][i] for i in range(len(self.config['enrich_functions']))])

                        #WebGuiDraw(gfu,mesh,"u")
                    
                        error = sqrt (Integrate ((gfu-self.config['exact'])*(gfu-self.config['exact']), mesh, order = 30 + bonus_int))
                        self.edg_table.loc[len(self.edg_table)] = [order, size, error, alpha, bonus_int, 'edg']

                        print ('Order:', order, 'Mesh Size:', size , "L2-error:", error)

        return self.edg_table


    '''Solves the convection-diffusion equations by the standard DG 
    methods'''
    def _solveDG(self):
        self.dg_table.iloc[0:0]
        for order in self.config['order']:
            for size in self.config['mesh_size']:
                for alpha in self.config['alpha']:
                    for bonus_int in self.config['bonus_int_order']:
                        mesh = Mesh(unit_square.GenerateMesh(maxh=size))
                        fes = L2(mesh, order=order, dgjumps=True)
                        u, v = fes.TnT()
    
                        # print("fes dofs:", fes.ndof)
                        jump_u = u-u.Other()
                        jump_v = v-v.Other()
                        n = specialcf.normal(2)
                        mean_dudn = 0.5 * n * (grad(u) + grad(u.Other()))
                        mean_dvdn = 0.5 * n * (grad(v) + grad(v.Other()))
    
                        h = specialcf.mesh_size
    
    
                        dy = dx(bonus_intorder = bonus_int)
                        dX = dx(skeleton=True, bonus_intorder= bonus_int)
                        dS = ds(skeleton=True, bonus_intorder= bonus_int)
                        
                        # diffusion
                        diffusion = grad(u) * grad(v) * dy \
                            +alpha * order ** 2/ h * jump_u * jump_v * dX \
                            +(-mean_dudn * jump_v - mean_dvdn * jump_u) * dX \
                            +alpha * order ** 2/h * u * v * dS \
                            + (-n * grad(u) * v -n * grad(v) * u) * dS
    
                        # convection
                        b = CoefficientFunction((self.config['beta'][0], self.config['beta'][1]))                    
                        uup = IfPos(b * n, u, u.Other())
                        convection = -b * u * grad(v) * dy + b * n * uup * jump_v * dX
                        
                        acd = BilinearForm(fes)
                        acd += self.config['epsilon']  * diffusion + convection
                        acd.Assemble()
                        
                        # rhs
                        f = LinearForm(fes)
                        f += self.config['coeff'] * v * dy
                        f.Assemble()
    
                        gfu = GridFunction(fes, name="uDG")  
                        gfu.vec.data = acd.mat.Inverse(freedofs=fes.FreeDofs(),inverse="umfpack") * f.vec
                        
                        error = sqrt (Integrate ((gfu-self.config['exact'])*(gfu-self.config['exact']), mesh, order = 30 + bonus_int))
                        self.dg_table.loc[len(self.dg_table)] = [order, size, error, alpha, bonus_int, 'dg']

                        print ('Order:', order, 'Mesh Size:', size , "L2-error:", error)
            return self.dg_table
