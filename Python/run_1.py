# netgen
from numpy.lib.function_base import hanning
from netgen.geom2d import unit_square
from ngsolve import *
from ngsolve import grad as ngsolvegrad
from ngsolve.comp import ProxyFunction
from ngsolve import ngsglobals
ngsglobals.msg_level = 0
#from ngsolve.webgui import Draw as WebGuiDraw
#import netgen.gui

# packages
import numpy as np
from scipy import linalg
import pandas as pd

# modules
from enrichment_proxy import *
from debug import *
from utils import *



### Problem configuration
beta = (1, 2)
eps = 1e-4
ce = sqrt(5*eps)
p =  lambda x: (1-(exp(2*(x-0.5)/ce) - 1 ) / (exp(2*(x-0.5)/ce) + 1 ))
q =  lambda y: (1-(exp(2*(y-0.5)/ce) - 1 ) / (exp(2*(y-0.5)/ce) + 1 )) 

exact = p(x) * q(y)

coeff =  beta[1] * p(x) +  beta[0] * q(y)


config = {
    'order': [1],
    'beta': (beta[0],beta[1]),
    'mesh_size': np.logspace(0,-1,num=20),
    'mesh_size' :[0.0625],
    'epsilon': eps,
    'exact': exact,
    'coeff': coeff,
    'bonus_int' : 10,
    'theta': 1e-3,
    'enrich_functions':[p(x), q(y)],
    'enrich_domain_ind':[lambda x,y,h: x > 1 - h, lambda x,y,h: y > - 1 - h],
}


if __name__ == "__main__":

    # with enrichment
    CT = Convection_Diffusion(config)
    edg_table, alpha_edg  = CT._solveEDG()
    # ehdg_table, alpha_ehdg = CT._solveEHDG()
    
    # # # without enrichment
    # dict = {'enrich_functions': [], 'enrich_domain_ind': []}
    # config.update(dict)
    # CT = Convection_Diffusion(config)
    # # #dg_table, alpha_dg = CT._solveEDG()
    # hdg_table, alpha_hdg = CT._solveEHDG()

    # # # # # write to files
    # alphas = pd.concat([alpha_edg, alpha_dg])
    # alphas.to_csv('alphas_dg.csv')

    # # # # visualizations
    # #dg = pd.concat([hdg_table, ehdg_table])
    # # # #print(dg.to_latex(index=False)) 
    # hdg = pd.concat([hdg_table, ehdg_table])
    plot_error_mesh(edg_table) ## Plots with the mesh_size
    # plot_error_dof(hdg_table)